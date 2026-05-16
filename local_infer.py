from __future__ import annotations

import importlib
import io
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from PIL import Image

from plugins.moondream_vision.config_model import MoondreamVisionConfig

logger = logging.getLogger(__name__)

_moondream_hf_http_log_demoted: bool = False
_moondream_cuda_sdp_workaround_applied: bool = False


def _ensure_moondream_cuda_sdp_workaround() -> None:
    """PyTorch 2.11+ CUDA 上 Flash/mem-efficient SDPA 与 Moondream 的 sdpa 路径组合曾导致 logits 异常、解码乱码。

    固定使用 math 实现（通常与 2.7.x 行为更接近）。5080 等 GPU 上若仍正常可设
    EASYAI_MOONDREAM_CUDA_SDPA_MATH_ONLY=0 关闭；或在任意版本强制开启设 =1。
    """
    global _moondream_cuda_sdp_workaround_applied

    import torch

    if not torch.cuda.is_available() or _moondream_cuda_sdp_workaround_applied:
        return

    raw = os.environ.get("EASYAI_MOONDREAM_CUDA_SDPA_MATH_ONLY", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        _moondream_cuda_sdp_workaround_applied = True
        return

    want_math = raw in ("1", "true", "yes", "on")
    if not want_math and not raw:
        try:
            parts = torch.__version__.split("+")[0].split(".")
            major, minor = int(parts[0]), int(parts[1])
            want_math = major == 2 and minor >= 11
        except (ValueError, IndexError):
            want_math = False

    _moondream_cuda_sdp_workaround_applied = True
    if not want_math:
        return

    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info(
            "Moondream: CUDA SDPA 已改用 math 内核（缓解 PyTorch 2.11+ 解码乱码）。"
            "关闭请设 EASYAI_MOONDREAM_CUDA_SDPA_MATH_ONLY=0"
        )
    except Exception:
        logger.exception("Moondream: 设置 CUDA SDPA math 后备失败")


def _demote_hf_http_loggers_once() -> None:
    """打包/嵌入式常为 root/basicConfig=INFO，httpx 会把 Hub 每次 HEAD 都打出来；conda/IDE 下常默认更安静。"""
    global _moondream_hf_http_log_demoted
    if _moondream_hf_http_log_demoted:
        return
    if os.environ.get("EASYAI_MOONDREAM_HUB_VERBOSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        _moondream_hf_http_log_demoted = True
        return
    for name in (
        "httpx",
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
        "huggingface_hub",
        "urllib3",
        "urllib3.connectionpool",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    _moondream_hf_http_log_demoted = True


_model: Any = None
_model_key: tuple[str, str, str, str, str, str] | None = None
_lock = threading.Lock()
_model_loading = False
_loading_started_at: float = 0.0

# 所有 load + infer 仅在此时上执行，避免占用 Qt 主线程或 LLMWorker 的 QThread。
_infer_executor: ThreadPoolExecutor | None = None
_infer_exec_lock = threading.Lock()

# transformers 新版在 _finalize_model_loading 中用 ``all_tied_weights_keys``；vikhyatk/moondream2 等远端
# ``HfMoondream`` 仍只有 Legacy ``_tied_weights_keys``，会触发 AttributeError。仅在 from_pretrained 前后打补丁。
_compat_nn_module_attr_patch_depth = 0
_stashed_torch_nn_module_getattr: Any | None = None


def _legacy_tied_keys_as_mapping(legacy: Any) -> dict[str, Any]:
    """将远端模型上的 `_tied_weights_keys` 规范为可用于 ``missing_keys - .keys()`` 的 mapping。"""
    if isinstance(legacy, dict):
        return legacy
    if isinstance(legacy, (list, tuple)):
        first = legacy[0] if legacy else None
        if first is None:
            return {}
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            try:
                return dict(legacy)
            except (TypeError, ValueError):
                return {}
        if isinstance(first, str):
            # 常见：键名列表
            return {k: True for k in legacy if isinstance(k, str)}
        return {}
    return {}


def _push_torch_module_all_tied_weights_keys_compat() -> None:
    """支持嵌套 from_pretrained：深度为 0 时安装补丁，每层 push/pop 配对。"""
    global _compat_nn_module_attr_patch_depth, _stashed_torch_nn_module_getattr

    import torch.nn as nn

    if _compat_nn_module_attr_patch_depth == 0:
        orig = nn.Module.__getattr__

        def _wrapped(self: nn.Module, name: str) -> Any:
            if name == "all_tied_weights_keys":
                try:
                    return orig(self, name)
                except AttributeError:
                    pass
                return _legacy_tied_keys_as_mapping(
                    getattr(self, "_tied_weights_keys", None)
                )

            return orig(self, name)

        _stashed_torch_nn_module_getattr = orig
        nn.Module.__getattr__ = _wrapped  # type: ignore[assignment]

    _compat_nn_module_attr_patch_depth += 1


def _pop_torch_module_all_tied_weights_keys_compat() -> None:
    global _compat_nn_module_attr_patch_depth, _stashed_torch_nn_module_getattr

    import torch.nn as nn

    if _compat_nn_module_attr_patch_depth <= 0:
        return

    _compat_nn_module_attr_patch_depth -= 1

    if _compat_nn_module_attr_patch_depth != 0:
        return

    if _stashed_torch_nn_module_getattr is not None:
        nn.Module.__getattr__ = _stashed_torch_nn_module_getattr  # type: ignore[assignment]

    _stashed_torch_nn_module_getattr = None


# Moondream2 的 vision 用自定义 F.linear；bitsandbytes 量化后权重为 int8，与 float 激活在 F.linear 里不兼容，须跳过量化。
_MOONDREAM_BNB_SKIP_MODULES: tuple[str, ...] = ("vision",)


def _moondream_inner(model: Any) -> Any | None:
    """定位带 vision.patch_emb 的 Moondream 核心模块（可能被 CausalLM 包装）。"""
    cur: Any = model
    for _ in range(12):
        vis = getattr(cur, "vision", None)
        if vis is not None:
            try:
                if "patch_emb" in vis:
                    return cur
            except TypeError:
                pass
        nxt = getattr(cur, "model", None) or getattr(cur, "base_model", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return None


def _moondream_vision_sample_float_dtypes(vision: Any) -> set[Any]:
    """抽样 vision 关键层的浮点权重 dtype；用于检测混用 float16/bfloat16/float32。"""
    import torch

    out: set[Any] = set()

    def add_mod(m: Any) -> None:
        try:
            w = getattr(m, "weight", None)
            if w is not None and isinstance(w, torch.Tensor):
                d = w.dtype
                if d in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                    torch.float64,
                ):
                    out.add(d)
        except Exception:
            pass

    try:
        add_mod(vision["patch_emb"])
    except Exception:
        return out
    try:
        b0 = vision["blocks"][0]
        for k in ("ln1", "ln2"):
            if k in b0:
                add_mod(b0[k])
        add_mod(vision["post_ln"])
    except Exception:
        pass
    return out


def _unify_moondream_vision_dtype_if_needed(inner: Any) -> None:
    """BNB / device_map 偶发使 vision 各层 dtype 不一致，首层 layer_norm 即报错，统一为 float32。"""
    import torch

    vis = getattr(inner, "vision", None)
    if vis is None:
        return
    dtypes = _moondream_vision_sample_float_dtypes(vis)
    if len(dtypes) <= 1:
        return
    try:
        vis.float()
        logger.info(
            "Moondream: vision 内层浮点 dtype 不一致 %s，已 .float() 统一。",
            sorted(str(d) for d in dtypes),
        )
    except Exception:
        logger.exception("Moondream: vision.float() 未成功")


def _moondream_vision_skip_prepare_crops_patch(vision: Any) -> bool:
    """与 Hub 一致：vision 全为 bfloat16 时沿用官方 prepare_crops（bf16 归一化）。"""
    import torch

    try:
        if vision["patch_emb"].weight.dtype != torch.bfloat16:
            return False
        if vision["blocks"][0]["ln1"].weight.dtype != torch.bfloat16:
            return False
        if vision["post_ln"].weight.dtype != torch.bfloat16:
            return False
    except Exception:
        return False
    return True


def _moondream_torch_dtype_is_float(d: Any) -> bool:
    """仅浮点可作为 vision 激活 / crop 最终 dtype；整型（含 Char/int8、uint8）会导致 div_ 归一化报错。"""
    import torch

    try:
        torch.finfo(d)
        return True
    except (TypeError, RuntimeError):
        return False


def _vision_prepare_cast_dtype(vision: Any) -> Any:
    """以首层 LayerNorm 权重 dtype 为准（报错栈多在 ln1），否则回退 patch_emb / float32。"""
    import torch

    try:
        d = vision["blocks"][0]["ln1"].weight.dtype
        if _moondream_torch_dtype_is_float(d):
            return d
    except Exception:
        pass
    try:
        d = vision["patch_emb"].weight.dtype
        if _moondream_torch_dtype_is_float(d):
            return d
    except Exception:
        pass
    return torch.float32


def _patch_moondream_prepare_crops(inner: Any) -> None:
    """上游 prepare_crops 固定用 bfloat16；其余情况按 vision 实际 dtype 输出 crop 张量。"""
    try:
        import numpy as np
        import torch
        from PIL import Image as PILImage
    except ImportError:
        return
    try:
        _ = inner.vision["patch_emb"]
    except Exception:
        return
    if _moondream_vision_skip_prepare_crops_patch(inner.vision):
        return
    mod_name = inner.__class__.__module__
    if not mod_name or "transformers_modules" not in mod_name:
        return
    pkg = mod_name.rsplit(".", 1)[0]
    try:
        image_crops_mod = importlib.import_module(f"{pkg}.image_crops")
        vision_mod = importlib.import_module(f"{pkg}.vision")
    except Exception as e:
        logger.warning("Moondream: 无法 patch prepare_crops（import 失败）: %s", e)
        return

    def prepare_crops(
        image: PILImage.Image,
        config: Any,
        device: Any,
    ) -> tuple[Any, Any]:
        np_image = np.array(image.convert("RGB"))
        overlap_crops = image_crops_mod.overlap_crop_image(
            np_image,
            max_crops=config.max_crops,
            overlap_margin=config.overlap_margin,
        )
        all_crops = overlap_crops["crops"]
        all_crops = np.ascontiguousarray(
            np.transpose(all_crops, (0, 3, 1, 2))
        )
        # overlap_crop 多为 uint8：必须在浮点上做 div/sub，否则 .div_(255.0) 会报
        # 「result type Float can't be cast to the desired output type Char」。
        cast_dtype = _vision_prepare_cast_dtype(inner.vision)
        if not _moondream_torch_dtype_is_float(cast_dtype):
            cast_dtype = torch.float32
        tensors = torch.from_numpy(all_crops).float().to(device=device)
        tensors = tensors.div_(255.0).sub_(0.5).div_(0.5)
        if tensors.dtype != cast_dtype:
            tensors = tensors.to(dtype=cast_dtype)
        return tensors, overlap_crops["tiling"]

    vision_mod.prepare_crops = prepare_crops
    # moondream.py 里 `from .vision import prepare_crops` 已拷贝函数引用，仅改 vision 模块无效。
    try:
        moondream_mod = importlib.import_module(f"{pkg}.moondream")
        moondream_mod.prepare_crops = prepare_crops
    except Exception as e:
        logger.warning("Moondream: 无法将 prepare_crops 同步到 moondream 模块: %s", e)
    logger.info(
        "Moondream: 已 patch prepare_crops（crop cast_dtype=%s）",
        _vision_prepare_cast_dtype(inner.vision),
    )


def _bitsandbytes_quant_config(mode: str) -> Any:
    import torch
    from transformers import BitsAndBytesConfig

    if mode == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=list(_MOONDREAM_BNB_SKIP_MODULES),
        )
    if mode == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            ),
            bnb_4bit_use_double_quant=True,
            # transformers bitsandbytes 集成在 4bit 下仍用该字段跳过指定子模块的 Linear4bit 替换
            llm_int8_skip_modules=list(_MOONDREAM_BNB_SKIP_MODULES),
        )
    raise ValueError(f"unknown quantization mode: {mode!r}")


def _torch_load_kw(device: str) -> dict[str, Any]:
    import torch

    pref = (device or "auto").strip().lower()
    # Moondream2 的自定义 vision 模块在 float16 下会在 layer_norm 等处出现
    # 「expected scalar type Float but found Half」。非量化路径统一用 float32 加载。
    if pref == "auto":
        if torch.cuda.is_available():
            return {"device_map": "cuda", "dtype": torch.float32}
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {"device_map": "mps", "dtype": torch.float32}
        return {"device_map": {"": "cpu"}, "dtype": torch.float32}
    if pref == "cuda":
        return {"device_map": "cuda", "dtype": torch.float32}
    if pref == "mps":
        return {"device_map": "mps", "dtype": torch.float32}
    return {"device_map": {"": "cpu"}, "dtype": torch.float32}


def _model_load_fp_kw(
    cfg: MoondreamVisionConfig,
) -> dict[str, Any]:
    """from_pretrained 中除 trust_remote_code / revision / cache_dir 外的关键字。"""
    import torch

    q = (cfg.quantization or "none").strip().lower()
    if q in ("int8", "int4"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "INT8 / INT4 量化需要 NVIDIA GPU 与 CUDA。"
                "当前未检测到 CUDA，请将「量化」设为「无」或换用支持 CUDA 的环境。"
            )
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "INT8 / INT4 量化需要安装 bitsandbytes："
                "pip install bitsandbytes"
            ) from e
        # 勿传顶层 modules_to_not_convert：trust_remote_code 的 HfMoondream 会把它原样传入 __init__ 并报错。
        return {
            "device_map": "auto",
            "quantization_config": _bitsandbytes_quant_config(q),
        }
    return _torch_load_kw((cfg.device or "auto").strip().lower())


def _model_cache_key(cfg: MoondreamVisionConfig) -> tuple[str, str, str, str, str, str]:
    q = (cfg.quantization or "none").strip().lower()
    # 与 _model_load_fp_kw 的非量化 dtype 策略一致；变更时需 bump 以丢弃旧缓存。
    dtype_tag = "bnb_skip_vision" if q in ("int8", "int4") else "fp32_md_v6"
    return (
        (cfg.model_id or "").strip() or "vikhyatk/moondream2",
        (cfg.revision or "").strip(),
        (cfg.cache_dir or "").strip(),
        (cfg.device or "auto").strip().lower(),
        q,
        dtype_tag,
    )


def get_model(cfg: MoondreamVisionConfig) -> Any:
    """懒加载 HuggingFace 上的 Moondream2（首次调用会下载权重到本地缓存）。"""
    global _model, _model_key
    key = _model_cache_key(cfg)
    with _lock:
        if _model is not None and _model_key == key:
            return _model
        if _model is not None:
            try:
                del _model
            except Exception:
                pass
            _model = None
            _model_key = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            chain: BaseException | None = e
            missing_filecmp = False
            while chain is not None:
                if isinstance(chain, ModuleNotFoundError) and chain.name == "filecmp":
                    missing_filecmp = True
                    break
                chain = chain.__cause__
            if getattr(sys, "frozen", False) and missing_filecmp:
                raise RuntimeError(
                    "当前为打包 exe，主程序未包含标准库模块 filecmp（transformers 需要）。"
                    "请用最新构建脚本重新打包主程序（已加入 PyInstaller hiddenimport filecmp），"
                    "或使用源码/Python 解释器运行；不是 requirements.txt 未安装。"
                ) from e
            raise RuntimeError(
                "请安装插件依赖：pip install -r plugins/moondream_vision/requirements.txt"
            ) from e

        mid = key[0]
        revision = key[1]
        cache_dir = key[2] or None
        load_kw = _model_load_fp_kw(cfg)
        fp_kw: dict[str, Any] = {
            "trust_remote_code": True,
            **load_kw,
        }
        if revision:
            fp_kw["revision"] = revision
        if cache_dir:
            fp_kw["cache_dir"] = cache_dir

        _demote_hf_http_loggers_once()

        from plugins.moondream_vision.ui_busy import _post_busy

        _post_busy("Moondream: 正在加载模型（首次需下载，请稍候）…", 0.0)
        logger.info("Moondream 正在加载模型 %s（如需下载请稍候）…", mid)
        _push_torch_module_all_tied_weights_keys_compat()
        try:
            _model = AutoModelForCausalLM.from_pretrained(mid, **fp_kw)
        finally:
            _pop_torch_module_all_tied_weights_keys_compat()
        quant = key[4]
        if quant not in ("int8", "int4"):
            try:
                _model.float()
            except Exception:
                logger.exception("Moondream: model.float() 未完全成功，将仍尝试推理")
        try:
            _model.eval()
        except Exception:
            pass
        inner = _moondream_inner(_model)
        if inner is not None:
            _unify_moondream_vision_dtype_if_needed(inner)
            _patch_moondream_prepare_crops(inner)
        else:
            logger.warning(
                "Moondream: 未找到 vision 子模块，若推理报 dtype 错误请检查 transformers 版本与模型结构。"
            )
        _model_key = key
        logger.info("Moondream 模型已就绪。")
        _post_busy("Moondream: reading screen…", 0.0)
        return _model


def is_tool_ready() -> bool:
    """模型是否已加载完毕，可在不阻塞的情况下推理。"""
    return _model is not None


def loading_status_message() -> str:
    """动态生成模型加载状态消息，包含已等待时长。"""
    if _loading_started_at > 0:
        elapsed = int(time.time() - _loading_started_at)
        if elapsed < 60:
            return (
                f"Moondream 视觉模型仍在加载中（已等待 {elapsed} 秒），"
                "首次需从 HuggingFace 下载模型约 2-10 分钟。"
                "请直接告诉用户「视觉模型正在加载，请稍等几分钟」，不要重复调用本工具或任何 moondream_* 工具。"
            )
        else:
            minutes = elapsed // 60
            seconds = elapsed % 60
            return (
                f"Moondream 视觉模型仍在加载中（已等待 {minutes} 分 {seconds} 秒），"
                "仍在下载/加载模型到显存。请告诉用户再等 1-3 分钟，不要重复调用本工具。"
            )
    return (
        "Moondream 视觉模型正在后台加载（首次需从 HuggingFace 下载，约 2-10 分钟）。"
        "请直接告诉用户「视觉模型正在初始化，请稍等几分钟」，不要重复调用本工具或任何 moondream_* 工具。"
    )


def start_preload_model(cfg: MoondreamVisionConfig) -> None:
    """在后台线程启动模型加载，不阻塞调用方。已加载或正在加载时无操作。"""
    global _model_loading, _loading_started_at
    with _lock:
        if _model is not None or _model_loading:
            return
        _model_loading = True
        _loading_started_at = time.time()

    def _load() -> None:
        global _model_loading
        try:
            cfg_id = cfg.model_id if cfg else "unknown"
            print(f"[moondream] 后台线程开始加载模型 {cfg_id}…")
            get_model(cfg)
            print("[moondream] 后台加载完成，视觉模型已就绪")
        except Exception:
            print("[moondream] 后台加载失败！详见日志")
            logger.exception("Moondream 后台预加载失败")
        else:
            # 加载成功 → 通知宿主清除冷却 + 推送聊天通知
            try:
                from sdk.tool_registry import notify_tool_ready
                notify_tool_ready("vision", "视觉模型已就绪，可以使用识屏功能了。")
            except Exception:
                logger.exception("moondream 就绪通知失败")
        finally:
            with _lock:
                _model_loading = False

    print("[moondream] 启动后台加载线程…")
    t = threading.Thread(target=_load, name="moondream-preloader", daemon=True)
    t.start()
    logger.info("Moondream 后台预加载线程已启动")


def _maybe_downscale_infer_image(
    image: Image.Image, infer_max_side: int
) -> Image.Image:
    """将截图较长边限制在 infer_max_side；0 表示不缩放。"""
    cap = int(infer_max_side)
    if cap <= 0:
        return image
    w, h = image.size
    side = max(w, h)
    if side <= cap:
        return image
    scale = cap / float(side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return image.resize((nw, nh), Image.Resampling.LANCZOS)


def _ensure_infer_executor() -> ThreadPoolExecutor:
    global _infer_executor
    with _infer_exec_lock:
        if _infer_executor is None:
            _infer_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="moondream_infer",
            )
        return _infer_executor


def _infer_screen_png_worker(
    png: bytes, question: str, cfg: MoondreamVisionConfig
) -> str:
    """在专用线程中执行：get_model + query（可能含首次下载权重）。"""
    from contextlib import nullcontext

    import torch

    model = get_model(cfg)
    image = Image.open(io.BytesIO(png)).convert("RGB")
    ow, oh = image.size
    image = _maybe_downscale_infer_image(image, cfg.infer_max_side)
    rw, rh = image.size
    if (rw, rh) != (ow, oh):
        logger.info(
            "Moondream 推理前已缩放截图 %dx%d → %dx%d（infer_max_side=%d）",
            ow,
            oh,
            rw,
            rh,
            int(cfg.infer_max_side),
        )
    text_q = (question or "").strip() or "Briefly describe the visible screen in English for a chat assistant."
    dev = next(model.parameters()).device
    if dev.type == "cuda":
        _ensure_moondream_cuda_sdp_workaround()
        amp_ctx = torch.autocast(device_type="cuda", enabled=False)
    elif dev.type == "mps":
        try:
            amp_ctx = torch.autocast(device_type="mps", enabled=False)
        except (TypeError, ValueError, RuntimeError):
            amp_ctx = nullcontext()
    else:
        amp_ctx = nullcontext()
    hint = ""
    if dev.type == "cpu":
        # 大图 + CPU 才真正可能「很久」；你已缩到 512 边量级时十多秒很常见，勿误导为必达数分钟。
        mp = (rw * rh) / 1_000_000.0
        long_side = max(rw, rh)
        if long_side <= 640 or mp <= 0.22:
            hint = " （CPU·小图：通常在数十秒内。）"
        elif long_side <= 1280 or mp <= 1.8:
            hint = (
                " （CPU·中等图：可能需要一两分钟量级；仍可尝试再「降低」推理输入最长边或使用 CUDA。）"
            )
        else:
            hint = (
                " （CPU·大图：单次 query 可达数分钟以上；请将「推理输入最长边」"
                "调小（如 896～1280）或启用 CUDA。）"
            )
    logger.info(
        "Moondream 开始 query（device=%s %s，图 %dx%d）…%s",
        dev.type,
        dev,
        rw,
        rh,
        hint,
    )
    t0 = time.monotonic()
    try:
        with torch.inference_mode():
            with amp_ctx:
                out = model.query(image, text_q)
    finally:
        elapsed = time.monotonic() - t0
        logger.info("Moondream query 结束，用时 %.1f s。", elapsed)
    if isinstance(out, dict):
        ans = out.get("answer")
        if isinstance(ans, str) and ans.strip():
            return ans.strip()
    return str(out)[:4000]


_OCR_PROMPT = (
    "Read all visible text in this screenshot. "
    "Output only the exact text, preserving line breaks. "
    "If there is no text, reply with an empty string. "
    "Do not describe the image or add commentary."
)


def ocr_screen_png(png: bytes, cfg: MoondreamVisionConfig) -> str:
    """OCR-only shortcut: reuses the same Moondream model with a text-extraction prompt."""
    return infer_screen_png(png, _OCR_PROMPT, cfg)


def infer_screen_png(png: bytes, question: str, cfg: MoondreamVisionConfig) -> str:
    """将加载与推理派发到单线程池，调用方线程（含 UI / QThread）仅等待结果。"""
    ex = _ensure_infer_executor()
    fut = ex.submit(_infer_screen_png_worker, png, question, cfg)
    return fut.result()


def _release_weights_sync() -> None:
    global _model, _model_key
    with _lock:
        _model = None
        _model_key = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def unload_model() -> None:
    """在推理线程上释放权重，避免与正在执行的 infer 竞态。"""
    ex: ThreadPoolExecutor | None = None
    with _infer_exec_lock:
        ex = _infer_executor
    if ex is not None:
        try:
            ex.submit(_release_weights_sync).result(timeout=120.0)
        except Exception:
            logger.exception("Moondream: 提交释放权重任务失败或超时")
    else:
        _release_weights_sync()


def shutdown() -> None:
    """释放模型权重并关闭推理线程池。"""
    global _infer_executor
    unload_model()
    with _infer_exec_lock:
        if _infer_executor is not None:
            _infer_executor.shutdown(wait=False)
            _infer_executor = None
