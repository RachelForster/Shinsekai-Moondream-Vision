"""
Moondream 视屏能力暴露给 LLM 的 function-calling 工具。

模块被 ``plugin`` 导入时登记到 :mod:`sdk.tool_registry`；
宿主在 ``ensure_plugins_loaded`` 时统一注入 :class:`~llm.tools.tool_manager.ToolManager`。
"""

from __future__ import annotations

import logging
from typing import Any

from sdk.logging.timing import tracker
from sdk.tool_registry import ToolNotReady, tool

logger = logging.getLogger(__name__)

VISION_TOOL_GROUP = "vision"

@tool(
    name="moondream_query_screen",
    description=(
        "Capture the given monitor and answer your question using the configured vision adapter. "
        "Use when the user needs on-screen facts (UI text, errors, URLs, window contents). "
        "Pass question: a clear instruction in English, e.g. 'What error text is shown in the dialog?' "
        "Optional monitor_index: mss monitor index; default -1 uses the plugin setting; 0 = virtual full desktop, 1 = primary. "
        "When the local Moondream backend is selected, the first call may download/load weights. "
        "If you get status:'loading', follow the message instruction and tell the user — do NOT retry this tool or any moondream_* tool."
    ),
    group=VISION_TOOL_GROUP,
)
def moondream_query_screen(question: str, monitor_index: int = -1) -> dict[str, Any]:
    """
    Answer ``question`` from a fresh screenshot (English instructions work best).
    """
    q = (question or "").strip()
    if not q:
        return {"error": "question must not be empty: say what to read from the screen (English recommended)."}
    # 自动追加简短回复要求，防止模型啰嗦
    if "answer in" not in q.lower() and "one word" not in q.lower():
        q = q + " Answer in under 15 words."

    try:
        from plugins.moondream_vision.capture_infer import grab_screen_png
        from plugins.moondream_vision.config_model import load_config
        from plugins.moondream_vision.screen_infer import infer_screen_png, prepare_tool
        from plugins.moondream_vision import runtime
    except ImportError as e:
        return {"error": f"读屏插件依赖未就绪: {e}"}

    try:
        cfg_path = runtime.plugin_config_path()
    except RuntimeError:
        return {
            "error": "读屏插件尚未完成初始化。请先启动主程序并确保识屏插件已加载。",
        }

    cfg = load_config(cfg_path)
    mi = int(monitor_index)
    if mi >= 0:
        cfg.monitor_index = mi
    cfg.clamp()

    prepare_tool(cfg)

    try:
        from plugins.moondream_vision.ui_busy import moondream_busy

        with moondream_busy(ok_message="识屏完成"):
            with tracker.track("moondream query_screen"):
                png = grab_screen_png(cfg.monitor_index)
                from plugins.moondream_vision.runtime import _mask_chat_window
                png = _mask_chat_window(png)
                text = infer_screen_png(png, q, cfg)
    except Exception as e:
        logger.exception("moondream_query_screen 推理失败")
        return {"error": str(e)}

    return {
        "answer": text,
        "monitor_index": int(cfg.monitor_index),
    }


@tool(
    name="moondream_ocr_screen",
    description=(
        "Extract all visible text from the given monitor using Chinese OCR (RapidOCR) "
        "or the configured vision adapter as fallback. "
        "Returns the exact on-screen text, preserving line breaks. "
        "Use when the user needs to read text from the screen (error messages, code, documents, web pages). "
        "Optional monitor_index: mss monitor index; default -1 uses the plugin setting. "
        "NOTE: first call may return status:'loading'. If so, follow the message — do NOT retry any moondream_* tool."
    ),
    group=VISION_TOOL_GROUP,
)
def moondream_ocr_screen(monitor_index: int = -1) -> dict[str, Any]:
    """OCR extraction from a fresh screenshot — prefers RapidOCR for Chinese accuracy."""
    try:
        from plugins.moondream_vision.capture_infer import grab_screen_png
        from plugins.moondream_vision.config_model import load_config
        from plugins.moondream_vision.screen_infer import ocr_screen_png
        from plugins.moondream_vision import runtime
    except ImportError as e:
        return {"error": f"读屏插件依赖未就绪: {e}"}

    try:
        cfg_path = runtime.plugin_config_path()
    except RuntimeError:
        return {
            "error": "读屏插件尚未完成初始化。请先启动主程序并确保识屏插件已加载。",
        }

    cfg = load_config(cfg_path)
    mi = int(monitor_index)
    if mi >= 0:
        cfg.monitor_index = mi
    cfg.clamp()

    try:
        from plugins.moondream_vision.ui_busy import moondream_busy

        with moondream_busy(ok_message="识屏完成"):
            with tracker.track("moondream ocr_screen"):
                png = grab_screen_png(cfg.monitor_index)
                text, engine = ocr_screen_png(png, cfg)
    except ToolNotReady:
        raise
    except Exception as e:
        logger.exception("moondream_ocr_screen 推理失败")
        return {"error": str(e)}

    return {
        "text": text,
        "monitor_index": int(cfg.monitor_index),
        "engine": engine,
    }
