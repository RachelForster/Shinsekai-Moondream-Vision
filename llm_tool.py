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
        "Capture the given monitor and answer your question using the local Moondream2 vision model. "
        "Use when the user needs on-screen facts (UI text, errors, URLs, window contents). "
        "Pass question: a clear instruction in English, e.g. 'What error text is shown in the dialog?' "
        "Optional monitor_index: mss monitor index; default -1 uses the plugin setting; 0 = virtual full desktop, 1 = primary. "
        "NOTE: first call may trigger model download/load (2-10 min). If you get status:'loading', follow the message instruction and tell the user — do NOT retry this tool or any moondream_* tool."
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

    try:
        from plugins.moondream_vision.capture_infer import grab_screen_png
        from plugins.moondream_vision.config_model import load_config
        from plugins.moondream_vision.local_infer import infer_screen_png, is_tool_ready, start_preload_model, loading_status_message
        from plugins.moondream_vision import runtime
    except ImportError as e:
        return {"error": f"Moondream 插件依赖未就绪: {e}"}

    try:
        cfg_path = runtime.plugin_config_path()
    except RuntimeError:
        return {
            "error": "Moondream 尚未完成初始化。请先启动主程序并确保 Moondream 识屏插件已加载。",
        }

    cfg = load_config(cfg_path)
    mi = int(monitor_index)
    if mi >= 0:
        cfg.monitor_index = mi
    cfg.clamp()

    if not is_tool_ready():
        start_preload_model(cfg)
        raise ToolNotReady(loading_status_message())

    try:
        from plugins.moondream_vision.ui_busy import moondream_busy

        with moondream_busy(ok_message="Moondream: 识屏完成"):
            with tracker.track("moondream query_screen"):
                png = grab_screen_png(cfg.monitor_index)
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
        "or Moondream2 as fallback. "
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
        from plugins.moondream_vision.local_infer import ocr_screen_png, is_tool_ready, start_preload_model, loading_status_message
        from plugins.moondream_vision import runtime
    except ImportError as e:
        return {"error": f"Moondream 插件依赖未就绪: {e}"}

    try:
        cfg_path = runtime.plugin_config_path()
    except RuntimeError:
        return {
            "error": "Moondream 尚未完成初始化。请先启动主程序并确保 Moondream 识屏插件已加载。",
        }

    cfg = load_config(cfg_path)
    mi = int(monitor_index)
    if mi >= 0:
        cfg.monitor_index = mi
    cfg.clamp()

    if not is_tool_ready():
        start_preload_model(cfg)
        raise ToolNotReady(loading_status_message())

    try:
        from plugins.moondream_vision.ui_busy import moondream_busy

        with moondream_busy(ok_message="Moondream: 识屏完成"):
            with tracker.track("moondream ocr_screen"):
                png = grab_screen_png(cfg.monitor_index)
                try:
                    from plugins.moondream_vision.chinese_ocr import ocr_png_bytes
                    text = ocr_png_bytes(png)
                    engine = "rapidocr"
                except (ImportError, RuntimeError):
                    text = ocr_screen_png(png, cfg)
                    engine = "moondream"
    except Exception as e:
        logger.exception("moondream_ocr_screen 推理失败")
        return {"error": str(e)}

    return {
        "text": text,
        "monitor_index": int(cfg.monitor_index),
        "engine": engine,
    }
