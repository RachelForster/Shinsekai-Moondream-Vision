"""Route screen understanding through the selected backend without loading local weights."""

from __future__ import annotations

import io
import sys

from plugins.moondream_vision.config_model import MoondreamVisionConfig


_OCR_PROMPT = (
    "Read all visible text in this screenshot. Output only the exact text, "
    "preserving line breaks. If there is no text, reply with an empty string. "
    "Do not describe the image or add commentary."
)


def prepare_tool(cfg: MoondreamVisionConfig) -> None:
    """Only the explicitly selected local model needs background preloading."""
    if cfg.adapter != "moondream":
        return
    from sdk.tool_registry import ToolNotReady
    from plugins.moondream_vision.local_infer import (
        is_tool_ready,
        loading_status_message,
        start_preload_model,
    )

    if not is_tool_ready():
        start_preload_model(cfg)
        raise ToolNotReady(loading_status_message())


def _resize_png(png: bytes, max_side: int) -> bytes:
    if max_side <= 0:
        return png
    from PIL import Image

    with Image.open(io.BytesIO(png)) as image:
        width, height = image.size
        if max(width, height) <= max_side:
            return png
        scale = max_side / max(width, height)
        image = image.convert("RGB").resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


def infer_screen_png(png: bytes, question: str, cfg: MoondreamVisionConfig) -> str:
    if cfg.adapter == "moondream":
        from plugins.moondream_vision.local_infer import infer_screen_png as local_infer

        return local_infer(png, question, cfg)

    from ai.vision import configured_vision_available, configured_vision_manager

    if not configured_vision_available():
        raise RuntimeError("视觉适配器尚未就绪，请在 AI 服务设置中配置可用的视觉服务。")
    prompt = (question or "").strip() or "Briefly describe the visible screen for a chat assistant."
    return configured_vision_manager().describe(_resize_png(png, cfg.infer_max_side), prompt)


def ocr_screen_png(png: bytes, cfg: MoondreamVisionConfig) -> tuple[str, str]:
    try:
        from plugins.moondream_vision.chinese_ocr import ocr_png_bytes

        return ocr_png_bytes(png), "rapidocr"
    except (ImportError, RuntimeError):
        prepare_tool(cfg)
        engine = "moondream" if cfg.adapter == "moondream" else "vision_adapter"
        return infer_screen_png(png, _OCR_PROMPT, cfg), engine


def shutdown() -> None:
    local_infer = sys.modules.get("plugins.moondream_vision.local_infer")
    if local_infer is not None:
        local_infer.shutdown()
