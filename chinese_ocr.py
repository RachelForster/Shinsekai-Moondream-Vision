"""
轻量中文 OCR（RapidOCR / ONNX Runtime），与 Moondream 推理互不干扰。

安装依赖: pip install rapidocr-onnxruntime
首次运行会自动下载 ONNX 模型到 ~/.rapidocr 目录。
"""

from __future__ import annotations

import io
import logging
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

_ocr: Any = None


def _get_ocr() -> Any:
    global _ocr
    if _ocr is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise RuntimeError(
                "请安装 rapidocr-onnxruntime：pip install rapidocr-onnxruntime"
            )
        _ocr = RapidOCR()
        logger.info("RapidOCR 引擎已初始化。")
    return _ocr


def ocr_image(image: Image.Image) -> str:
    """对 PIL Image 做中文 OCR，返回提取的纯文本（按行用 \\n 连接）。"""
    engine = _get_ocr()
    result, _ = engine(image)
    if not result:
        return ""
    lines: list[str] = []
    for (_box, text, _conf) in result:
        t = str(text).strip()
        if t:
            lines.append(t)
    return "\n".join(lines)


def ocr_png_bytes(png: bytes) -> str:
    """对 PNG 字节数据做中文 OCR。"""
    image = Image.open(io.BytesIO(png)).convert("RGB")
    return ocr_image(image)
