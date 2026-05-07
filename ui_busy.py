"""从 Moondream 后台线程经 ChatUIContext 安全地显示 / 收起聊天窗 BusyBar。"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

_DEFAULT_MESSAGE = "Moondream: reading screen…"


def _post_busy(text: str, duration_seconds: float = 0.0) -> None:
    try:
        from ui.chat_ui.context import try_get_chat_ui_context

        ctx = try_get_chat_ui_context()
        if ctx is not None:
            ctx.set_busy_bar(text, float(duration_seconds))
    except Exception:
        logger.debug("moondream busy bar: ChatUIContext 不可用", exc_info=True)


def _hide_busy() -> None:
    try:
        from ui.chat_ui.context import try_get_chat_ui_context

        ctx = try_get_chat_ui_context()
        if ctx is not None:
            ctx.hide_busy_bar()
    except Exception:
        pass


@contextmanager
def moondream_busy(message: str | None = None, *, ok_message: str = "") -> Iterator[None]:
    """在 with 块期间显示 BusyBar，退出时短暂显示结果。"""
    _post_busy(message if (message or "").strip() else _DEFAULT_MESSAGE, 0.0)
    try:
        yield
        if ok_message.strip():
            _post_busy(ok_message.strip(), 2.5)
        else:
            _hide_busy()
    except Exception:
        _post_busy("Moondream: 识屏失败", 4.0)
        raise
