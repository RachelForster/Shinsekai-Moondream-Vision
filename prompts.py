"""English vision prompts for each Moondream trigger type (and shared defaults)."""

from __future__ import annotations

from plugins.moondream_vision.config_model import MoondreamVisionConfig

# Built-in English prompts when config leaves a field empty.
DEFAULT_QUESTION_FALLBACK = (
    "In one or two short English sentences, describe what is most useful on "
    "screen for the chat assistant (apps, visible text, errors, UI state)."
)

DEFAULT_BY_REASON: dict[str, str] = {
    "screen_diff": (
        "The screen content changed noticeably since the last capture. "
        "Briefly describe what changed and what matters for helping the user "
        "(new windows, text, errors, or UI state). Answer in English."
    ),
    "mouse": (
        "The user moved the mouse a lot. "
        "Briefly describe the visible interface and any prominent text, controls, "
        "or highlighted areas that might reflect what they are doing. Answer in English."
    ),
    "new_window": (
        "A new top-level window likely appeared. "
        "Describe what window or dialog is visible and its main content in English, briefly."
    ),
    "foreground": (
        "The foreground (focused) window changed. "
        "Describe the active window and the key visible information "
        "(title bar, main text, buttons, error messages). Answer in English."
    ),
}

_REASON_PRIORITY: tuple[str, ...] = (
    "screen_diff",
    "foreground",
    "new_window",
    "mouse",
)


def question_for_triggers(cfg: MoondreamVisionConfig, reasons: list[str]) -> str:
    """
    Pick one English instruction for the model from trigger reason(s).

    When multiple reasons fire, uses priority: screen_diff > foreground > new_window > mouse.
    Non-empty per-field strings in ``cfg`` override the built-in defaults for that reason.

    If all four per-reason fields are empty but ``cfg.question`` is set (legacy single prompt),
    that string is used for every trigger. Otherwise built-in English defaults apply per reason.
    """
    reason_set = {r.strip() for r in reasons if r and str(r).strip()}
    field_for_reason = {
        "screen_diff": str(getattr(cfg, "question_screen_diff", "") or "").strip(),
        "mouse": str(getattr(cfg, "question_mouse", "") or "").strip(),
        "new_window": str(getattr(cfg, "question_new_window", "") or "").strip(),
        "foreground": str(getattr(cfg, "question_foreground", "") or "").strip(),
    }
    legacy = str(getattr(cfg, "question", "") or "").strip()
    all_specific_empty = not any(field_for_reason[k] for k in _REASON_PRIORITY)

    for key in _REASON_PRIORITY:
        if key not in reason_set:
            continue
        if field_for_reason[key]:
            return field_for_reason[key]
        if legacy and all_specific_empty:
            return legacy
        return DEFAULT_BY_REASON[key]
    if legacy:
        return legacy
    return DEFAULT_QUESTION_FALLBACK
