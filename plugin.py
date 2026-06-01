"""Moondream screen understanding plugin."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from sdk.plugin import PluginBase
from sdk.plugin_host_context import PluginHostContext
from sdk.register import PluginCapabilityRegistry
from sdk.types import FrontendConfigContribution, ToolsTabContribution

from plugins.moondream_vision import runtime
from plugins.moondream_vision.config_model import (
    MoondreamVisionConfig,
    default_config_path,
    load_config,
    save_config,
)

import plugins.moondream_vision.llm_tool as _moondream_llm_tool  # noqa: F401


def _number_value(values: Mapping[str, object], key: str, default: float) -> float:
    raw = values.get(key, default)
    if raw is None or raw == "":
        return default
    return float(raw)


def _int_value(values: Mapping[str, object], key: str, default: int) -> int:
    raw = values.get(key, default)
    if raw is None or raw == "":
        return default
    return int(raw)


class MoondreamVisionPlugin(PluginBase):
    """Capture screen context with Moondream and submit it as chat input."""

    @property
    def plugin_id(self) -> str:
        return "com.shinsekai.moondream_vision"

    @property
    def plugin_version(self) -> str:
        return "0.1.0"

    @property
    def priority(self) -> int:
        return 80

    def initialize(
        self,
        register: PluginCapabilityRegistry,
        plugin_root: Path,
        host: PluginHostContext,
    ) -> None:
        _ = host
        runtime.set_plugin_root(plugin_root)
        register.register_user_input_trigger(runtime.bind_emit)

        def build_tools(plg):
            _ = plg
            from plugins.moondream_vision.settings_tab import MoondreamVisionSettingsTab

            return MoondreamVisionSettingsTab(plg, plugin_root)

        register.register_tools_tab(
            ToolsTabContribution(
                tab_id="moondream_vision",
                title="Moondream Vision",
                build=build_tools,
                order=45.0,
            )
        )
        register.register_frontend_config_page(
            FrontendConfigContribution(
                page_id="moondream_vision",
                title="Moondream Vision",
                kind="tools",
                description=(
                    "Use local screen captures and Moondream to turn screen activity "
                    "into chat input."
                ),
                restart_hint="Restart chat after changing model, device, quantization, or cache settings.",
                schema=[
                    {
                        "description": "Model loading may download weights on first use.",
                        "fields": [
                            {
                                "defaultValue": False,
                                "key": "enabled",
                                "label": "Enable screen understanding",
                                "span": "full",
                                "type": "boolean",
                            },
                            {
                                "defaultValue": "vikhyatk/moondream2",
                                "key": "model_id",
                                "label": "Model ID",
                                "placeholder": "vikhyatk/moondream2",
                                "type": "text",
                            },
                            {
                                "defaultValue": "",
                                "key": "revision",
                                "label": "Revision",
                                "placeholder": "Optional model revision",
                                "type": "text",
                            },
                            {
                                "defaultValue": "",
                                "key": "cache_dir",
                                "label": "Cache directory",
                                "placeholder": "Leave empty for the default Hugging Face cache",
                                "span": "full",
                                "type": "text",
                            },
                            {
                                "defaultValue": "auto",
                                "key": "device",
                                "label": "Device",
                                "options": [
                                    {"label": "Auto", "value": "auto"},
                                    {"label": "CUDA", "value": "cuda"},
                                    {"label": "Apple MPS", "value": "mps"},
                                    {"label": "CPU", "value": "cpu"},
                                ],
                                "type": "select",
                            },
                            {
                                "defaultValue": "none",
                                "description": "INT8 and INT4 usually require NVIDIA CUDA and bitsandbytes.",
                                "key": "quantization",
                                "label": "Quantization",
                                "options": [
                                    {"label": "None", "value": "none"},
                                    {"label": "INT8", "value": "int8"},
                                    {"label": "INT4", "value": "int4"},
                                ],
                                "type": "select",
                            },
                        ],
                        "id": "model",
                        "title": "Model",
                    },
                    {
                        "fields": [
                            {
                                "defaultValue": 0.35,
                                "description": "Sampling interval for screen diff, mouse, and window changes.",
                                "key": "motion_poll_sec",
                                "label": "Trigger sample interval",
                                "max": 3.0,
                                "min": 0.12,
                                "step": 0.05,
                                "type": "number",
                            },
                            {
                                "defaultValue": 0.35,
                                "description": "Ratio of changed thumbnail pixels needed to trigger inference.",
                                "key": "diff_threshold",
                                "label": "Screen diff threshold",
                                "max": 0.35,
                                "min": 0.003,
                                "step": 0.002,
                                "type": "number",
                            },
                            {
                                "defaultValue": 1.1,
                                "description": "Mouse movement as a percentage of the current monitor's larger side.",
                                "key": "mouse_move_percent",
                                "label": "Mouse move threshold (%)",
                                "max": 25.0,
                                "min": 0.02,
                                "step": 0.05,
                                "type": "number",
                            },
                            {
                                "defaultValue": 30,
                                "key": "interval_sec",
                                "label": "Minimum inference interval",
                                "max": 600.0,
                                "min": 5.0,
                                "step": 1.0,
                                "type": "number",
                            },
                            {
                                "defaultValue": 1,
                                "description": "mss monitor index. 0 means all monitors combined.",
                                "key": "monitor_index",
                                "label": "Monitor index",
                                "max": 16,
                                "min": 0,
                                "step": 1,
                                "type": "integer",
                            },
                            {
                                "defaultValue": 512,
                                "description": "Resize the longer side before inference. 0 disables resizing.",
                                "key": "infer_max_side",
                                "label": "Max inference side (px)",
                                "max": 8192,
                                "min": 0,
                                "step": 128,
                                "type": "integer",
                            },
                        ],
                        "id": "triggers",
                        "title": "Triggers",
                    },
                    {
                        "fields": [
                            {
                                "defaultValue": "",
                                "key": "question_screen_diff",
                                "label": "Screen diff prompt",
                                "placeholder": "screen thumbnail changed a lot since last successful capture",
                                "span": "full",
                                "type": "textarea",
                            },
                            {
                                "defaultValue": "",
                                "key": "question_foreground",
                                "label": "Foreground switch prompt",
                                "placeholder": "focused window changed",
                                "span": "full",
                                "type": "textarea",
                            },
                            {
                                "defaultValue": "",
                                "key": "question_new_window",
                                "label": "New window prompt",
                                "placeholder": "new top-level window opened",
                                "span": "full",
                                "type": "textarea",
                            },
                            {
                                "defaultValue": "",
                                "key": "question_mouse",
                                "label": "Mouse movement prompt",
                                "placeholder": "user moved mouse beyond threshold",
                                "span": "full",
                                "type": "textarea",
                            },
                            {
                                "defaultValue": "",
                                "key": "question",
                                "label": "Fallback prompt",
                                "span": "full",
                                "type": "textarea",
                            },
                            {
                                "defaultValue": "[Screen] ",
                                "key": "message_prefix",
                                "label": "Message prefix",
                                "span": "full",
                                "type": "text",
                            },
                        ],
                        "id": "prompts",
                        "title": "Prompts",
                    },
                ],
                load_values=lambda: asdict(load_config(default_config_path(plugin_root))),
                save_values=lambda values: self._save_frontend_config(plugin_root, values),
                order=45.0,
            )
        )

    def _save_frontend_config(self, plugin_root: Path, values: Mapping[str, object]) -> None:
        cfg = MoondreamVisionConfig(
            enabled=bool(values.get("enabled", False)),
            model_id=str(values.get("model_id") or "vikhyatk/moondream2").strip() or "vikhyatk/moondream2",
            revision=str(values.get("revision") or "").strip(),
            cache_dir=str(values.get("cache_dir") or "").strip(),
            device=str(values.get("device") or "auto").strip().lower(),
            quantization=str(values.get("quantization") or "none").strip().lower(),
            motion_poll_sec=_number_value(values, "motion_poll_sec", MoondreamVisionConfig.motion_poll_sec),
            diff_threshold=_number_value(values, "diff_threshold", MoondreamVisionConfig.diff_threshold),
            mouse_move_percent=_number_value(
                values,
                "mouse_move_percent",
                MoondreamVisionConfig.mouse_move_percent,
            ),
            interval_sec=_number_value(values, "interval_sec", MoondreamVisionConfig.interval_sec),
            monitor_index=_int_value(values, "monitor_index", MoondreamVisionConfig.monitor_index),
            infer_max_side=_int_value(values, "infer_max_side", MoondreamVisionConfig.infer_max_side),
            question=str(values.get("question") or "").strip(),
            question_screen_diff=str(values.get("question_screen_diff") or "").strip(),
            question_mouse=str(values.get("question_mouse") or "").strip(),
            question_new_window=str(values.get("question_new_window") or "").strip(),
            question_foreground=str(values.get("question_foreground") or "").strip(),
            message_prefix=str(values.get("message_prefix") or MoondreamVisionConfig.message_prefix),
        )
        save_config(default_config_path(plugin_root), cfg)

    def shutdown(self) -> None:
        runtime.shutdown()
