from __future__ import annotations

import io
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from plugins.moondream_vision import screen_infer
from plugins.moondream_vision.config_model import (
    MoondreamVisionConfig,
    load_config,
    save_config,
)
from sdk.tool_registry import ToolNotReady


@pytest.fixture
def png():
    output = io.BytesIO()
    Image.new("RGB", (1000, 500), "white").save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def local_backend(monkeypatch):
    backend = SimpleNamespace(
        infer_screen_png=Mock(return_value="local answer"),
        is_tool_ready=Mock(return_value=True),
        start_preload_model=Mock(),
        loading_status_message=Mock(return_value="loading local model"),
        shutdown=Mock(),
    )
    monkeypatch.setitem(sys.modules, "plugins.moondream_vision.local_infer", backend)
    return backend


@pytest.fixture
def host_backend(monkeypatch):
    import ai.vision.service as service

    backend = SimpleNamespace(describe=Mock(return_value="host answer"))
    monkeypatch.setattr(service, "configured_vision_available", lambda: True)
    monkeypatch.setattr(service, "configured_vision_manager", lambda: backend)
    return backend


@pytest.mark.parametrize("raw", [None, {}, {"enabled": True, "model_id": "old-model"}])
def test_missing_adapter_defaults_to_host_vision(tmp_path, raw):
    path = tmp_path / "config.json"
    if raw is not None:
        path.write_text(json.dumps(raw), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.adapter == "vision"
    if raw and "model_id" in raw:
        assert cfg.model_id == "old-model"


@pytest.mark.parametrize("adapter,expected", [("Moondream", "moondream"), ("other", "vision"), ("", "vision")])
def test_adapter_normalization(tmp_path, adapter, expected):
    path = tmp_path / "config.json"
    save_config(path, MoondreamVisionConfig(adapter=adapter, model_id="saved-model", device="cpu"))
    cfg = load_config(path)
    assert cfg.adapter == expected
    assert cfg.model_id == "saved-model"
    assert cfg.device == "cpu"


def test_default_inference_uses_host_and_never_imports_local(monkeypatch, host_backend, png):
    monkeypatch.setitem(sys.modules, "plugins.moondream_vision.local_infer", None)
    cfg = MoondreamVisionConfig(infer_max_side=0)
    screen_infer.prepare_tool(cfg)
    assert screen_infer.infer_screen_png(png, "Read the dialog", cfg) == "host answer"
    host_backend.describe.assert_called_once_with(png, "Read the dialog")
    screen_infer.shutdown()


@pytest.mark.parametrize("cap,size", [(512, (512, 256)), (2000, (1000, 500)), (0, (1000, 500))])
def test_host_inference_honors_capture_resize(host_backend, png, cap, size):
    screen_infer.infer_screen_png(png, "What is shown?", MoondreamVisionConfig(infer_max_side=cap))
    sent_png = host_backend.describe.call_args.args[0]
    with Image.open(io.BytesIO(sent_png)) as image:
        assert image.size == size


def test_unconfigured_host_gives_guidance_without_local_fallback(monkeypatch, local_backend, png):
    import ai.vision.service as service

    monkeypatch.setattr(service, "configured_vision_available", lambda: False)
    with pytest.raises(RuntimeError, match="AI 服务"):
        screen_infer.infer_screen_png(png, "Read", MoondreamVisionConfig())
    local_backend.infer_screen_png.assert_not_called()
    local_backend.start_preload_model.assert_not_called()


def test_explicit_local_backend_still_uses_local_settings(local_backend, png):
    cfg = MoondreamVisionConfig(adapter="moondream", model_id="custom-model", device="cpu")
    screen_infer.prepare_tool(cfg)
    assert screen_infer.infer_screen_png(png, "Read", cfg) == "local answer"
    local_backend.infer_screen_png.assert_called_once_with(png, "Read", cfg)
    screen_infer.shutdown()
    local_backend.shutdown.assert_called_once()


def test_local_preload_preserves_tool_not_ready(local_backend):
    local_backend.is_tool_ready.return_value = False
    cfg = MoondreamVisionConfig(adapter="moondream")
    with pytest.raises(ToolNotReady, match="loading local model"):
        screen_infer.prepare_tool(cfg)
    local_backend.start_preload_model.assert_called_once_with(cfg)


@pytest.mark.parametrize("adapter", ["vision", "moondream"])
def test_rapidocr_does_not_require_local_weights(monkeypatch, local_backend, png, adapter):
    ocr = SimpleNamespace(ocr_png_bytes=Mock(return_value="屏幕文字"))
    monkeypatch.setitem(sys.modules, "plugins.moondream_vision.chinese_ocr", ocr)
    assert screen_infer.ocr_screen_png(png, MoondreamVisionConfig(adapter=adapter)) == ("屏幕文字", "rapidocr")
    local_backend.is_tool_ready.assert_not_called()
    local_backend.start_preload_model.assert_not_called()


def test_ocr_falls_back_to_host_adapter(monkeypatch, host_backend, local_backend, png):
    monkeypatch.setitem(sys.modules, "plugins.moondream_vision.chinese_ocr", None)
    assert screen_infer.ocr_screen_png(png, MoondreamVisionConfig()) == ("host answer", "vision_adapter")
    assert "preserving line breaks" in host_backend.describe.call_args.args[1]
    local_backend.start_preload_model.assert_not_called()


def test_query_and_ocr_tools_use_the_selected_backend(tmp_path, monkeypatch, host_backend, local_backend, png):
    from plugins.moondream_vision import capture_infer, llm_tool, runtime

    monkeypatch.setattr(runtime, "_plugin_root", tmp_path)
    capture = Mock(return_value=png)
    monkeypatch.setattr(capture_infer, "grab_screen_png", capture)
    monkeypatch.setitem(sys.modules, "plugins.moondream_vision.chinese_ocr", None)
    query = llm_tool.moondream_query_screen("Read the error", monitor_index=2)
    assert query == {"answer": "host answer", "monitor_index": 2}
    capture.assert_called_with(2)
    ocr = llm_tool.moondream_ocr_screen()
    assert ocr == {"text": "host answer", "monitor_index": 1, "engine": "vision_adapter"}
    local_backend.start_preload_model.assert_not_called()


def test_ocr_tool_propagates_local_loading(tmp_path, monkeypatch, local_backend, png):
    from plugins.moondream_vision import capture_infer, llm_tool, runtime

    save_config(tmp_path / "config.json", MoondreamVisionConfig(adapter="moondream"))
    monkeypatch.setattr(runtime, "_plugin_root", tmp_path)
    monkeypatch.setattr(capture_infer, "grab_screen_png", lambda _monitor: png)
    monkeypatch.setitem(sys.modules, "plugins.moondream_vision.chinese_ocr", None)
    local_backend.is_tool_ready.return_value = False
    with pytest.raises(ToolNotReady):
        llm_tool.moondream_ocr_screen()


def test_automatic_trigger_uses_host_adapter(tmp_path, monkeypatch, host_backend, local_backend, png):
    from plugins.moondream_vision import runtime

    save_config(tmp_path / "config.json", MoondreamVisionConfig(enabled=True, motion_poll_sec=0.12))
    monkeypatch.setattr(runtime, "_plugin_root", tmp_path)
    monkeypatch.setattr(runtime, "grab_screen_png", lambda _monitor: png)
    monkeypatch.setattr(runtime, "grab_screen_thumbnail", lambda _monitor: object())
    state = SimpleNamespace(evaluate=lambda *_args: (True, ["screen_diff"]), on_infer_done=Mock())
    monkeypatch.setattr(runtime, "MoondreamTriggerState", lambda: state)
    done = threading.Event()
    messages = []

    def emit(message):
        messages.append(message)
        runtime._stop_event.set()
        done.set()

    monkeypatch.setattr(runtime, "_emit_user_text", emit)
    try:
        runtime._restart_worker()
        assert done.wait(3), "Automatic screen trigger did not emit a message"
        assert messages == ["[Screen] host answer"]
        local_backend.infer_screen_png.assert_not_called()
    finally:
        runtime._stop_worker()


def test_frontend_schema_hides_local_fields_and_save_retains_them(tmp_path):
    from plugins.moondream_vision.plugin import MoondreamVisionPlugin

    register = SimpleNamespace(
        register_user_input_trigger=Mock(),
        register_tools_tab=Mock(),
        register_frontend_config_page=Mock(),
    )
    plugin = MoondreamVisionPlugin()
    plugin.initialize(register, tmp_path, None)
    page = register.register_frontend_config_page.call_args.args[0]
    fields = {field["key"]: field for group in page.schema for field in group["fields"]}
    assert fields["adapter"]["defaultValue"] == "vision"
    for key in ("model_id", "revision", "cache_dir", "device", "quantization"):
        assert fields[key]["visibleWhen"] == {"adapter": "moondream"}
    for key in ("enabled", "adapter", "infer_max_side"):
        assert "visibleWhen" not in fields[key]
    values = page.load_values()
    values.update(adapter="moondream", model_id="custom", device="cpu", quantization="int8")
    page.save_values(values)
    assert page.load_values()["adapter"] == "moondream"
    values["adapter"] = "vision"
    page.save_values(values)
    saved = page.load_values()
    assert (saved["adapter"], saved["model_id"], saved["device"], saved["quantization"]) == (
        "vision", "custom", "cpu", "int8",
    )
