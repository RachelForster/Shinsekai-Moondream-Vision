# Shinsekai-Moondream-Vision
Screen understanding for Shinsekai. Automatic screen triggers and the existing
`moondream_query_screen` / `moondream_ocr_screen` tools use the vision adapter
configured in **AI Service** by default. Configure a working visual service there
before enabling screen understanding.

Select **Moondream (local)** in the plugin's **Adapter** setting to use the
plugin's local model instead. Its model ID, revision, cache, device, and
quantization settings appear only in this mode. Existing configurations without
an `adapter` value also default to the host vision adapter; saved local model
settings are retained when switching backends.

OCR prefers RapidOCR when available, then uses the selected backend. Moondream
weights are preloaded only when the local backend needs them.

Run the plugin tests from a Shinsekai checkout with this plugin installed:

```sh
python -m pytest plugins/moondream_vision/tests
```
