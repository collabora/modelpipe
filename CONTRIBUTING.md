# Contributing to ModelPipe

This guide covers how to get started.

## Development Setup

```bash
git clone https://github.com/collabora/modelpipe.git
cd modelpipe
uv venv --python /usr/bin/python3 --system-site-packages
uv sync --extra all
```

## Code Style

We use **ruff** for linting and **black** for formatting:

```bash
uvx ruff check plugins/ tests/ utils/
uvx black plugins/ tests/ utils/
```

CI enforces these on every PR.

## Shaders

The Vulkan overlay ships the SPIR-V next to its GLSL source in
`plugins/python/overlay_helper/shaders/`. After editing a shader, recompile it
with the source embedded, which `tests/test_overlay_shaders.py` checks:

```bash
cd plugins/python/overlay_helper/shaders
glslc -g overlay.vert -o overlay.vert.spv
glslc -g overlay.frag -o overlay.frag.spv
```

## Adding a New Plugin

All GStreamer Python elements live in `plugins/python/`. To add a new element:

1. Create `plugins/python/my_element.py`
2. Import the appropriate base class (see table below)
3. Define your class with `__gstmetadata__` and `__gstelementfactory__`
4. Register with `GObject.type_register()`

### Base Classes

| Base Class | Module | Use Case |
|---|---|---|
| `BaseTransform` | `base_transform` | Generic video transforms |
| `BaseObjectDetector` | `base_objectdetector` | Object detection |
| `BaseClassifier` | `base_classifier` | Image classification |
| `BaseCaption` | `base_caption` | Video/image captioning |
| `BaseLLM` | `base_llm` | Large language models |
| `BaseTranscribe` | `base_transcribe` | Speech-to-text |
| `BaseTranslate` | `base_translate` | Text translation |
| `BaseTTS` | `base_tts` | Text-to-speech |
| `BaseSeparate` | `base_separate` | Audio source separation |

### Minimal Example

```python
CAN_REGISTER_ELEMENT = True
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GObject", "2.0")
    from gi.repository import GObject, Gst, GstBase
    from base_transform import BaseTransform
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    print(f"my_element not available: {e}")

if CAN_REGISTER_ELEMENT:
    class MyElement(BaseTransform):
        __gstmetadata__ = (
            "My Element",
            "Video/Filter",
            "Description of what it does",
            "Your Name <email@example.com>",
        )

    GObject.type_register(MyElement)
    __gstelementfactory__ = ("my_element", Gst.Rank.NONE, MyElement)
```

## Adding a New ML Engine

Engine implementations live in `plugins/python/engine/`. To add a new engine:

1. Create `plugins/python/engine/myengine_engine.py`
2. Inherit from `MLEngine` (in `plugins/python/engine/ml_engine.py`)
3. Implement the required methods: `load_model()`, `predict()`, `get_input_shape()`
4. Register in `plugins/python/engine/engine_factory.py`

## Testing

Run tests with:

```bash
export GST_PLUGIN_PATH=$PWD/plugins
pytest tests/
```

For pipeline testing:

```bash
gst-inspect-1.0 python  # Verify all elements load
python mp-launch.py filesrc location=data/people.mp4 num-buffers=5 \
  ! decodebin ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_objectdetector engine-name=onnx model-name=yolo11n.onnx device=cpu \
    input-format=nchw post-process=anchor_free \
  ! fakesink
```

## Pull Request Guidelines

- Ensure `ruff check` and `black --check` pass
- Add yourself to the element's `__gstmetadata__` author field
- Test with at least one engine (PyTorch or ONNX recommended for CI)
- Keep the `try/except ImportError` pattern so elements degrade gracefully when optional dependencies are missing

## License

This project is licensed under MPL-2.0. By contributing, you agree that your contributions will be licensed under the same terms.

The patches under `patches/` and `extern/migraphx/` change other projects and carry those projects' licenses.
