import importlib
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)
torch = pytest.importorskip("torch")

from backend import VideoTransform, CUDA_INFERENCE_FRAME_CAPABILITIES  # noqa: E402
from engine.engine_factory import EngineFactory  # noqa: E402
from engine.pytorch_engine import PyTorchEngine  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and Gst.ElementFactory.find("cudaupload")),
    reason="needs a cuda device and the cudaupload element",
)

URGENT_PRIORITY = -1
FRAME_COUNT = 3
PIPELINE_TIMEOUT = 30 * Gst.SECOND
CURRENT_STREAM_ENGINE = "test_current_stream_engine"
# the backend.gst package exports a frameio object under the module's name
gst_frameio = importlib.import_module("backend.gst.frameio")


class CurrentStreamEngine(PyTorchEngine):
    def do_load_model(self, model_name, **kwargs):
        self.model = model_name

    def do_forward(self, frames):
        return torch.cuda.current_stream()


class CurrentStreamRecorder(VideoTransform):
    __gstmetadata__ = (
        "Current Stream Recorder",
        "Transform",
        "records the cuda stream each frame is inferred on",
        "test",
    )
    FRAME_CAPABILITIES = CUDA_INFERENCE_FRAME_CAPABILITIES

    def __init__(self):
        super().__init__()
        EngineFactory.register(CURRENT_STREAM_ENGINE, CurrentStreamEngine)
        self.mgr.engine_name = CURRENT_STREAM_ENGINE
        self.forward_streams = []

    def process_frames(self, frames, num_sources, fmt, target):
        self.forward_streams.append(self.engine.do_forward(frames))


Gst.Element.register(
    None, "currentstreamrecorder", Gst.Rank.NONE, CurrentStreamRecorder
)


def cuda_classifier(property_order):
    from classifier import Classifier

    element = Classifier()
    element.set_property("engine-name", "pytorch")
    for name, value in property_order:
        element.set_property(name, value)
    element.initialize_engine()
    return element


def test_two_cuda_elements_infer_on_two_streams_apart_from_the_default():
    first, second = (cuda_classifier([("device", "cuda")]) for _ in range(2))

    stream_handles = {element.engine.stream.cuda_stream for element in (first, second)}

    assert len(stream_handles) == 2
    assert torch.cuda.default_stream().cuda_stream not in stream_handles


@pytest.mark.parametrize(
    "property_order",
    [
        [("device", "cuda"), ("stream-priority", URGENT_PRIORITY)],
        [("stream-priority", URGENT_PRIORITY), ("device", "cuda")],
    ],
)
def test_the_stream_priority_reaches_the_stream(property_order):
    element = cuda_classifier(property_order)

    assert element.engine.stream.priority == URGENT_PRIORITY


def test_a_forward_runs_on_the_engine_stream():
    engine = CurrentStreamEngine()
    engine.do_set_device("cuda")

    assert engine.do_forward(None) == engine.stream
    assert torch.cuda.current_stream() != engine.stream


def test_a_device_frame_is_mapped_and_inferred_on_the_engine_stream(monkeypatch):
    map_streams = []
    map_device_frame = gst_frameio.map_device_frame

    def map_recording_the_stream(*arguments):
        map_streams.append(torch.cuda.current_stream())
        return map_device_frame(*arguments)

    monkeypatch.setattr(gst_frameio, "map_device_frame", map_recording_the_stream)
    pipeline = Gst.parse_launch(
        f"videotestsrc num-buffers={FRAME_COUNT} "
        "! video/x-raw,format=RGBA,width=64,height=48 ! cudaupload "
        "! video/x-raw(memory:CUDAMemory),format=RGBA "
        "! currentstreamrecorder name=recorder model-name=none device=cuda "
        "! fakesink"
    )
    pipeline.set_state(Gst.State.PLAYING)
    message = pipeline.get_bus().timed_pop_filtered(
        PIPELINE_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)

    assert message is not None and message.type == Gst.MessageType.EOS
    recorder = pipeline.get_by_name("recorder")
    engine_streams = [recorder.engine.stream] * FRAME_COUNT
    assert map_streams == engine_streams
    assert recorder.forward_streams == engine_streams
