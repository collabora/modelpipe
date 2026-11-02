import os
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
PLUGIN_DIR = BASE_DIR / "plugins" / "python"

# the plugin loader reads GST_PLUGIN_PATH at Gst.init
os.environ["GST_PLUGIN_PATH"] = str(BASE_DIR / "plugins")
sys.path.insert(0, str(PLUGIN_DIR))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst  # noqa: E402

Gst.init(None)

from backend import VideoTransform  # noqa: E402
from engine.engine_factory import EngineFactory  # noqa: E402
from engine.ml_engine import MLEngine  # noqa: E402

FAILING_MODEL_NAME = "does-not-exist.pt"
LOAD_FAILURE = "no weights at does-not-exist.pt"
FAILING_ENGINE = "test_failing_engine"
WORKING_ENGINE = "test_working_engine"
LOADABLE_MODEL_NAME = "loads-fine.pt"
DEVICE_REFUSING_ENGINE = "test_device_refusing_engine"
REFUSED_DEVICE = "cuda"
DEVICE_REFUSAL = "no cuda device here"
PIPELINE_TIMEOUT = 10 * Gst.SECOND


class FailingEngine(MLEngine):
    def do_load_model(self, model_name, **kwargs):
        self.model_name = model_name
        raise FileNotFoundError(LOAD_FAILURE)

    def do_set_device(self, device):
        self.device = device

    def do_forward(self, frames):
        raise AssertionError("inference ran without a model")

    def do_generate(self, input_text, max_length=1000, system_prompt=None):
        raise AssertionError("generation ran without a model")


class WorkingEngine(FailingEngine):
    def do_load_model(self, model_name, **kwargs):
        self.model = model_name


class LoadFails(VideoTransform):
    __gstmetadata__ = (
        "Load Fails",
        "Transform",
        "an element whose model never loads",
        "test",
    )
    # pygobject reads the templates off the class's own dict, not the base's
    __gsttemplates__ = VideoTransform.__gsttemplates__

    def __init__(self):
        super().__init__()
        EngineFactory.register(FAILING_ENGINE, FailingEngine)
        self.mgr.engine_name = FAILING_ENGINE

    def process_frames(self, frames, num_sources, fmt, target):
        raise AssertionError("a frame was processed without a model")


GObject.type_register(LoadFails)
Gst.Element.register(None, "loadfails", Gst.Rank.NONE, LoadFails)


def test_a_model_that_cannot_load_fails_the_pipeline_with_the_cause():
    pipeline = Gst.parse_launch(
        "videotestsrc num-buffers=5 "
        "! video/x-raw,width=64,height=48,format=RGB "
        f"! loadfails model-name={FAILING_MODEL_NAME} "
        "! fakesink"
    )

    assert pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.SUCCESS
    message = pipeline.get_bus().timed_pop_filtered(
        PIPELINE_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)

    assert message is not None, "the pipeline neither errored nor finished"
    assert message.type == Gst.MessageType.ERROR, "the pipeline ran with no model"
    error, debug = message.parse_error()
    assert FAILING_MODEL_NAME in error.message
    assert LOAD_FAILURE in error.message
    assert "FileNotFoundError" in debug


def test_the_engine_keeps_no_model_after_a_failed_load():
    element = LoadFails()
    element.set_property("model-name", FAILING_MODEL_NAME)

    with pytest.raises(FileNotFoundError):
        element.do_load_model()

    assert element.engine.model is None
    assert element.engine.model_name is None
    assert element.engine.tokenizer is None


class LoadWorks(VideoTransform):
    __gstmetadata__ = (
        "Load Works",
        "Transform",
        "an element whose model loads",
        "test",
    )
    __gsttemplates__ = VideoTransform.__gsttemplates__

    def __init__(self):
        super().__init__()
        EngineFactory.register(WORKING_ENGINE, WorkingEngine)
        self.mgr.engine_name = WORKING_ENGINE
        self.processed = 0

    def process_frames(self, frames, num_sources, fmt, target):
        self.processed += 1


GObject.type_register(LoadWorks)
Gst.Element.register(None, "loadworks", Gst.Rank.NONE, LoadWorks)


def test_a_model_that_loads_leaves_the_pipeline_running():
    pipeline = Gst.parse_launch(
        "videotestsrc num-buffers=5 "
        "! video/x-raw,width=64,height=48,format=RGB "
        f"! loadworks name=element model-name={LOADABLE_MODEL_NAME} "
        "! fakesink"
    )
    element = pipeline.get_by_name("element")

    pipeline.set_state(Gst.State.PLAYING)
    message = pipeline.get_bus().timed_pop_filtered(
        PIPELINE_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)

    assert message is not None and message.type == Gst.MessageType.EOS
    assert element.processed == 5
    assert element.engine.model == LOADABLE_MODEL_NAME


class EngineNamedOnThePipeline(VideoTransform):
    __gstmetadata__ = (
        "Engine Named On The Pipeline",
        "Transform",
        "an element that takes its engine from the engine-name property",
        "test",
    )
    __gsttemplates__ = VideoTransform.__gsttemplates__

    def process_frames(self, frames, num_sources, fmt, target):
        pass


GObject.type_register(EngineNamedOnThePipeline)


def test_an_engine_name_set_after_the_device_replaces_the_engine():
    EngineFactory.register(FAILING_ENGINE, FailingEngine)
    EngineFactory.register(WORKING_ENGINE, WorkingEngine)
    element = EngineNamedOnThePipeline()
    element.set_property("engine-name", FAILING_ENGINE)
    element.set_property("device", "cpu")
    element.set_property("engine-name", WORKING_ENGINE)
    element.set_property("model-name", LOADABLE_MODEL_NAME)
    element.do_load_model()

    assert type(element.engine) is WorkingEngine
    assert element.engine.device == "cpu"


def test_a_layout_set_before_the_device_reaches_the_engine():
    EngineFactory.register(WORKING_ENGINE, WorkingEngine)
    element = EngineNamedOnThePipeline()
    element.set_property("engine-name", WORKING_ENGINE)
    element.set_property("input-format", "nchw")
    element.set_property("post-process", "anchor_free")
    element.set_property("device", "cpu")
    element.initialize_engine()

    assert element.engine.input_format == "nchw"
    assert element.engine.post_process == "anchor_free"


class DeviceRefusingEngine(WorkingEngine):
    def do_set_device(self, device):
        if device == REFUSED_DEVICE:
            raise RuntimeError(DEVICE_REFUSAL)
        self.device = device


class DeviceRefused(VideoTransform):
    __gstmetadata__ = (
        "Device Refused",
        "Transform",
        "an element whose engine refuses the device",
        "test",
    )
    __gsttemplates__ = VideoTransform.__gsttemplates__

    def __init__(self):
        super().__init__()
        EngineFactory.register(DEVICE_REFUSING_ENGINE, DeviceRefusingEngine)
        self.mgr.engine_name = DEVICE_REFUSING_ENGINE

    def process_frames(self, frames, num_sources, fmt, target):
        raise AssertionError("a frame was processed on a device the engine refused")


GObject.type_register(DeviceRefused)
Gst.Element.register(None, "devicerefused", Gst.Rank.NONE, DeviceRefused)


@pytest.mark.parametrize(
    "properties",
    [
        f"device={REFUSED_DEVICE} model-name={LOADABLE_MODEL_NAME}",
        f"device=cpu model-name={LOADABLE_MODEL_NAME} device={REFUSED_DEVICE}",
    ],
    ids=["first device", "device changed on a live engine"],
)
def test_a_device_the_engine_refuses_fails_the_pipeline_with_the_cause(properties):
    pipeline = Gst.parse_launch(
        "videotestsrc num-buffers=5 "
        "! video/x-raw,width=64,height=48,format=RGB "
        f"! devicerefused {properties} "
        "! fakesink"
    )

    assert pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.SUCCESS
    message = pipeline.get_bus().timed_pop_filtered(
        PIPELINE_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)

    assert message is not None, "the pipeline neither errored nor finished"
    assert message.type == Gst.MessageType.ERROR, "the pipeline ran on another device"
    error, _ = message.parse_error()
    assert DEVICE_REFUSAL in error.message


class HalfPrecisionEngine(WorkingEngine):
    PRECISIONS = ("fp32", "fp16")


HALF_PRECISION_ENGINE = "test_half_precision_engine"


def test_a_precision_the_engine_does_not_run_fails_the_load_naming_it():
    element = LoadWorks()
    element.set_property("model-name", LOADABLE_MODEL_NAME)
    element.set_property("precision", "fp16")

    with pytest.raises(ValueError, match="WorkingEngine runs fp32, not precision=fp16"):
        element.do_load_model()
    assert element.engine.model is None


def test_an_engine_that_runs_the_precision_loads_with_it():
    element = LoadWorks()
    EngineFactory.register(HALF_PRECISION_ENGINE, HalfPrecisionEngine)
    element.mgr.engine_name = HALF_PRECISION_ENGINE
    element.set_property("model-name", LOADABLE_MODEL_NAME)
    element.set_property("precision", "fp16")

    element.do_load_model()
    assert element.engine.model == LOADABLE_MODEL_NAME
    assert element.engine.precision == "fp16"
