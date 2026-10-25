import json
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
from gi.repository import Gst  # noqa: E402

Gst.init(None)

np = pytest.importorskip("numpy")

from engine.zero_shot_detector_engine import ZeroShotDetectorEngine  # noqa: E402
from zeroshotdetector import DEFAULT_MODEL_NAME  # noqa: E402

pytestmark = pytest.mark.skipif(
    Gst.ElementFactory.find("mp_zeroshotdetector") is None,
    reason="the python plugin loader did not register the modelpipe elements",
)

FRAME_SIZE = 64
FRAMES = 2
FRAMERATE = 30
CONFIDENCE = 0.05
# the model downloads on first use and runs on cpu
PIPELINE_TIMEOUT = 600 * Gst.SECOND


@pytest.fixture(scope="module")
def engine():
    engine = ZeroShotDetectorEngine()
    engine.do_set_device("cpu")
    engine.do_load_model(DEFAULT_MODEL_NAME)
    engine.labels = ["ball", "person"]
    engine.confidence = CONFIDENCE
    return engine


def test_the_engine_returns_one_detection_dict_per_frame(engine):
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    frame[16:48, 16:48] = 255

    result = engine.do_forward(frame)

    assert set(result) == {"boxes", "labels", "scores"}
    assert len(result["boxes"]) == len(result["labels"]) == len(result["scores"])
    assert all(len(box) == 4 for box in result["boxes"])
    assert all(0 <= label < len(engine.labels) for label in result["labels"])
    assert all(0.0 <= score <= 1.0 for score in result["scores"])

    batch = np.stack([frame, frame])
    batched = engine.do_forward(batch)
    assert len(batched) == len(batch)
    assert batched[0]["boxes"] == result["boxes"]


def test_the_engine_refuses_to_run_without_labels(engine):
    engine_labels = engine.labels
    engine.labels = []
    try:
        with pytest.raises(ValueError):
            engine.do_forward(np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))
    finally:
        engine.labels = engine_labels


def test_the_element_names_detections_after_the_label_text(tmp_path):
    records = tmp_path / "records.jsonl"
    pipeline = Gst.parse_launch(
        f"videotestsrc num-buffers={FRAMES} pattern=ball "
        f"! video/x-raw,width={FRAME_SIZE},height={FRAME_SIZE},"
        f"framerate={FRAMERATE}/1,format=RGBA "
        f"! mp_zeroshotdetector device=cpu labels=ball confidence={CONFIDENCE} "
        f"! mp_metasink location={records.as_posix()}"
    )
    pipeline.set_state(Gst.State.PLAYING)
    message = pipeline.get_bus().timed_pop_filtered(
        PIPELINE_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)
    assert message is not None, "pipeline did not finish"
    if message.type == Gst.MessageType.ERROR:
        error, debug = message.parse_error()
        pytest.fail(f"{error.message}\n{debug}")

    lines = [
        json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()
    ]
    detections = [detection for line in lines for detection in line["detections"]]
    assert len(lines) == FRAMES
    assert detections
    assert all(detection["label"] == "stream_0_ball" for detection in detections)
