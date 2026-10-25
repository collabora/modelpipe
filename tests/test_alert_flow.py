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
gi.require_version("GstBase", "1.0")
from gi.repository import GObject, Gst, GstBase  # noqa: E402

Gst.init(None)

from backend import analytics  # noqa: E402
from metasink import buffer_record  # noqa: E402

pytestmark = pytest.mark.skipif(
    Gst.ElementFactory.find("mp_alert") is None,
    reason="the python plugin loader did not register the modelpipe elements",
)

FRAMERATE = 30
ALERT_FRAME = 30
FRAMES = 60
CLIP_SECONDS = 0.5
PIPELINE_TIMEOUT = 30 * Gst.SECOND


def run_to_eos(pipeline):
    pipeline.set_state(Gst.State.PLAYING)
    message = pipeline.get_bus().timed_pop_filtered(
        PIPELINE_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)
    assert message is not None, "pipeline did not finish"
    if message.type == Gst.MessageType.ERROR:
        error, debug = message.parse_error()
        pytest.fail(f"{error.message}\n{debug}")


# a pad probe's buffer is not writable, an in-place transform's is
class TagOneFrame(GstBase.BaseTransform):
    __gstmetadata__ = ("Tag One Frame", "Filter", "adds one detection", "test")
    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, Gst.Caps.new_any()
        ),
        Gst.PadTemplate.new(
            "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, Gst.Caps.new_any()
        ),
    )

    def __init__(self):
        super().__init__()
        self.set_in_place(True)

    def do_transform_ip(self, buffer):
        if buffer.pts == ALERT_FRAME * Gst.SECOND // FRAMERATE:
            meta = analytics.add_relation_meta(buffer)
            analytics.add_object(meta, "thing", 0, 0, 10, 10, 0.9)
        return Gst.FlowReturn.OK


GObject.type_register(TagOneFrame)
Gst.Element.register(None, "tagoneframe", Gst.Rank.NONE, TagOneFrame)


def test_alert_records_a_clip_and_the_sink_reports_it(tmp_path):
    clip_pattern = tmp_path / "clip-%s.webm"
    records = tmp_path / "records.jsonl"
    pipeline = Gst.parse_launch(
        f"videotestsrc num-buffers={FRAMES} "
        f"! video/x-raw,width=64,height=48,framerate={FRAMERATE}/1,format=RGBA "
        "! tagoneframe "
        '! mp_alert rules={"class":"thing"} draw-alert=false '
        f"! mp_alertrecorder location={clip_pattern.as_posix()} "
        f"seconds-before={CLIP_SECONDS} seconds-after={CLIP_SECONDS} "
        f"! mp_metasink location={records.as_posix()}"
    )
    run_to_eos(pipeline)

    lines = [
        json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()
    ]
    alerted = [line for line in lines if "alert" in line]
    assert len(alerted) == 1
    assert alerted[0]["pts"] == pytest.approx(ALERT_FRAME / FRAMERATE)
    assert alerted[0]["alert"][0]["detection"]["label"] == "thing"
    assert alerted[0]["detections"][0]["label"] == "thing"

    clips = list(tmp_path.glob("clip-*.webm"))
    assert len(clips) == 1
    assert count_decoded_frames(clips[0]) == pytest.approx(
        2 * CLIP_SECONDS * FRAMERATE, abs=2
    )


def count_decoded_frames(path):
    decoded = []
    pipeline = Gst.parse_launch(
        f"filesrc location={path.as_posix()} ! matroskademux ! vp8dec "
        "! fakesink name=out signal-handoffs=true"
    )
    pipeline.get_by_name("out").connect(
        "handoff", lambda sink, buffer, pad: decoded.append(buffer.pts)
    )
    run_to_eos(pipeline)
    return len(decoded)


def test_a_text_buffer_becomes_a_text_record():
    buffer = Gst.Buffer.new_wrapped(b"hello")
    buffer.pts = 2 * Gst.SECOND
    assert buffer_record(buffer, "text/x-raw") == {"pts": 2.0, "text": "hello"}


def test_a_frame_with_nothing_attached_gives_an_empty_record():
    buffer = Gst.Buffer.new_wrapped(bytes(16))
    buffer.pts = 0
    assert buffer_record(buffer, "video/x-raw") == {"pts": 0.0, "detections": []}
