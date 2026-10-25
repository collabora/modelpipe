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

pytestmark = pytest.mark.skipif(
    Gst.ElementFactory.find("mp_metareplay") is None,
    reason="the python plugin loader did not register the modelpipe elements",
)

FRAMERATE = 30
TAGGED_FRAME = 30
FRAMES = 60
PIPELINE_TIMEOUT = 30 * Gst.SECOND
CAPS = f"video/x-raw,width=64,height=48,framerate={FRAMERATE}/1,format=RGBA"


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
        if buffer.pts == TAGGED_FRAME * Gst.SECOND // FRAMERATE:
            meta = analytics.add_relation_meta(buffer)
            analytics.add_object(meta, "thing", 4, 6, 10, 12, 0.9)
        return Gst.FlowReturn.OK


GObject.type_register(TagOneFrame)
Gst.Element.register(None, "metareplaytagger", Gst.Rank.NONE, TagOneFrame)


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_replayed_records_alert_at_the_recorded_timestamp(tmp_path):
    records = tmp_path / "records.jsonl"
    replayed = tmp_path / "replayed.jsonl"

    run_to_eos(
        Gst.parse_launch(
            f"videotestsrc num-buffers={FRAMES} ! {CAPS} "
            "! metareplaytagger "
            f"! mp_metasink location={records.as_posix()}"
        )
    )
    recorded = [line for line in read_lines(records) if line["detections"]]
    assert len(recorded) == 1
    assert recorded[0]["pts"] == pytest.approx(TAGGED_FRAME / FRAMERATE)

    run_to_eos(
        Gst.parse_launch(
            f"videotestsrc num-buffers={FRAMES} ! {CAPS} "
            f"! mp_metareplay location={records.as_posix()} "
            '! mp_alert rules={"class":"thing"} draw-alert=false '
            f"! mp_metasink location={replayed.as_posix()}"
        )
    )
    lines = [line for line in read_lines(replayed) if line["detections"]]
    assert len(lines) == 1
    assert lines[0]["pts"] == pytest.approx(recorded[0]["pts"])
    assert lines[0]["detections"] == recorded[0]["detections"]
    assert lines[0]["alert"][0]["detection"]["label"] == "thing"


def test_a_blob_record_comes_back_as_the_same_blob(tmp_path):
    records = tmp_path / "records.jsonl"
    replayed = tmp_path / "replayed.jsonl"
    payload = {"kind": "note", "value": 7}
    records.write_text(
        json.dumps({"pts": TAGGED_FRAME / FRAMERATE, "caption": payload}) + "\n"
    )

    run_to_eos(
        Gst.parse_launch(
            f"videotestsrc num-buffers={FRAMES} ! {CAPS} "
            f"! mp_metareplay location={records.as_posix()} "
            f"! mp_metasink location={replayed.as_posix()}"
        )
    )
    lines = [line for line in read_lines(replayed) if "caption" in line]
    assert len(lines) == 1
    assert lines[0]["caption"] == payload
