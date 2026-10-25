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

pytestmark = pytest.mark.skipif(
    Gst.ElementFactory.find("mp_digest") is None,
    reason="the python plugin loader did not register the modelpipe elements",
)

WINDOW_SECONDS = 1.0
PIPELINE_TIMEOUT = 30 * Gst.SECOND
CAPTIONS = [
    (0.0, "a player runs"),
    (0.4, "a player shoots"),
    (0.8, "the keeper dives"),
    (1.2, "the ball is out"),
    (1.6, "a throw in"),
    (2.4, "the whistle blows"),
]


def run_to_eos(pipeline):
    message = pipeline.get_bus().timed_pop_filtered(
        PIPELINE_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)
    assert message is not None, "pipeline did not finish"
    if message.type == Gst.MessageType.ERROR:
        error, debug = message.parse_error()
        pytest.fail(f"{error.message}\n{debug}")


def test_a_digest_holds_one_record_per_window(tmp_path):
    records = tmp_path / "records.jsonl"
    pipeline = Gst.parse_launch(
        "appsrc name=source format=time caps=text/x-raw,format=utf8 "
        f"! mp_digest window-seconds={WINDOW_SECONDS} "
        f"! mp_metasink location={records.as_posix()}"
    )
    source = pipeline.get_by_name("source")
    pipeline.set_state(Gst.State.PLAYING)
    for seconds, caption in CAPTIONS:
        buffer = Gst.Buffer.new_wrapped(caption.encode("utf-8"))
        buffer.pts = int(seconds * Gst.SECOND)
        assert source.emit("push-buffer", buffer) == Gst.FlowReturn.OK
    source.emit("end-of-stream")
    run_to_eos(pipeline)

    lines = [
        json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()
    ]
    assert [line["pts"] for line in lines] == [0.0, 1.2, 2.4]
    assert lines[0]["text"] == (
        "At 0.0s: a player runs\n"
        "At 0.4s: a player shoots\n"
        "At 0.8s: the keeper dives"
    )
    assert lines[1]["text"] == "At 1.2s: the ball is out\nAt 1.6s: a throw in"
    assert lines[2]["text"] == "At 2.4s: the whistle blows"
