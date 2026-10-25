import asyncio
import io
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "python"))

pytest.importorskip("mcp")
gi = pytest.importorskip("gi")
PillowImage = pytest.importorskip("PIL.Image")

import mp_mcp  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

gi.require_version("GstPbutils", "1.0")
from gi.repository import Gst, GstPbutils  # noqa: E402

pytestmark = pytest.mark.skipif(
    not any(element["name"] == "mp_metasink" for element in mp_mcp.list_elements()),
    reason="the python plugin loader did not register the modelpipe elements",
)

RECORD_WAIT_SECONDS = 10
RECENT_MESSAGES_READ = 100
FRAME_WAIT_SECONDS = 2
MISSING_KEY_WAIT_SECONDS = 0.5
MISSING_KEY_DEADLINE_SECONDS = 5
LOADED_WAIT_SECONDS = 1
LOADED_DEADLINE_SECONDS = 0.5
JPEG_MAGIC = b"\xff\xd8"
YELLOW_BAR_PIXEL = (14, 10)
YELLOW = (255, 255, 0)
JPEG_CHANNEL_TOLERANCE = 32
CLIP_SOURCE_FRAMES = 180
CLIP_BUILD_SECONDS = 30
CLIP_DISCOVER_SECONDS = 5
LEAST_CLIP_FRAMES = 55
MOST_CLIP_FRAMES = 65
CLIP_SECONDS = 2.0
CLIP_DURATION_TOLERANCE = 0.2
FLOW_SETTLE_SECONDS = 1.0
SLOW_ELEMENT_MICROSECONDS = 60000

VIDEO_PIPELINE = "videotestsrc ! video/x-raw,width=64,height=48 ! mp_metasink name=sink"
# 66 wide is 198 bytes a row, padded to a multiple of 4
PADDED_VIDEO_PIPELINE = (
    "videotestsrc ! video/x-raw,width=66,height=48 ! mp_metasink name=sink"
)
CLIP_SOURCE_PIPELINE = (
    "videotestsrc num-buffers={frames} ! video/x-raw,framerate=30/1,width=64,height=48 "
    "! vp8enc deadline=1 ! webmmux ! filesink location={path}"
)


class StubVlmEngine:
    def __init__(self):
        self.calls = []

    def do_generate(self, image, prompt, system_prompt, max_tokens, temperature):
        self.calls.append((image, prompt, system_prompt, max_tokens, temperature))
        return "a test pattern"


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "source.webm"
    pipeline = Gst.parse_launch(
        CLIP_SOURCE_PIPELINE.format(frames=CLIP_SOURCE_FRAMES, path=path.as_posix())
    )
    pipeline.set_state(Gst.State.PLAYING)
    pipeline.get_bus().timed_pop_filtered(
        CLIP_BUILD_SECONDS * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)
    return path


def test_text_flows_from_the_sink_to_latest_metadata(tmp_path):
    text = tmp_path / "hello.txt"
    text.write_text("hello", encoding="utf-8")
    mp_mcp.start_pipeline(
        f"filesrc location={text.as_posix()} ! text/x-raw,format=utf8 ! mp_metasink"
    )
    waited = mp_mcp.wait_for_records(1, timeout=RECORD_WAIT_SECONDS)
    assert [record["text"] for record in waited["records"]] == ["hello"]
    assert "state" in waited["status"]
    assert [record["text"] for record in mp_mcp.latest_metadata()] == ["hello"]
    assert mp_mcp.stop_pipeline() == {"state": "none"}


def test_element_messages_from_any_element_reach_latest_bus_messages():
    mp_mcp.start_pipeline("audiotestsrc num-buffers=20 ! level name=meter ! fakesink")
    mp_mcp.wait_for_records(1, timeout=RECORD_WAIT_SECONDS)
    readings = [
        message
        for message in mp_mcp.latest_bus_messages(RECENT_MESSAGES_READ)
        if message["source"] == "meter"
    ]
    assert readings
    assert all(reading["text"].startswith("level, ") for reading in readings)
    assert "rms=" in readings[0]["text"]
    mp_mcp.stop_pipeline()


def test_an_error_keeps_the_reason_gstreamer_gave(tmp_path):
    with pytest.raises(ToolError, match="No such file"):
        mp_mcp.start_pipeline(
            f"filesrc location={(tmp_path / 'missing').as_posix()} ! fakesink"
        )


def test_a_sink_dropping_late_frames_reports_qos():
    mp_mcp.start_pipeline(
        "videotestsrc ! video/x-raw,framerate=30/1 "
        f"! identity sleep-time={SLOW_ELEMENT_MICROSECONDS} "
        "! fakesink name=screen sync=true qos=true max-lateness=1"
    )
    time.sleep(FLOW_SETTLE_SECONDS)
    reports = [
        message
        for message in mp_mcp.latest_bus_messages(RECENT_MESSAGES_READ)
        if message["type"] == "qos"
    ]
    mp_mcp.stop_pipeline()
    assert reports
    assert reports[0]["source"] == "screen"
    assert "dropped" in reports[0]["text"]


def test_buffer_flow_shows_where_the_data_stops():
    mp_mcp.start_pipeline(
        "videotestsrc name=source ! video/x-raw,framerate=30/1 "
        "! valve name=gate drop=false ! fakesink sync=true"
    )
    time.sleep(FLOW_SETTLE_SECONDS)
    mp_mcp.set_property("gate", "drop", "true")
    time.sleep(FLOW_SETTLE_SECONDS)
    flows = {flow["pad"]: flow for flow in mp_mcp.buffer_flow()}
    mp_mcp.stop_pipeline()
    assert flows["gate.src"]["buffers"] > 0
    assert flows["gate.src"]["seconds_since_last"] >= FLOW_SETTLE_SECONDS / 2
    assert flows["source.src"]["seconds_since_last"] < FLOW_SETTLE_SECONDS / 2


def test_buffer_flow_measures_how_long_each_element_holds_a_buffer():
    mp_mcp.start_pipeline(
        "videotestsrc name=source ! video/x-raw,framerate=30/1 ! identity name=fast "
        f"! identity name=slow sleep-time={SLOW_ELEMENT_MICROSECONDS} ! fakesink"
    )
    time.sleep(FLOW_SETTLE_SECONDS)
    flows = {flow["pad"]: flow for flow in mp_mcp.buffer_flow()}
    mp_mcp.stop_pipeline()
    assert flows["slow.src"]["latency_ms"] >= SLOW_ELEMENT_MICROSECONDS / 1000
    assert flows["fast.src"]["latency_ms"] < flows["slow.src"]["latency_ms"]
    assert "latency_ms" not in flows["source.src"]


def test_element_latencies_ranks_elements_slowest_first_without_queues():
    mp_mcp.start_pipeline(
        "videotestsrc name=source ! video/x-raw,framerate=30/1 ! identity name=fast "
        f"! queue ! identity name=slow sleep-time={SLOW_ELEMENT_MICROSECONDS} "
        "! fakesink"
    )
    time.sleep(FLOW_SETTLE_SECONDS)
    flows = mp_mcp.buffer_flow()
    latencies = mp_mcp.element_latencies()
    mp_mcp.stop_pipeline()
    assert any(
        flow["pad"].startswith("queue") and "latency_ms" in flow for flow in flows
    )
    elements = [latency["element"] for latency in latencies]
    assert not any(element.startswith("queue") for element in elements)
    assert elements[0] == "slow"
    milliseconds = [latency["latency_ms"] for latency in latencies]
    assert milliseconds == sorted(milliseconds, reverse=True)


def test_buffer_flow_counts_a_pad_decodebin_adds(video_file):
    mp_mcp.start_pipeline(
        f"filesrc location={video_file.as_posix()} ! decodebin name=decoder ! fakesink"
    )
    mp_mcp.wait_for_records(1, timeout=RECORD_WAIT_SECONDS)
    flows = mp_mcp.buffer_flow()
    mp_mcp.stop_pipeline()
    decoded = [flow for flow in flows if flow["pad"].startswith("decoder.")]
    assert decoded
    assert decoded[0]["buffers"] == CLIP_SOURCE_FRAMES


def test_snapshot_frame_returns_a_jpeg_of_the_newest_frame():
    mp_mcp.start_pipeline(VIDEO_PIPELINE)
    mp_mcp.wait_for_records(1, timeout=FRAME_WAIT_SECONDS)
    for snapshot in (mp_mcp.snapshot_frame(), mp_mcp.snapshot_frame("sink")):
        assert snapshot.data.startswith(JPEG_MAGIC)
        assert snapshot.to_image_content().mime_type == "image/jpeg"
    mp_mcp.stop_pipeline()


def test_snapshot_frame_refuses_a_sink_that_carries_no_video(tmp_path):
    text = tmp_path / "hello.txt"
    text.write_text("hello", encoding="utf-8")
    mp_mcp.start_pipeline(
        f"filesrc location={text.as_posix()} ! text/x-raw,format=utf8 ! mp_metasink"
    )
    mp_mcp.wait_for_records(1, timeout=RECORD_WAIT_SECONDS)
    with pytest.raises(ToolError):
        mp_mcp.snapshot_frame()
    mp_mcp.stop_pipeline()


def test_snapshot_frame_names_the_element_it_cannot_find():
    mp_mcp.start_pipeline(VIDEO_PIPELINE)
    with pytest.raises(ToolError, match="no element named"):
        mp_mcp.snapshot_frame("nosuch")
    mp_mcp.stop_pipeline()


def test_wait_for_records_gives_up_on_a_key_no_record_carries():
    mp_mcp.start_pipeline(VIDEO_PIPELINE)
    started = time.monotonic()
    waited = mp_mcp.wait_for_records(1, key="alert", timeout=MISSING_KEY_WAIT_SECONDS)
    assert waited["records"] == []
    assert time.monotonic() - started < MISSING_KEY_DEADLINE_SECONDS
    mp_mcp.stop_pipeline()


def test_a_property_can_change_while_the_pipeline_runs():
    mp_mcp.start_pipeline(
        "videotestsrc name=source pattern=snow ! videobalance name=balance "
        "! mp_metasink name=sink"
    )
    assert mp_mcp.get_property("source", "pattern") == "snow"
    assert mp_mcp.set_property("source", "pattern", "ball") == {"pattern": "ball"}
    assert mp_mcp.set_property("balance", "contrast", "0.99") == {"contrast": "0.99"}
    assert mp_mcp.set_property("sink", "location", "a b") == {"location": "a b"}
    with pytest.raises(ToolError, match="no property"):
        mp_mcp.set_property("source", "no-such-property", "1")
    mp_mcp.stop_pipeline()


def test_a_sink_without_a_location_is_kept_off_stdout():
    mp_mcp.start_pipeline("videotestsrc ! mp_metasink name=sink")
    assert mp_mcp.get_property("sink", "location") == os.devnull
    mp_mcp.stop_pipeline()


def test_a_bad_description_is_reported_and_leaves_nothing_running():
    with pytest.raises(ToolError, match="nosuchelement"):
        mp_mcp.start_pipeline("nosuchelement ! fakesink")
    assert mp_mcp.pipeline_status() == {"state": "none"}


def test_a_looping_pipeline_seeks_instead_of_ending():
    mp_mcp.start_pipeline("videotestsrc num-buffers=5 ! fakesink sync=true", loop=True)
    time.sleep(1)
    status = mp_mcp.pipeline_status()
    mp_mcp.stop_pipeline()
    assert status["errors"] == []
    assert status["ended"] is False
    assert status["state"] == "playing"


def test_describe_frame_captions_the_newest_frame(monkeypatch):
    engine = StubVlmEngine()
    monkeypatch.setattr(mp_mcp, "vlm_engine", lambda model_name: engine)
    mp_mcp.start_pipeline(VIDEO_PIPELINE)
    mp_mcp.wait_for_records(1, timeout=FRAME_WAIT_SECONDS)
    assert mp_mcp.describe_frame("what is this", max_tokens=8) == "a test pattern"
    image, prompt, _system_prompt, max_tokens, temperature = engine.calls[0]
    assert (prompt, max_tokens, temperature) == ("what is this", 8, 0.0)
    assert image.size == (64, 48)
    assert image.mode == "RGB"
    jpeg = PillowImage.open(io.BytesIO(mp_mcp.snapshot_frame().data))
    difference = [
        abs(raw - encoded)
        for raw, encoded in zip(
            image.getpixel(YELLOW_BAR_PIXEL), jpeg.getpixel(YELLOW_BAR_PIXEL)
        )
    ]
    assert max(difference) <= JPEG_CHANNEL_TOLERANCE
    mp_mcp.stop_pipeline()


def test_describe_frame_drops_the_padding_at_the_end_of_a_row(monkeypatch):
    engine = StubVlmEngine()
    monkeypatch.setattr(mp_mcp, "vlm_engine", lambda model_name: engine)
    mp_mcp.start_pipeline(PADDED_VIDEO_PIPELINE)
    mp_mcp.wait_for_records(1, timeout=FRAME_WAIT_SECONDS)
    mp_mcp.describe_frame()
    image = engine.calls[0][0]
    assert image.size == (66, 48)
    assert image.getpixel(YELLOW_BAR_PIXEL) == YELLOW
    mp_mcp.stop_pipeline()


def test_describe_frame_refuses_a_sink_that_carries_no_video(tmp_path):
    text = tmp_path / "hello.txt"
    text.write_text("hello", encoding="utf-8")
    mp_mcp.start_pipeline(
        f"filesrc location={text.as_posix()} ! text/x-raw,format=utf8 ! mp_metasink"
    )
    mp_mcp.wait_for_records(1, timeout=RECORD_WAIT_SECONDS)
    with pytest.raises(ToolError, match="is not carrying video"):
        mp_mcp.describe_frame()
    mp_mcp.stop_pipeline()


def test_describe_frame_needs_a_running_pipeline():
    mp_mcp.stop_pipeline()
    with pytest.raises(ToolError, match="no pipeline is running"):
        mp_mcp.describe_frame()


def test_clip_at_cuts_the_seconds_around_a_pts(video_file):
    clip = mp_mcp.clip_at(str(video_file), 3.0, CLIP_SECONDS)
    assert (clip["start"], clip["end"]) == (2.0, 4.0)
    assert Path(clip["path"]).stat().st_size > 0
    assert LEAST_CLIP_FRAMES <= clip["frames"] <= MOST_CLIP_FRAMES
    discoverer = GstPbutils.Discoverer.new(CLIP_DISCOVER_SECONDS * Gst.SECOND)
    written = discoverer.discover_uri(Gst.filename_to_uri(clip["path"]))
    assert written.get_duration() / Gst.SECOND == pytest.approx(
        CLIP_SECONDS, abs=CLIP_DURATION_TOLERANCE
    )


def test_clip_at_names_the_video_it_cannot_find(tmp_path):
    with pytest.raises(ToolError, match="no video"):
        mp_mcp.clip_at(str(tmp_path / "missing.webm"), 1.0)


def test_load_metadata_stands_in_for_a_finished_run(tmp_path):
    recorded = tmp_path / "records.jsonl"
    recorded.write_text(
        '{"pts": 0.0, "text": "one"}\n\n{"pts": 0.1, "text": "two"}\n', encoding="utf-8"
    )
    assert mp_mcp.load_metadata(str(recorded)) == {"records": 2}
    assert [record["text"] for record in mp_mcp.latest_metadata()] == ["one", "two"]
    assert mp_mcp.pipeline_status() == {"state": "none"}
    started = time.monotonic()
    waited = mp_mcp.wait_for_records(1, timeout=LOADED_WAIT_SECONDS)
    assert [record["text"] for record in waited["records"]] == ["one", "two"]
    assert time.monotonic() - started < LOADED_DEADLINE_SECONDS


def test_load_metadata_names_the_file_it_cannot_find(tmp_path):
    with pytest.raises(ToolError, match="no metadata file"):
        mp_mcp.load_metadata(str(tmp_path / "missing.jsonl"))


def test_every_pipelines_section_is_offered_as_a_prompt():
    names = [prompt.name for prompt in asyncio.run(mp_mcp.server.list_prompts())]
    assert "object_detection" in names
    assert "metadata_sink" in names
    result = asyncio.run(mp_mcp.server.get_prompt("metadata_sink", None))
    assert "mp_metasink" in result.messages[0].content.text


def test_inspect_lists_the_sink_location():
    described = mp_mcp.inspect("mp_metasink")
    defaults = {spec["name"]: spec["default"] for spec in described["properties"]}
    assert defaults["location"] == ""
    assert defaults["last-sample"] == ""
