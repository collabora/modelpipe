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
    Gst.ElementFactory.find("mp_streammux") is None,
    reason="the python plugin loader did not register the modelpipe elements",
)

FRAMES_PER_SOURCE = (5, 3)
PIPELINE_TIMEOUT = 30 * Gst.SECOND


def test_the_longer_source_keeps_its_frames_after_the_shorter_one_ends():
    caps = "video/x-raw,width=16,height=8,format=RGB"
    sources = " ".join(
        f"videotestsrc num-buffers={frames} ! {caps} ! mux.sink_{index}"
        for index, frames in enumerate(FRAMES_PER_SOURCE)
    )
    sinks = " ".join(
        f"demux.src_{index} ! queue ! fakesink name=sink_{index} signal-handoffs=true"
        for index in range(len(FRAMES_PER_SOURCE))
    )
    pipeline = Gst.parse_launch(
        f"mp_streammux name=mux ! mp_streamdemux name=demux {sources} {sinks}"
    )
    frames_received = [0] * len(FRAMES_PER_SOURCE)
    for index in range(len(FRAMES_PER_SOURCE)):

        def count(sink, buffer, pad, index=index):
            frames_received[index] += 1

        pipeline.get_by_name(f"sink_{index}").connect("handoff", count)

    pipeline.set_state(Gst.State.PLAYING)
    message = pipeline.get_bus().timed_pop_filtered(
        PIPELINE_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)
    assert message is not None, "pipeline did not finish"
    if message.type == Gst.MessageType.ERROR:
        error, debug = message.parse_error()
        pytest.fail(f"{error.message}\n{debug}")

    assert tuple(frames_received) == FRAMES_PER_SOURCE


BATCH_COUNT = 5


def test_each_batch_runs_later_than_the_one_before():
    caps = "video/x-raw,width=16,height=8,format=RGB,framerate=50/1"
    pipeline = Gst.parse_launch(
        f"videotestsrc num-buffers={BATCH_COUNT} ! {caps} ! mp_streammux name=mux "
        "! fakesink name=sink signal-handoffs=true"
    )
    running_times = []

    def record(sink, buffer, pad):
        segment = pad.get_sticky_event(Gst.EventType.SEGMENT, 0).parse_segment()
        running_times.append(segment.to_running_time(Gst.Format.TIME, buffer.pts))

    pipeline.get_by_name("sink").connect("handoff", record)
    pipeline.set_state(Gst.State.PLAYING)
    message = pipeline.get_bus().timed_pop_filtered(
        PIPELINE_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)
    assert message is not None and message.type == Gst.MessageType.EOS

    assert len(running_times) == BATCH_COUNT
    assert running_times == sorted(set(running_times)), running_times


def test_the_demux_keeps_each_detections_track_id():
    from backend import analytics
    from streamdemux import StreamDemux

    buffer = Gst.Buffer.new_wrapped(bytes(16))
    meta = analytics.add_relation_meta(buffer)
    person = analytics.add_object(meta, "stream_0_person", 1, 2, 3, 4, 0.9)
    analytics.relate(meta, person, analytics.add_tracking(meta, 7))
    analytics.add_object(meta, "stream_0_car", 5, 6, 7, 8, 0.5)
    analytics.add_object(meta, "stream_1_dog", 9, 10, 11, 12, 0.25)

    out_buffer = StreamDemux().process_src_pad(
        buffer, buffer.get_memory(0), batch_position=0, source_index=2
    )

    found = analytics.read_objects(analytics.get_relation_meta(out_buffer))
    assert [(obj["label"], obj["stream"], obj["track_id"]) for obj in found] == [
        ("person", 2, 7),
        ("car", 2, None),
    ]
