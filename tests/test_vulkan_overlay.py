import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ["GST_PLUGIN_PATH"] = str(BASE_DIR / "plugins")
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
try:
    gi.require_version("GstVulkan", "1.0")
except ValueError:
    pytest.skip("the GstVulkan typelib is not installed", allow_module_level=True)
from gi.repository import Gst  # noqa: E402

Gst.init(None)

from overlay import Overlay  # noqa: E402

WIDTH, HEIGHT = 160, 120
LEFT, TOP, RIGHT, BOTTOM = 40, 40, 100, 80
PULL_TIMEOUT = 20 * Gst.SECOND
OVERLAY_ELEMENT = "mp_overlay"
if Gst.ElementFactory.find(OVERLAY_ELEMENT) is None:
    OVERLAY_ELEMENT = "vulkanoverlayundertest"
    Gst.Element.register(None, OVERLAY_ELEMENT, Gst.Rank.NONE, Overlay)

pytestmark = pytest.mark.skipif(
    any(
        Gst.ElementFactory.find(name) is None
        for name in (OVERLAY_ELEMENT, "vulkanupload", "vulkandownload")
    ),
    reason="the Vulkan overlay elements are unavailable",
)


LABELS = ("person", "person", "car")


def pull_vulkan_overlay(metadata_path, video_format):
    pipeline = Gst.parse_launch(
        "appsrc name=source format=time "
        f"caps=video/x-raw,format={video_format},width={WIDTH},height={HEIGHT},framerate=30/1 "
        "! vulkanupload "
        f"! video/x-raw(memory:VulkanImage),format={video_format} "
        f"! {OVERLAY_ELEMENT} name=overlay tracking=false "
        f"! vulkandownload ! video/x-raw,format={video_format} "
        "! appsink name=sink sync=false"
    )
    pipeline.get_by_name("overlay").set_property("meta-path", str(metadata_path))
    source = pipeline.get_by_name("source")
    sink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        black = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
        black[..., 3] = 255
        for _ in LABELS:
            assert (
                source.emit("push-buffer", Gst.Buffer.new_wrapped(black.tobytes()))
                == Gst.FlowReturn.OK
            )
        source.emit("end-of-stream")
        frames = []
        for _ in LABELS:
            sample = sink.emit("try-pull-sample", PULL_TIMEOUT)
            if sample is None:
                message = pipeline.get_bus().timed_pop_filtered(
                    0, Gst.MessageType.ERROR
                )
                if message is not None:
                    error, debug = message.parse_error()
                    pytest.fail(f"{error.message}\n{debug}")
            assert sample is not None, "the Vulkan overlay produced too few frames"
            frames.append(
                np.frombuffer(
                    sample.get_buffer().extract_dup(0, WIDTH * HEIGHT * 4), np.uint8
                ).reshape(HEIGHT, WIDTH, 4)
            )
        return frames
    finally:
        pipeline.set_state(Gst.State.NULL)


@pytest.mark.parametrize(
    "video_format, red_channel, blue_channel", [("RGBA", 0, 2), ("BGRA", 2, 0)]
)
def test_vulkan_overlay_draws_box_and_label_on_the_gpu(
    monkeypatch, tmp_path, video_format, red_channel, blue_channel
):
    import backend

    def refuse_system_map(*_arguments, **_keywords):
        raise AssertionError("the Vulkan video frame was mapped into system memory")

    monkeypatch.setattr(backend.frameio, "read_frame_for_drawing", refuse_system_map)
    box = {"x1": LEFT, "y1": TOP, "x2": RIGHT, "y2": BOTTOM}
    metadata_path = tmp_path / "detections.json"
    metadata_path.write_text(
        json.dumps(
            {
                "frames": [
                    {"frame_index": index, "objects": [{"label": label, "box": box}]}
                    for index, label in enumerate(LABELS)
                ]
            }
        )
    )

    frames = pull_vulkan_overlay(metadata_path, video_format)

    labels = []
    for frame in frames:
        outline = frame[TOP, LEFT + 5 : RIGHT - 5]
        assert (outline[:, red_channel] == 255).all()
        assert not outline[:, blue_channel].any()
        assert not frame[TOP + 4 : BOTTOM - 4, LEFT + 4 : RIGHT - 4, :3].any()
        assert not frame[TOP - 30 :, RIGHT + 4 :, :3].any()
        labels.append(frame[TOP - 30 : TOP - 2, LEFT : LEFT + 60, red_channel])
        assert (labels[-1] > 128).sum() > 20
    assert (labels[0] == labels[1]).all()
    assert (labels[0] != labels[2]).any()
