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

from cudatovulkan import CudaToVulkan  # noqa: E402
from overlay import Overlay  # noqa: E402

WIDTH, HEIGHT = 160, 120
FRAME_COUNT = 3
PULL_TIMEOUT = 20 * Gst.SECOND
ELEMENTS = {"mp_cudatovulkan": CudaToVulkan, "mp_overlay": Overlay}
for name, element_class in list(ELEMENTS.items()):
    if Gst.ElementFactory.find(name) is None:
        ELEMENTS[name] = f"{name}undertest"
        Gst.Element.register(None, ELEMENTS[name], Gst.Rank.NONE, element_class)
    else:
        ELEMENTS[name] = name


def cuda_works():
    from backend.cuda_driver import call_cuda_driver

    try:
        call_cuda_driver("cuInit", 0)
    except (OSError, RuntimeError):
        return False
    return True


pytestmark = pytest.mark.skipif(
    any(
        Gst.ElementFactory.find(name) is None
        for name in ("cudaupload", "vulkandownload")
    )
    or not cuda_works(),
    reason="CUDA or the GStreamer CUDA and Vulkan elements are unavailable",
)


def gradient_frame(index, video_format):
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    pixels = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    pixels[..., 0] = columns + index
    pixels[..., 1] = rows
    pixels[..., 2] = 200
    pixels[..., 3] = 255
    return pixels


def run(middle, video_format):
    pipeline = Gst.parse_launch(
        "appsrc name=source format=time "
        f"caps=video/x-raw,format={video_format},width={WIDTH},height={HEIGHT},framerate=30/1 "
        f"! cudaupload ! video/x-raw(memory:CUDAMemory),format={video_format} "
        f"! {ELEMENTS['mp_cudatovulkan']} {middle}"
        f"! vulkandownload ! video/x-raw,format={video_format} "
        "! appsink name=sink sync=false"
    )
    source = pipeline.get_by_name("source")
    sink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        for index in range(FRAME_COUNT):
            frame = gradient_frame(index, video_format)
            assert (
                source.emit("push-buffer", Gst.Buffer.new_wrapped(frame.tobytes()))
                == Gst.FlowReturn.OK
            )
        source.emit("end-of-stream")
        frames = []
        for _ in range(FRAME_COUNT):
            sample = sink.emit("try-pull-sample", PULL_TIMEOUT)
            if sample is None:
                message = pipeline.get_bus().timed_pop_filtered(
                    0, Gst.MessageType.ERROR
                )
                if message is not None:
                    error, debug = message.parse_error()
                    pytest.fail(f"{error.message}\n{debug}")
            assert sample is not None, "too few frames reached the sink"
            frames.append(
                np.frombuffer(
                    sample.get_buffer().extract_dup(0, WIDTH * HEIGHT * 4), np.uint8
                ).reshape(HEIGHT, WIDTH, 4)
            )
        return frames
    finally:
        pipeline.set_state(Gst.State.NULL)


@pytest.mark.parametrize("video_format", ["RGBA", "BGRA"])
def test_cuda_frames_reach_vulkan_unchanged(video_format):
    frames = run("", video_format)

    for index, frame in enumerate(frames):
        assert (frame == gradient_frame(index, video_format)).all()


def test_vulkan_overlay_draws_on_frames_copied_from_cuda(tmp_path):
    left, top, right, bottom = 40, 40, 100, 80
    metadata_path = tmp_path / "detections.json"
    metadata_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_index": index,
                        "objects": [
                            {
                                "label": "",
                                "box": {
                                    "x1": left,
                                    "y1": top,
                                    "x2": right,
                                    "y2": bottom,
                                },
                            }
                        ],
                    }
                    for index in range(FRAME_COUNT)
                ]
            }
        )
    )

    frames = run(
        f"! {ELEMENTS['mp_overlay']} meta-path={metadata_path} tracking=false ",
        "RGBA",
    )

    for index, frame in enumerate(frames):
        expected = gradient_frame(index, "RGBA")
        outline = frame[top, left + 5 : right - 5]
        assert (outline == (255, 0, 0, 255)).all()
        inside = (slice(top + 4, bottom - 4), slice(left + 4, right - 4))
        assert (frame[inside] == expected[inside]).all()
