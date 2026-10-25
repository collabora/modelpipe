import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)
torch = pytest.importorskip("torch")

from backend import frameio  # noqa: E402
from backend.gst.cuda_memory import map_device_frame  # noqa: E402
from yolo_reference import PORTRAIT  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and Gst.ElementFactory.find("cudaupload")),
    reason="needs a cuda device and the cudaupload element",
)

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_CORRELATION_FLOOR = 0.98
# the cuda resize of the model input puts a fifth of the depth levels one off the cpu one
MATCHING_COLOUR_FRACTION = 0.75
LEVELS_WITHIN_ONE_FRACTION = 0.99
WIDTH, HEIGHT = 640, 480
PORTRAIT_WIDTH, PORTRAIT_HEIGHT = 800, 600
RGBA_CHANNELS = 4
RGB_CHANNELS = 3
ALPHA_CHANNEL = 3
WRITTEN_ALPHA = 77
LEVEL_COUNT = 256
PULL_TIMEOUT = 10 * Gst.SECOND
# the depth elements load their models before the first frame comes out
MODEL_PULL_TIMEOUT = 120 * Gst.SECOND
DEVICE_CAPS = "video/x-raw(memory:CUDAMemory),format=RGBA"


def pull_tee_buffers(source, branch, timeout=PULL_TIMEOUT):
    pipeline = Gst.parse_launch(
        f"{source} ! tee name=split "
        f"split. ! queue ! cudaupload ! cudaconvert ! {DEVICE_CAPS} {branch}"
        "! appsink name=device_sink sync=false "
        f"split. ! queue {branch}! appsink name=system_sink sync=false"
    )
    pipeline.set_state(Gst.State.PLAYING)
    buffers = {}
    try:
        for name in ("device_sink", "system_sink"):
            sample = pipeline.get_by_name(name).emit("try-pull-sample", timeout)
            assert sample is not None
            buffers[name] = sample.get_buffer().copy_deep()
    finally:
        pipeline.set_state(Gst.State.NULL)
    return buffers["device_sink"], buffers["system_sink"]


@pytest.fixture
def test_frame_buffers():
    return pull_tee_buffers(
        "videotestsrc num-buffers=1 "
        f"! video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT}",
        "",
    )


def pattern(channels):
    rows, columns, channel_indices = np.indices((HEIGHT, WIDTH, channels))
    return ((rows * 3 + columns * 5 + channel_indices * 7) % LEVEL_COUNT).astype(
        np.uint8
    )


def on_a_streaming_thread(function, *arguments):
    # a streaming thread has no cuda context current
    with ThreadPoolExecutor(max_workers=1) as streaming_thread:
        return streaming_thread.submit(function, *arguments).result()


def device_pixels(buffer, width=WIDTH, height=HEIGHT):
    device_map = map_device_frame(buffer, height, width, RGBA_CHANNELS)
    try:
        return device_map.tensor.cpu().numpy()
    finally:
        device_map.unmap()


def system_pixels(buffer, width=WIDTH, height=HEIGHT):
    frame = np.frombuffer(
        buffer.extract_dup(0, width * height * RGBA_CHANNELS), np.uint8
    )
    return frame.reshape(height, width, RGBA_CHANNELS)


def write_then_read(buffer, frame):
    assert frameio.write_frame(buffer, frame)
    return device_pixels(buffer)


@pytest.mark.parametrize("on_the_device", [True, False])
def test_a_written_frame_reads_back_as_the_system_write(
    test_frame_buffers, on_the_device
):
    device_buffer, system_buffer = test_frame_buffers
    frame = pattern(RGBA_CHANNELS)
    written = torch.from_numpy(frame).cuda() if on_the_device else frame

    read_back = on_a_streaming_thread(write_then_read, device_buffer, written)
    assert frameio.write_frame(system_buffer, frame)

    assert np.array_equal(read_back, system_pixels(system_buffer))
    assert np.array_equal(read_back, frame)


def test_a_frame_of_another_height_is_refused(test_frame_buffers):
    device_buffer, _ = test_frame_buffers
    frame = torch.zeros(HEIGHT // 2, WIDTH, RGBA_CHANNELS, dtype=torch.uint8)

    with pytest.raises(ValueError, match="does not fit"):
        on_a_streaming_thread(frameio.write_frame, device_buffer, frame.cuda())


def test_a_colour_frame_leaves_the_alpha_channel_as_is(test_frame_buffers):
    device_buffer, _ = test_frame_buffers
    with_alpha = np.full((HEIGHT, WIDTH, RGBA_CHANNELS), WRITTEN_ALPHA, np.uint8)
    on_a_streaming_thread(write_then_read, device_buffer, with_alpha)
    colour = pattern(RGB_CHANNELS)

    read_back = on_a_streaming_thread(
        write_then_read, device_buffer, torch.from_numpy(colour).cuda()
    )

    assert np.array_equal(read_back[..., :RGB_CHANNELS], colour)
    assert (read_back[..., ALPHA_CHANNEL] == WRITTEN_ALPHA).all()


@pytest.fixture(scope="module")
def depth_buffers():
    from depth import DepthTransform

    Gst.Element.register(None, "depthundertest", Gst.Rank.NONE, DepthTransform)
    return pull_tee_buffers(
        f"filesrc location={PORTRAIT.as_posix()} ! jpegdec ! videoconvert "
        "! video/x-raw,format=RGBA",
        f"! depthundertest model-name={DEPTH_MODEL} engine-name=pytorch "
        "device=cuda ",
        MODEL_PULL_TIMEOUT,
    )


def blob_depth(buffer):
    depth = np.frombuffer(frameio.read_blobs(buffer)["depth"], np.uint8)
    return depth.reshape(PORTRAIT_HEIGHT, PORTRAIT_WIDTH)


def test_depth_on_a_cuda_frame_matches_depth_on_the_system_frame(depth_buffers):
    device_buffer, system_buffer = depth_buffers

    device_depth, system_depth = blob_depth(device_buffer), blob_depth(system_buffer)

    correlation = np.corrcoef(device_depth.ravel(), system_depth.ravel())[0, 1]
    assert correlation > DEPTH_CORRELATION_FLOOR
    level_difference = np.abs(device_depth.astype(int) - system_depth)
    assert (level_difference <= 1).mean() > LEVELS_WITHIN_ONE_FRACTION


def test_the_depth_colours_written_on_the_device_match_the_system_colours(
    depth_buffers,
):
    import cv2
    from tasks.depth import COLORMAP_IDS

    device_buffer, system_buffer = depth_buffers

    device_frame = on_a_streaming_thread(
        device_pixels, device_buffer, PORTRAIT_WIDTH, PORTRAIT_HEIGHT
    )
    system_frame = system_pixels(system_buffer, PORTRAIT_WIDTH, PORTRAIT_HEIGHT)

    levels = np.arange(LEVEL_COUNT, dtype=np.uint8).reshape(LEVEL_COUNT, 1)
    default_colours = cv2.applyColorMap(levels, COLORMAP_IDS["inferno"])
    colours = cv2.cvtColor(default_colours, cv2.COLOR_BGR2RGBA)
    assert np.array_equal(device_frame, colours[blob_depth(device_buffer), 0])
    matching = (device_frame == system_frame).all(axis=-1)
    assert matching.mean() > MATCHING_COLOUR_FRACTION


def test_the_colormap_table_names_the_opencv_colormaps():
    import cv2

    from tasks.depth import COLORMAP_IDS

    for name, identifier in COLORMAP_IDS.items():
        assert identifier == getattr(cv2, f"COLORMAP_{name.upper()}")
