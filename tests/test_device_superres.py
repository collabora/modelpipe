import sys
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
pytest.importorskip("spandrel")

from test_device_write import (  # noqa: E402
    ALPHA_CHANNEL,
    MODEL_PULL_TIMEOUT,
    RGB_CHANNELS,
    device_pixels,
    on_a_streaming_thread,
    pull_tee_buffers,
    system_pixels,
)
from yolo_reference import PORTRAIT  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and Gst.ElementFactory.find("cudaupload")),
    reason="needs a cuda device and the cudaupload element",
)

WIDTH, HEIGHT = 320, 240
SUPERRES_MODEL = "real-esrgan-x2"
# bicubic on the device against lanczos on the host
MEAN_LEVEL_DIFFERENCE_CEILING = 3.0
CORRELATION_FLOOR = 0.99
OPAQUE = 255


@pytest.fixture(scope="module")
def superres_buffers():
    from superres import SuperResTransform

    Gst.Element.register(None, "superresundertest", Gst.Rank.NONE, SuperResTransform)
    return pull_tee_buffers(
        f"filesrc location={PORTRAIT.as_posix()} ! jpegdec ! videoconvert ! videoscale "
        f"! video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT}",
        f"! superresundertest model-name={SUPERRES_MODEL} engine-name=pytorch "
        "device=cuda ",
        MODEL_PULL_TIMEOUT,
    )


@pytest.fixture(scope="module")
def superres_frames(superres_buffers):
    device_buffer, system_buffer = superres_buffers
    device_frame = on_a_streaming_thread(device_pixels, device_buffer, WIDTH, HEIGHT)
    return device_frame, system_pixels(system_buffer, WIDTH, HEIGHT)


def test_superres_on_a_cuda_frame_matches_superres_on_the_system_frame(
    superres_frames,
):
    device_frame, system_frame = superres_frames

    device_colour = device_frame[..., :RGB_CHANNELS].astype(float)
    system_colour = system_frame[..., :RGB_CHANNELS].astype(float)
    assert np.abs(device_colour - system_colour).mean() < MEAN_LEVEL_DIFFERENCE_CEILING
    correlation = np.corrcoef(device_colour.ravel(), system_colour.ravel())[0, 1]
    assert correlation > CORRELATION_FLOOR


def test_superres_leaves_the_alpha_channel_of_the_cuda_frame(superres_frames):
    device_frame, _ = superres_frames

    assert (device_frame[..., ALPHA_CHANNEL] == OPAQUE).all()
