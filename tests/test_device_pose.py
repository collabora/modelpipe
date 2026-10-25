import json
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
pytest.importorskip("ultralytics")

from backend import frameio  # noqa: E402
from test_device_write import (  # noqa: E402
    DEVICE_CAPS,
    MODEL_PULL_TIMEOUT,
    PORTRAIT_HEIGHT,
    PORTRAIT_WIDTH,
    RGB_CHANNELS,
    device_pixels,
    on_a_streaming_thread,
    system_pixels,
)
from yolo_reference import PORTRAIT, YOLO_POSE_MODEL  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and Gst.ElementFactory.find("cudaupload")),
    reason="needs a cuda device and the cudaupload element",
)

# a limb crossing the midpoint of another paints it in its own colour
MATCHING_LIMB_FRACTION = 0.9
NEIGHBOURHOOD = 1
# squares for circles and no antialiasing change the drawn pixel count a little
DRAWN_PIXEL_COUNT_RATIO = (0.6, 1.3)


@pytest.fixture(scope="module")
def pose_frames():
    from pose import KEYPOINT_RADIUS, YOLOPoseTransform

    Gst.Element.register(None, "poseundertest", Gst.Rank.NONE, YOLOPoseTransform)
    pipeline = Gst.parse_launch(
        f"filesrc location={PORTRAIT.as_posix()} ! jpegdec ! videoconvert "
        "! video/x-raw,format=RGBA ! tee name=split "
        "split. ! queue ! appsink name=source_sink sync=false "
        f"split. ! queue ! cudaupload ! cudaconvert ! {DEVICE_CAPS} "
        f"! poseundertest model-name={YOLO_POSE_MODEL} engine-name=pytorch "
        "device=cuda ! appsink name=device_sink sync=false"
    )
    pipeline.set_state(Gst.State.PLAYING)
    buffers = {}
    try:
        for name in ("source_sink", "device_sink"):
            sample = pipeline.get_by_name(name).emit(
                "try-pull-sample", MODEL_PULL_TIMEOUT
            )
            assert sample is not None
            buffers[name] = sample.get_buffer().copy_deep()
    finally:
        pipeline.set_state(Gst.State.NULL)
    persons = json.loads(frameio.read_blobs(buffers["device_sink"])["pose"])
    assert persons
    source = system_pixels(buffers["source_sink"], PORTRAIT_WIDTH, PORTRAIT_HEIGHT)
    device = on_a_streaming_thread(
        device_pixels, buffers["device_sink"], PORTRAIT_WIDTH, PORTRAIT_HEIGHT
    )
    return (
        source[..., :RGB_CHANNELS],
        device[..., :RGB_CHANNELS],
        persons,
        ultralytics_drawing(source[..., :RGB_CHANNELS], persons, KEYPOINT_RADIUS),
    )


def ultralytics_drawing(source, persons, radius):
    from ultralytics.utils.plotting import Annotator

    annotator = Annotator(np.ascontiguousarray(source[..., ::-1]))
    for person in persons:
        keypoints = np.array(person["keypoints"], np.float32)
        confidences = np.array(person["kp_conf"], np.float32)[:, None]
        annotator.kpts(np.hstack([keypoints, confidences]), radius=radius)
    return annotator.result()[..., ::-1]


def style():
    from ultralytics.utils.plotting import Annotator

    return Annotator(np.zeros((PORTRAIT_HEIGHT, PORTRAIT_WIDTH, RGB_CHANNELS)))


def rgb(palette_row):
    return tuple(int(channel) for channel in palette_row[::-1])


def drawn_keypoints(persons):
    from pose import DRAWN_KEYPOINT_CONFIDENCE

    for person in persons:
        seen = [
            confidence >= DRAWN_KEYPOINT_CONFIDENCE for confidence in person["kp_conf"]
        ]
        yield person["keypoints"], seen


def limbs(persons):
    for keypoints, seen in drawn_keypoints(persons):
        for index, (first, second) in enumerate(style().skeleton):
            if seen[first - 1] and seen[second - 1]:
                yield keypoints[first - 1], keypoints[second - 1], index


def test_every_keypoint_centre_is_painted_in_its_colour_or_a_limb_colour(pose_frames):
    _, device, persons, _ = pose_frames
    drawing = style()

    for keypoints, seen in drawn_keypoints(persons):
        for index, (x, y) in enumerate(keypoints):
            if not seen[index]:
                continue
            allowed = {rgb(drawing.kpt_color[index])} | {
                rgb(drawing.limb_color[limb])
                for limb, (first, second) in enumerate(drawing.skeleton)
                if index + 1 in (first, second) and seen[first - 1] and seen[second - 1]
            }
            assert tuple(device[int(y), int(x)].tolist()) in allowed


def test_every_limb_is_painted_in_its_colour_at_its_midpoint(pose_frames):
    from pose import KEYPOINT_RADIUS

    _, device, persons, _ = pose_frames
    drawing = style()

    matched, checked = 0, 0
    for (first_x, first_y), (second_x, second_y), index in limbs(persons):
        if max(abs(second_x - first_x), abs(second_y - first_y)) <= 2 * (
            KEYPOINT_RADIUS + NEIGHBOURHOOD
        ):
            continue
        checked += 1
        line, column = int((first_y + second_y) / 2), int((first_x + second_x) / 2)
        around = device[
            line - NEIGHBOURHOOD : line + NEIGHBOURHOOD + 1,
            column - NEIGHBOURHOOD : column + NEIGHBOURHOOD + 1,
        ].reshape(-1, RGB_CHANNELS)
        matched += any(
            tuple(pixel.tolist()) == rgb(drawing.limb_color[index]) for pixel in around
        )
    assert checked
    assert matched / checked >= MATCHING_LIMB_FRACTION


def test_the_device_paints_as_many_pixels_as_ultralytics(pose_frames):
    source, device, _, expected = pose_frames

    device_count = (device != source).any(axis=-1).sum()
    expected_count = (expected != source).any(axis=-1).sum()

    assert expected_count
    low, high = DRAWN_PIXEL_COUNT_RATIO
    assert low <= device_count / expected_count <= high
