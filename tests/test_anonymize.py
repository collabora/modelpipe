import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo  # noqa: E402

Gst.init(None)

from anonymize import Anonymize  # noqa: E402
from backend import analytics  # noqa: E402
from backend.gst.cuda_memory import map_device_frame  # noqa: E402

WIDTH, HEIGHT = 96, 72
CHANNEL_COUNT = 4
COLOUR_CHANNEL_COUNT = 3
ALPHA_LEVEL = 173
PERSON_BOX = (20, 16, 32, 24)
CAR_BOX = (62, 18, 22, 20)
SYSTEM_CAPS = f"video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT}"
DEVICE_CAPS = (
    f"video/x-raw(memory:CUDAMemory),format=RGBA,width={WIDTH},height={HEIGHT}"
)
PULL_TIMEOUT = 10 * Gst.SECOND


def test_anonymizer_declares_what_frames_it_reads_and_writes():
    capabilities = Anonymize.FRAME_CAPABILITIES

    assert capabilities.memory_types == ("system", "cuda")
    assert capabilities.pixel_formats == ("RGBA",)
    assert capabilities.reads_pixels and capabilities.writes_pixels
    assert capabilities.synchronization == "call-completion"


def patterned_frame():
    rows, columns = np.indices((HEIGHT, WIDTH))
    frame = np.empty((HEIGHT, WIDTH, CHANNEL_COUNT), np.uint8)
    frame[..., 0] = (rows * 17 + columns * 7) % 256
    frame[..., 1] = (rows * 5 + columns * 19) % 256
    frame[..., 2] = (rows * 23 + columns * 3) % 256
    frame[..., 3] = ALPHA_LEVEL
    return frame


def system_buffer(frame):
    buffer = Gst.Buffer.new_allocate(None, frame.nbytes, None)
    buffer.fill(0, frame.tobytes())
    GstVideo.buffer_add_video_meta(
        buffer, GstVideo.VideoFrameFlags.NONE, GstVideo.VideoFormat.RGBA, WIDTH, HEIGHT
    )
    return buffer


def add_detections(buffer, detections):
    meta = analytics.add_relation_meta(buffer)
    for label, box in detections:
        analytics.add_object(meta, label, *box, 1.0)
    return meta


def configured_anonymizer(caps, **properties):
    anonymizer = Anonymize()
    parsed_caps = Gst.Caps.from_string(caps)
    anonymizer.do_set_caps(parsed_caps, parsed_caps)
    for name, value in properties.items():
        anonymizer.set_property(name.replace("_", "-"), value)
    return anonymizer


def system_pixels(buffer):
    pixels = buffer.extract_dup(0, WIDTH * HEIGHT * CHANNEL_COUNT)
    return np.frombuffer(pixels, np.uint8).reshape(HEIGHT, WIDTH, CHANNEL_COUNT)


def box_slices(box):
    x, y, width, height = box
    return slice(y, y + height), slice(x, x + width)


@pytest.mark.parametrize("mode", ["blur", "pixelate", "blackout"])
def test_each_mode_changes_only_the_selected_box(mode):
    original = patterned_frame()
    buffer = system_buffer(original)
    meta = add_detections(buffer, [("stream_0_person", PERSON_BOX)])
    detections = analytics.read_objects(meta)
    anonymizer = configured_anonymizer(SYSTEM_CAPS, mode=mode, box_expansion=0.0)

    assert anonymizer.do_transform_ip(buffer) == Gst.FlowReturn.OK

    changed = system_pixels(buffer)
    rows, columns = box_slices(PERSON_BOX)
    outside = np.ones((HEIGHT, WIDTH), dtype=bool)
    outside[rows, columns] = False
    assert np.any(
        changed[rows, columns, :COLOUR_CHANNEL_COUNT]
        != original[rows, columns, :COLOUR_CHANNEL_COUNT]
    )
    assert np.array_equal(changed[outside], original[outside])
    assert np.array_equal(changed[..., 3], original[..., 3])
    assert analytics.read_objects(meta) == detections


def test_blackout_is_black_inside_the_expanded_clipped_box():
    original = patterned_frame()
    buffer = system_buffer(original)
    add_detections(buffer, [("stream_0_person", (-4, -3, 20, 18))])
    anonymizer = configured_anonymizer(SYSTEM_CAPS, mode="blackout", box_expansion=0.5)

    assert anonymizer.do_transform_ip(buffer) == Gst.FlowReturn.OK

    changed = system_pixels(buffer)
    assert not changed[:24, :26, :COLOUR_CHANNEL_COUNT].any()
    assert np.array_equal(changed[30:, 30:], original[30:, 30:])


def test_the_label_filter_leaves_other_detections_alone():
    original = patterned_frame()
    buffer = system_buffer(original)
    add_detections(
        buffer,
        [("stream_0_person_id_4", PERSON_BOX), ("stream_0_car_id_8", CAR_BOX)],
    )
    anonymizer = configured_anonymizer(
        SYSTEM_CAPS, mode="blackout", labels="person", box_expansion=0.0
    )

    assert anonymizer.do_transform_ip(buffer) == Gst.FlowReturn.OK

    changed = system_pixels(buffer)
    person_rows, person_columns = box_slices(PERSON_BOX)
    car_rows, car_columns = box_slices(CAR_BOX)
    assert not changed[person_rows, person_columns, :COLOUR_CHANNEL_COUNT].any()
    assert np.array_equal(
        changed[car_rows, car_columns], original[car_rows, car_columns]
    )


def test_overlapping_boxes_black_out_their_union():
    original = patterned_frame()
    buffer = system_buffer(original)
    add_detections(
        buffer,
        [
            ("stream_0_person", (12, 10, 24, 20)),
            ("stream_0_person", (28, 22, 26, 18)),
        ],
    )
    anonymizer = configured_anonymizer(SYSTEM_CAPS, mode="blackout", box_expansion=0.0)

    assert anonymizer.do_transform_ip(buffer) == Gst.FlowReturn.OK

    changed = system_pixels(buffer)
    selected = np.zeros((HEIGHT, WIDTH), dtype=bool)
    selected[10:30, 12:36] = True
    selected[22:40, 28:54] = True
    assert not changed[selected, :COLOUR_CHANNEL_COUNT].any()
    assert np.array_equal(changed[~selected], original[~selected])


def test_edge_fade_keeps_the_region_edge_and_changes_its_centre():
    original = patterned_frame()
    buffer = system_buffer(original)
    add_detections(buffer, [("stream_0_person", PERSON_BOX)])
    anonymizer = configured_anonymizer(
        SYSTEM_CAPS,
        mode="blackout",
        box_expansion=0.0,
        edge_fade=6,
    )

    assert anonymizer.do_transform_ip(buffer) == Gst.FlowReturn.OK

    changed = system_pixels(buffer)
    x, y, width, height = PERSON_BOX
    assert np.array_equal(changed[y, x : x + width], original[y, x : x + width])
    assert not changed[y + height // 2, x + width // 2, :COLOUR_CHANNEL_COUNT].any()


def test_no_detections_leave_the_frame_unchanged():
    original = patterned_frame()
    buffer = system_buffer(original)
    anonymizer = configured_anonymizer(SYSTEM_CAPS)

    assert anonymizer.do_transform_ip(buffer) == Gst.FlowReturn.OK
    assert np.array_equal(system_pixels(buffer), original)


def uploaded_buffer(frame):
    source_caps = f"{SYSTEM_CAPS},framerate=30/1"
    pipeline = Gst.parse_launch(
        f"appsrc name=source format=time caps={source_caps} ! cudaupload "
        f"! {DEVICE_CAPS} ! appsink name=sink sync=false"
    )
    pipeline.set_state(Gst.State.PLAYING)
    try:
        pipeline.get_by_name("source").emit("push-buffer", system_buffer(frame))
        sample = pipeline.get_by_name("sink").emit("try-pull-sample", PULL_TIMEOUT)
        assert sample is not None
        return sample.get_buffer().copy_deep()
    finally:
        pipeline.set_state(Gst.State.NULL)


def on_streaming_thread(function, *arguments):
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(function, *arguments).result()


def device_pixels(buffer):
    device_map = map_device_frame(buffer, HEIGHT, WIDTH, CHANNEL_COUNT)
    try:
        return device_map.tensor.cpu().numpy()
    finally:
        device_map.unmap()


def test_cuda_anonymization_never_uses_the_system_frame_map(monkeypatch):
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() and Gst.ElementFactory.find("cudaupload")):
        pytest.skip("needs a cuda device and the cudaupload element")
    import backend.gst.in_place_video_transform as in_place_video_transform

    original = patterned_frame()
    buffer = uploaded_buffer(original)
    add_detections(buffer, [("stream_0_person", PERSON_BOX)])
    anonymizer = configured_anonymizer(DEVICE_CAPS, mode="pixelate", box_expansion=0.0)

    def refuse_system_map(*_arguments, **_keywords):
        raise AssertionError("the CUDA frame was mapped as system memory")

    monkeypatch.setattr(in_place_video_transform, "mapped_frame", refuse_system_map)

    assert on_streaming_thread(anonymizer.do_transform_ip, buffer) == Gst.FlowReturn.OK
    changed = on_streaming_thread(device_pixels, buffer)
    rows, columns = box_slices(PERSON_BOX)
    outside = np.ones((HEIGHT, WIDTH), dtype=bool)
    outside[rows, columns] = False
    assert np.any(
        changed[rows, columns, :COLOUR_CHANNEL_COUNT]
        != original[rows, columns, :COLOUR_CHANNEL_COUNT]
    )
    assert np.array_equal(changed[outside], original[outside])
