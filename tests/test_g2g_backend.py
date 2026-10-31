"""Unit tests for the g2g element backend (`MP_BACKEND=g2g`).

These exercise the backend with no GStreamer present: the backend selection, the
`GObject` / `FlowReturn` shims, the `G2gFrameIO` buffer round-trip, the
`G2gAnalyticsBackend` mapping onto a flat sink, and an end-to-end `g2g_process`
on a `VideoTransform` subclass. The g2g host's `FrameBuffer` (a writable
buffer-protocol view) and `MetaSink` (write-only staging) are stubbed with a
`bytearray` and a recording object, so no Rust host is needed.

The payload tests at the end drive one text element through both backend
drivers, so they need GStreamer for the gst half and skip without it.
"""

import ctypes
import importlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "plugins" / "python"

# Select the g2g backend before importing `backend`, and make the plugin package
# importable.
os.environ["MP_BACKEND"] = "g2g"
sys.path.insert(0, str(PLUGIN_DIR))

import backend  # noqa: E402
from backend import (
    GObject,
    FlowReturn,
    frameio,
    analytics,
    BaseAggregator,
    VideoTransform,
    CUDA_INFERENCE_FRAME_CAPABILITIES,
)  # noqa: E402


class StubMetaSink:
    """Stand-in for the host's `g2g.MetaSink`.

    Like the real sink, every `add_*` stages one record into a single list and
    returns its index, which is the handle `relate` takes, and the read methods
    hand back the upstream metadata the host fills the sink with before the
    call, which is what the constructor takes.
    """

    def __init__(
        self,
        upstream_objects=(),
        upstream_blobs=None,
        upstream_class_names=(),
        upstream_tracking_ids=(),
        text_output=False,
    ):
        self.text_output = text_output
        self.emitted_outputs = []
        self.staged = []
        self.relations = []
        self.staged_class_names = None
        self.emitted = []
        self.emitted_durations = []
        self._objects = list(upstream_objects)
        self._blobs = dict(upstream_blobs or {})
        self._class_names = list(upstream_class_names)
        self._tracking_ids = list(upstream_tracking_ids) or [None] * len(self._objects)

    def objects(self):
        return list(self._objects)

    def class_names(self):
        return list(self._class_names)

    def blobs(self):
        return dict(self._blobs)

    def tracking_ids(self):
        return list(self._tracking_ids)

    def emit(self, payload, duration_ns=None, output=None):
        self.emitted.append(payload)
        self.emitted_durations.append(duration_ns)
        self.emitted_outputs.append(output)

    def set_class_names(self, names):
        self.staged_class_names = list(names)

    def _stage(self, record):
        self.staged.append(record)
        return len(self.staged) - 1

    def add_object(self, label, x, y, w, h, score):
        return self._stage(("object", label, x, y, w, h, score))

    def add_classification(self, label, score):
        return self._stage(("classification", label, score))

    def add_blob(self, header, payload):
        return self._stage(("blob", header, payload))

    def add_tracking(self, object_id):
        return self._stage(("tracking", object_id))

    def relate(self, src, dst):
        self.relations.append((src, dst))

    def track_upstream(self, index, object_id):
        self._stage(("upstream_tracking", index, object_id))

    def _of_kind(self, kind):
        return [record[1:] for record in self.staged if record[0] == kind]

    @property
    def staged_objects(self):
        return self._of_kind("object")

    @property
    def staged_blobs(self):
        return self._of_kind("blob")

    @property
    def staged_upstream_tracking(self):
        return self._of_kind("upstream_tracking")


#: What the shadowed `gi` raises, so a warning naming it means something wanted it.
NO_PYGOBJECT = "no pygobject"

#: The families that run hosted on g2g, so none of them may need GStreamer.
HOSTED_ELEMENT_MODULES = [
    "base_translate",
    "base_transcribe",
    "base_llm",
    "base_separate",
    "base_tts",
    "base_caption",
    "mariantranslate",
    "whispertranscribe",
    "whisperlive",
    "llm",
    "demucs",
    "sepformer",
    "whisperspeechtts",
    "caption_phi",
    "caption_qwen",
    "classifier",
    "yolo",
    "pose",
    "face",
    "tracker",
    "anonymize",
    "football_overlay",
    "baseball_overlay",
    "overlay_counter",
    "clap",
    "vad",
    "kafkasink",
]


def test_hosted_elements_import_under_g2g_with_no_pygobject(tmp_path):
    """A fresh interpreter, with `gi` shadowed by one that refuses to import.

    In-process the check cannot be honest: once any test here has initialised
    Gst, these imports succeed whether or not the element guards its Gst
    construction. Shadowing `gi` rather than just checking stderr also makes an
    element that reaches for GStreamer fail here, instead of quietly depending
    on a pygobject that happens to be installed.
    """
    shadow = tmp_path / "gi"
    shadow.mkdir()
    (shadow / "__init__.py").write_text(f'raise ImportError("{NO_PYGOBJECT}")\n')

    result = subprocess.run(
        [sys.executable, "-c", "import " + ", ".join(HOSTED_ELEMENT_MODULES)],
        env={
            **os.environ,
            "MP_BACKEND": "g2g",
            "PYTHONPATH": os.pathsep.join([str(tmp_path), str(PLUGIN_DIR)]),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    # The shadow raises with this text, so it appears only if something imported gi.
    assert (
        NO_PYGOBJECT not in result.stderr
    ), "an element reached for GStreamer at import time"


def test_every_element_module_imports_under_g2g_without_gst_init():
    """The same check over the whole plugin directory, so a new element cannot
    quietly reintroduce the crash.

    Kept apart from the payload check above because a module here may warn about
    a dependency it cannot find, which is not what this is looking for. A pad
    template built without `Gst.init` segfaults, so an element that forgets the
    backend guard takes down whatever process imports it.
    """
    pytest.importorskip("gi", reason="the element modules import it at module scope")

    modules = sorted(p.stem for p in PLUGIN_DIR.glob("*.py"))

    result = subprocess.run(
        [sys.executable, "-c", "import " + ", ".join(modules)],
        env={**os.environ, "MP_BACKEND": "g2g", "PYTHONPATH": str(PLUGIN_DIR)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_backend_selected_is_g2g():
    assert backend.BACKEND == "g2g"
    assert FlowReturn.OK == 0
    assert FlowReturn.ERROR != FlowReturn.OK


def test_gobject_property_shim_decorator_and_attribute_forms():
    class Widget:
        @GObject.Property(type=str)
        def name(self):
            return getattr(self, "_n", "default")

        @name.setter
        def name(self, value):
            self._n = value.upper()

        size = GObject.Property(type=int, default=7, nick="Size", blurb="px")

    w = Widget()
    assert w.name == "default"  # getter default
    w.name = "yolo"
    assert w.name == "YOLO"  # setter ran
    assert w.size == 7  # attribute-form default
    w.size = 42
    assert w.size == 42


def test_a_property_set_from_a_pipeline_line_arrives_as_its_declared_type():
    """The host cannot know a hosted class's property types, so it forwards the
    text a pipeline line carries and the declaration here converts it."""

    class Widget:
        count = GObject.Property(type=int, default=1)
        enabled = GObject.Property(type=bool, default=False)
        ratio = GObject.Property(type=float, default=0.0)
        name = GObject.Property(type=str, default="")

    w = Widget()
    w.count, w.enabled, w.ratio, w.name = "4", "TRUE", "0.5", "3"
    assert (w.count, w.enabled, w.ratio, w.name) == (4, True, 0.5, "3")

    w.enabled = "no"
    assert w.enabled is False
    w.count = 9  # already typed, set from Python
    assert w.count == 9

    with pytest.raises(ValueError, match="enabled"):
        w.enabled = "maybe"


def test_an_element_lists_the_properties_it_declares():
    """The host checks a pipeline against this, so a knob the element has must be
    in it and a name it does not have must not."""
    pytest.importorskip("gi", reason="the video transform needs it on the gst backend")
    pytest.importorskip("torch", reason="the depth engine imports it")
    from depth import DepthTransform

    declared = DepthTransform().g2g_properties()

    assert "colormap" in declared, "the element's own knob"
    assert "batch_size" in declared, "one it inherits from the shared tunables"
    assert "speaker" not in declared, "a detector has no speaker"
    assert len(declared) == len(set(declared)), "an overridden property listed twice"


def test_frameio_read_write_round_trip():
    width, height = 4, 3
    buf = bytearray(width * height * 3)  # RGB, writable buffer-protocol object
    sink = StubMetaSink()
    frameio.bind(sink, "RGB")

    frame, num_sources, fmt = frameio.read_frames(buf, None, width, height)
    assert num_sources == 1 and fmt == "RGB"
    assert frame.shape == (height, width, 3)

    frameio.write_frame(buf, np.full((height, width, 3), 200, dtype=np.uint8))
    assert all(b == 200 for b in buf), "write_frame must update the buffer in place"

    frameio.append_blob(buf, "tag", b"\x01\x02")
    assert sink.staged_blobs == [("tag", b"\x01\x02")]


def test_analytics_maps_onto_flat_sink():
    sink = StubMetaSink()
    analytics.bind(sink)

    meta = analytics.add_relation_meta(buf=None)
    assert meta is not None
    assert analytics.get_relation_meta(None) is meta

    # String labels intern to stable u32 ids (quark), reused across calls.
    analytics.add_object(meta, "person", 1, 2, 3, 4, 0.9)
    analytics.add_object(meta, "person", 5, 6, 7, 8, 0.8)
    analytics.add_object(meta, "handbag", 0, 0, 1, 1, 0.5)

    assert analytics.relation_length(meta) == 3
    labels = [o[0] for o in sink.staged_objects]
    assert labels[0] == labels[1], "same string -> same id"
    assert labels[2] != labels[0], "different string -> different id"
    assert sink.staged_objects[0][5] == 0.9


def test_class_names_are_published_so_a_consumer_can_name_a_label():
    sink = StubMetaSink()
    analytics.bind(sink)
    meta = analytics.add_relation_meta(buf=None)

    person = analytics.add_object(meta, "person", 1, 2, 3, 4, 0.9)
    handbag = analytics.add_object(meta, "handbag", 0, 0, 1, 1, 0.5)

    # The table is indexed by the label id staged on the detection, so a
    # consumer holding only the id can look the name up.
    names = sink.staged_class_names
    assert names is not None, "the sink was never sent a name table"
    assert names[sink.staged[person][1]] == "person"
    assert names[sink.staged[handbag][1]] == "handbag"


def test_a_torchvision_detection_is_named_from_the_model_categories():
    pytest.importorskip("torch")
    from engine.pytorch_engine import PyTorchEngine
    from objectdetector import ObjectDetector

    detector = ObjectDetector()
    detector.mgr.engine = PyTorchEngine()
    detector.engine.categories = ["__background__", "person", "bicycle"]
    sink = StubMetaSink()
    analytics.bind(sink)

    detector.do_decode(None, {"boxes": [[1, 2, 4, 6]], "labels": [1], "scores": [0.9]})

    (detection,) = sink.staged_objects
    assert sink.staged_class_names[detection[0]] == "stream_0_person"


def test_class_names_are_resent_for_each_frames_sink():
    """Each frame gets a fresh sink, so a name interned on an earlier frame has
    to be published again rather than assumed already known."""
    first = StubMetaSink()
    analytics.bind(first)
    staged = analytics.add_object(
        analytics.add_relation_meta(None), "person", 1, 2, 3, 4, 0.9
    )
    assert first.staged_class_names[first.staged[staged][1]] == "person"

    second = StubMetaSink()
    analytics.bind(second)
    staged = analytics.add_object(
        analytics.add_relation_meta(None), "person", 5, 6, 7, 8, 0.8
    )
    assert second.staged_class_names is not None, "the second sink was sent no table"
    assert second.staged_class_names[second.staged[staged][1]] == "person"


def test_own_labels_extend_the_upstream_name_table():
    upstream = ["player", "referee", "ball"]
    sink = StubMetaSink(upstream_class_names=upstream)
    analytics.bind(sink)
    meta = analytics.add_relation_meta(bytearray(4))
    tracked = analytics.add_object(meta, "player_id_0", 0, 0, 10, 10, 1.0)
    same_class = analytics.add_object(meta, "ball", 0, 0, 10, 10, 1.0)

    names = sink.staged_class_names
    assert names[:3] == upstream, "upstream's names keep their ids"
    assert sink.staged[same_class][1] == 2, "an upstream name keeps upstream's id"
    assert sink.staged[tracked][1] >= 3, "a new name takes an id after upstream's"
    assert names[sink.staged[tracked][1]] == "player_id_0"


def test_tracking_relates_to_its_detection():
    sink = StubMetaSink()
    analytics.bind(sink)
    meta = analytics.add_relation_meta(buf=None)

    od = analytics.add_object(meta, "person", 1, 2, 3, 4, 0.9)
    track = analytics.add_tracking(meta, 77)
    assert analytics.relate(meta, od, track) is True

    # Handles are the sink's own staging indices, so the relation names the
    # detection and the tracking record that were actually staged.
    assert sink.staged[od][0] == "object"
    assert sink.staged[track] == ("tracking", 77)
    assert sink.relations == [(od, track)]


#: One detection in the dict shape the analytics interface states, as the host
#: hands the frame's incoming objects over.
UPSTREAM_OBJECTS = [{"label": "person", "x": 1, "y": 2, "w": 3, "h": 4, "score": 0.9}]

#: One incoming blob, keyed the way the host spells `GST-ALERT:` back.
UPSTREAM_BLOBS = {"alert": b'[{"detection": {"label": "person"}}]'}


def test_the_frames_upstream_detections_read_back():
    sink = StubMetaSink(upstream_objects=UPSTREAM_OBJECTS)
    analytics.bind(sink)

    meta = analytics.get_relation_meta(None)
    assert meta is not None, "the frame arrived carrying detections"
    assert analytics.read_objects(meta) == [
        {**UPSTREAM_OBJECTS[0], "stream": None, "track_id": None, "ref": 0}
    ]


def test_a_frame_with_no_metadata_at_all_has_no_relation_meta():
    analytics.bind(StubMetaSink())
    assert analytics.get_relation_meta(None) is None


def test_the_frames_upstream_blobs_read_back():
    frameio.bind(StubMetaSink(upstream_blobs=UPSTREAM_BLOBS), "RGB")
    assert frameio.read_blobs(bytearray(4)) == UPSTREAM_BLOBS


def only_on_ran(only_on, sink):
    """Whether a gated element ran, driven exactly as the host drives it."""

    class CountFrames(VideoTransform):
        def process_frames(self, frames, num_sources, fmt, target):
            processed.append(fmt)

    processed = []
    elem = CountFrames()
    elem.engine_name = None  # nothing to load: the gate is what is under test
    elem.only_on = only_on
    width, height = 4, 3
    elem.g2g_process(bytearray(width * height * 3), width, height, "RGB", sink)
    return bool(processed)


def test_only_on_a_blob_skips_the_frame_that_does_not_carry_it():
    assert only_on_ran("alert", StubMetaSink()) is False
    assert only_on_ran("alert", StubMetaSink(upstream_blobs=UPSTREAM_BLOBS)) is True


def test_only_on_detections_keys_on_the_upstream_objects():
    assert only_on_ran("detections", StubMetaSink()) is False
    assert (
        only_on_ran("detections", StubMetaSink(upstream_objects=UPSTREAM_OBJECTS))
        is True
    )


def test_video_transform_g2g_process_end_to_end():
    """A VideoTransform subclass inverts the frame and stages one detection,
    driven exactly as the host drives it: instance.g2g_process(buf, w, h, fmt, sink)."""

    class InvertAndDetect(VideoTransform):
        def process_frames(self, frames, num_sources, fmt, target):
            assert num_sources == 1
            inverted = 255 - frames
            frameio.write_frame(target, inverted)
            meta = analytics.add_relation_meta(target)
            analytics.add_object(meta, "person", 0, 0, 10, 10, 0.99)

    width, height = 8, 8
    buf = bytearray([10] * (width * height * 3))
    sink = StubMetaSink()

    elem = InvertAndDetect()
    # EngineManager defaults to the pytorch engine, so the first frame would try
    # to load a model. This test covers the frame plumbing, not inference.
    elem.engine_name = None
    ret = elem.g2g_process(buf, width, height, "RGB", sink)

    assert ret is None
    assert all(b == 245 for b in buf), "frame inverted in place (255 - 10)"
    assert len(sink.staged_objects) == 1
    assert sink.staged_objects[0][5] == 0.99
    assert elem.width == width and elem.height == height


def test_tracker_gives_a_box_seen_twice_one_id():
    pytest.importorskip("scipy")
    from tracker import TrackerTransform

    tracker = TrackerTransform()
    tracker.min_hits = 1
    width, height = 64, 48
    box = {"label": "player", "x": 10, "y": 10, "w": 8, "h": 16, "score": 0.9}
    sinks = [StubMetaSink(upstream_objects=[box]) for _ in range(2)]
    for sink in sinks:
        tracker.g2g_process(bytearray(width * height * 4), width, height, "RGBA", sink)

    tagged = [sink.staged_upstream_tracking for sink in sinks]
    assert tagged[0] == tagged[1]
    assert [index for index, _id in tagged[1]] == [0]
    assert sinks[1].staged_objects == []


def test_anonymize_blacks_out_only_the_detected_region():
    from anonymize import Anonymize

    width, height = 64, 48
    original = np.full((height, width, 4), 200, np.uint8)
    original[..., 3] = 255
    frame = bytearray(original.tobytes())
    person = {"label": "person", "x": 16, "y": 12, "w": 20, "h": 18, "score": 1.0}
    anonymizer = Anonymize()
    anonymizer.mode = "blackout"
    anonymizer.box_expansion = 0.0

    anonymizer.g2g_process(
        frame, width, height, "RGBA", StubMetaSink(upstream_objects=[person])
    )

    pixels = np.frombuffer(frame, np.uint8).reshape(height, width, 4)
    assert not pixels[12:30, 16:36, :3].any()
    assert np.array_equal(pixels[:12], original[:12])
    assert (pixels[..., 3] == 255).all()


def test_football_overlay_draws_a_tracked_player_in_place():
    pytest.importorskip("cv2")
    from football_overlay import FootballOverlay

    overlay = FootballOverlay()
    overlay.team_colors = False
    width, height = 1280, 720
    frame = bytearray(width * height * 4)
    player = {
        "label": "player_id_3",
        "x": 600,
        "y": 300,
        "w": 40,
        "h": 120,
        "score": 1.0,
    }
    overlay.g2g_process(
        frame, width, height, "RGBA", StubMetaSink(upstream_objects=[player])
    )

    pixels = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 4)
    assert pixels[405:435, 580:660].any(), "the ellipse at the player's feet was drawn"
    assert not pixels[550:700, 900:1200].any(), "nothing drawn on the empty pitch"


BALL_TRAIL_FRAMES = 6
BALL_START_X = 300
BALL_STEP_PIXELS = 4
BALL_Y = 200
BALL_SIZE = 8


def ball_path_pixels(ball_trail):
    from football_overlay import FootballOverlay

    overlay = FootballOverlay()
    overlay.team_colors = False
    overlay.trails = False
    overlay.show_ball = False
    overlay.ball_trail = ball_trail
    width, height = 1280, 720
    player = {"label": "player_id_3", "x": 900, "y": 400, "w": 40, "h": 120}
    for step in range(BALL_TRAIL_FRAMES):
        frame = bytearray(width * height * 4)
        ball = {
            "label": "ball_id_7",
            "x": BALL_START_X + step * BALL_STEP_PIXELS,
            "y": BALL_Y,
            "w": BALL_SIZE,
            "h": BALL_SIZE,
        }
        upstream = [{**player, "score": 1.0}, {**ball, "score": 1.0}]
        overlay.g2g_process(
            frame, width, height, "RGBA", StubMetaSink(upstream_objects=upstream)
        )
    pixels = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 4)
    centre_y = BALL_Y + BALL_SIZE // 2
    return pixels[centre_y - 3 : centre_y + 4, BALL_START_X : BALL_START_X + 30]


def test_football_overlay_draws_the_ball_trail_without_the_player_trails():
    pytest.importorskip("cv2")
    assert ball_path_pixels(ball_trail=True).any(), "the ball trail was drawn"
    assert not ball_path_pixels(ball_trail=False).any(), "no ball trail when off"


BATTER_BOX = {"x": 500, "y": 200, "w": 80, "h": 300}
BAT_BOX = {"x": 540, "y": 150, "w": 30, "h": 120}
FIELDER_BOX = {"x": 1000, "y": 400, "w": 40, "h": 120}
BASEBALL_FRAMES = 3


def baseball_overlay_pixels(show_equipment):
    from baseball_overlay import BaseballOverlay, ROLE_RGBA, BATTER_ROLE, PLAYER_ROLE

    overlay = BaseballOverlay()
    overlay.show_labels = False
    overlay.show_equipment = show_equipment
    width, height = 1280, 720
    upstream = [
        {"label": "person_id_1", "score": 1.0, **BATTER_BOX},
        {"label": "person_id_2", "score": 1.0, **FIELDER_BOX},
        {"label": "baseball bat", "score": 0.9, **BAT_BOX},
    ]
    for _ in range(BASEBALL_FRAMES):
        frame = bytearray(width * height * 4)
        overlay.g2g_process(
            frame, width, height, "RGBA", StubMetaSink(upstream_objects=upstream)
        )
    pixels = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 4)
    return pixels, ROLE_RGBA[BATTER_ROLE][:3], ROLE_RGBA[PLAYER_ROLE][:3]


# the ellipse at the feet is this fraction of the box width tall
ELLIPSE_HEIGHT_RATIO = 0.35


def ellipse_colours(pixels, box):
    bottom = box["y"] + box["h"]
    centre = box["x"] + box["w"] // 2
    reach = int(ELLIPSE_HEIGHT_RATIO * box["w"]) + 3
    region = pixels[bottom - reach : bottom + reach, centre - 10 : centre + 10, :3]
    return {tuple(colour) for colour in region.reshape(-1, 3) if colour.any()}


def test_baseball_overlay_colours_the_person_holding_the_bat_as_the_batter():
    pytest.importorskip("cv2")
    pixels, batter_rgb, player_rgb = baseball_overlay_pixels(show_equipment=True)
    assert batter_rgb in ellipse_colours(pixels, BATTER_BOX)
    assert player_rgb in ellipse_colours(pixels, FIELDER_BOX)
    bat_edge = pixels[
        BAT_BOX["y"] : BAT_BOX["y"] + 1, BAT_BOX["x"] : BAT_BOX["x"] + BAT_BOX["w"]
    ]
    assert bat_edge.any(), "the bat outline was drawn"


def test_baseball_overlay_keeps_the_batter_role_from_a_tracking_relation():
    pytest.importorskip("cv2")
    from baseball_overlay import BaseballOverlay, ROLE_RGBA, BATTER_ROLE

    overlay = BaseballOverlay()
    overlay.show_labels = False
    width, height = 1280, 720
    batter = {"label": "person", "score": 0.9, **BATTER_BOX}
    bat = {"label": "baseball bat", "score": 0.9, **BAT_BOX}
    for _ in range(BASEBALL_FRAMES):
        overlay.g2g_process(
            bytearray(width * height * 4),
            width,
            height,
            "RGBA",
            StubMetaSink(
                upstream_objects=[batter, bat], upstream_tracking_ids=[1, None]
            ),
        )
    frame = bytearray(width * height * 4)
    overlay.g2g_process(
        frame,
        width,
        height,
        "RGBA",
        StubMetaSink(upstream_objects=[batter], upstream_tracking_ids=[1]),
    )
    pixels = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 4)
    assert ROLE_RGBA[BATTER_ROLE][:3] in ellipse_colours(pixels, BATTER_BOX)


def test_baseball_overlay_drops_a_box_nested_in_a_stronger_one():
    from baseball_overlay import drop_nested

    whole_bat = {"box": (100, 100, 300, 160), "confidence": 0.8}
    barrel = {"box": (200, 110, 300, 150), "confidence": 0.5}
    other_bat = {"box": (600, 100, 800, 160), "confidence": 0.4}
    assert drop_nested([barrel, whole_bat, other_bat]) == [whole_bat, other_bat]


def test_baseball_overlay_leaves_the_bat_unmarked_when_equipment_is_off():
    pytest.importorskip("cv2")
    pixels, _, _ = baseball_overlay_pixels(show_equipment=False)
    bat_top = BAT_BOX["y"] - 2
    assert not pixels[
        bat_top : bat_top + 1, BAT_BOX["x"] : BAT_BOX["x"] + BAT_BOX["w"]
    ].any()


COCO_KEYPOINT_COUNT = 17
LEFT_SHOULDER = 5
LEFT_ELBOW = 7


def upper_arm_pose(box, shoulder, elbow):
    keypoints = [[0.0, 0.0]] * COCO_KEYPOINT_COUNT
    confidences = [0.0] * COCO_KEYPOINT_COUNT
    for index, point in ((LEFT_SHOULDER, shoulder), (LEFT_ELBOW, elbow)):
        keypoints[index] = list(point)
        confidences[index] = 0.9
    corners = [box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]]
    return {"box": corners, "keypoints": keypoints, "kp_conf": confidences}


def test_baseball_overlay_draws_the_skeleton_of_the_batter_only():
    pytest.importorskip("cv2")
    pytest.importorskip("ultralytics")
    import json

    from baseball_overlay import BaseballOverlay, ROLE_RGBA, BATTER_ROLE

    overlay = BaseballOverlay()
    overlay.show_labels = False
    overlay.show_equipment = False
    width, height = 1280, 720
    upstream = [
        {"label": "person_id_1", "score": 1.0, **BATTER_BOX},
        {"label": "person_id_2", "score": 1.0, **FIELDER_BOX},
        {"label": "baseball bat", "score": 0.9, **BAT_BOX},
    ]
    poses = [
        upper_arm_pose(BATTER_BOX, (520, 260), (520, 330)),
        upper_arm_pose(FIELDER_BOX, (1020, 420), (1020, 480)),
    ]
    frame = bytearray(width * height * 4)
    overlay.g2g_process(
        frame,
        width,
        height,
        "RGBA",
        StubMetaSink(
            upstream_objects=upstream,
            upstream_blobs={"pose": json.dumps(poses).encode()},
        ),
    )
    pixels = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 4)
    assert tuple(pixels[295, 520, :3]) == ROLE_RGBA[BATTER_ROLE][:3]
    assert not pixels[440:470, 1015:1025].any(), "the fielder has no skeleton"


def test_aggregator_drives_the_same_hook_a_transform_fills_in():
    """The N-source case spells the same on both backends: one `process_frames`
    taking (H, W, C) for a single source and (N, H, W, C) for several."""
    seen = []

    class Batching(BaseAggregator):
        def process_frames(self, frames, num_sources, fmt, target):
            seen.append((frames.shape, num_sources, fmt))

    width, height = 4, 3
    elem = Batching()
    elem.engine_name = None

    buffers = [bytearray([n] * (width * height * 3)) for n in (1, 2)]
    elem.g2g_process_batch(buffers, width, height, "RGB", StubMetaSink())
    elem.g2g_process_batch(buffers[:1], width, height, "RGB", StubMetaSink())

    assert seen == [
        ((2, height, width, 3), 2, "RGB"),
        ((height, width, 3), 1, "RGB"),
    ]


class ScriptedDetectionEngine:
    categories = ["__background__", "person"]
    model = "model"
    track = False

    def __init__(self):
        self.batches = []

    def do_forward(self, frames):
        self.batches.append(frames.shape)
        return [
            {"boxes": [[1, 1, 3, 3]], "labels": [1], "scores": [0.9]} for _ in frames
        ]


def test_a_detector_given_a_batch_stages_each_streams_detections_under_its_prefix():
    from objectdetector import ObjectDetector

    detector = ObjectDetector()
    engine = ScriptedDetectionEngine()
    detector.mgr.engine = engine
    sink = StubMetaSink()
    width, height = 4, 3
    buffers = [bytearray(width * height * 4) for _ in range(2)]

    detector.g2g_process_batch(buffers, width, height, "RGBA", sink)

    assert engine.batches == [(2, height, width, 3)]
    labels = [sink.staged_class_names[label] for label, *_ in sink.staged_objects]
    assert sorted(labels) == ["stream_0_person", "stream_1_person"]


def test_g2g_caption_stages_the_caption_on_the_frame():
    """The caption family runs on the shared per-frame seam, so it works with no
    text pad and no GStreamer: the caption is staged as a classification."""
    from base_caption import BaseCaption

    class FakeCaption(BaseCaption):
        def forward(self, frames):
            return "a cat on a mat"

    leaf = FakeCaption()
    leaf.mgr.engine_name = None  # the fake above is the model
    sink = StubMetaSink()
    width, height = 4, 3

    leaf.g2g_process(bytearray(width * height * 4), width, height, "RGBA", sink)

    captions = [record for record in sink.staged if record[0] == "classification"]
    assert len(captions) == 1, "the caption was not staged"
    assert sink.staged_class_names[captions[0][1]] == "a cat on a mat"


def caption_into(sink):
    from base_caption import BaseCaption

    class FakeCaption(BaseCaption):
        def forward(self, frames):
            return "a cat on a mat"

    leaf = FakeCaption()
    leaf.mgr.engine_name = None
    width, height = 4, 3
    leaf.g2g_process(bytearray(width * height * 4), width, height, "RGBA", sink)


def test_g2g_caption_goes_out_text_src_when_it_is_linked():
    sink = StubMetaSink(text_output=True)

    caption_into(sink)

    assert sink.emitted == [b"a cat on a mat"]
    assert sink.emitted_outputs == ["text_src"]


def test_g2g_caption_emits_nothing_without_a_text_src_link():
    sink = StubMetaSink(text_output=False)

    caption_into(sink)

    assert sink.emitted == []


def test_a_packed_frame_is_reduced_to_rgb_in_channel_order():
    from backend.g2g.frameio import as_rgb

    pixel = np.array([[[10, 20, 30, 40]]], dtype=np.uint8)

    assert as_rgb(pixel, "RGBA").tolist() == [[[10, 20, 30]]]
    assert np.shares_memory(as_rgb(pixel, "RGBA"), pixel), "a leading RGB is a view"
    assert np.shares_memory(as_rgb(pixel, "ARGB"), pixel)
    assert as_rgb(pixel, "BGRA").tolist() == [[[30, 20, 10]]]
    assert as_rgb(pixel, "ARGB").tolist() == [[[20, 30, 40]]]
    assert as_rgb(pixel, "ABGR").tolist() == [[[40, 30, 20]]]

    rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
    assert as_rgb(rgb, "RGB") is rgb, "an RGB frame is handed over untouched"
    assert as_rgb(rgb, "BGR").tolist() == [[[30, 20, 10]]]


#: BT.601 limited range as a matrix over (Y - 16, Cb - 128, Cr - 128).
BT601_LIMITED_RANGE_TO_RGB = np.array(
    [[1.164, 0.0, 1.596], [1.164, -0.392, -0.813], [1.164, 2.017, 0.0]]
)
BT601_LIMITED_RANGE_OFFSETS = np.array([16, 128, 128])
#: Levels two independent rounding conversions may differ by.
CONVERSION_TOLERANCE = 2
LEGAL_LUMA_LEVELS = (16, 236)
LEGAL_CHROMA_LEVELS = (16, 241)
CHROMA_SUBSAMPLING = 2
MAXIMUM_LEVEL = 255
BT601_LIMITED_RANGE = (0.299, 0.114, False)
BT709_LIMITED_RANGE = (0.2126, 0.0722, False)
BT709_RED_SAMPLE = (63, 102, 240)
PURE_RED = (MAXIMUM_LEVEL, 0, 0)
RED_SAMPLE_TOLERANCE = 1
BT601_RED_CEILING = 240


def synthetic_nv12(height, width):
    generator = np.random.default_rng(0)
    luma = generator.integers(*LEGAL_LUMA_LEVELS, (height, width), dtype=np.uint8)
    chroma_height = -(-height // CHROMA_SUBSAMPLING)
    chroma_width = -(-width // CHROMA_SUBSAMPLING)
    chroma_shape = (chroma_height, chroma_width, CHROMA_SUBSAMPLING)
    chroma = generator.integers(*LEGAL_CHROMA_LEVELS, chroma_shape, dtype=np.uint8)
    return luma, chroma


def reference_rgb(luma, chroma):
    height, width = luma.shape
    full_chroma = chroma.repeat(CHROMA_SUBSAMPLING, axis=0).repeat(
        CHROMA_SUBSAMPLING, axis=1
    )[:height, :width]
    yuv = np.concatenate([luma[..., None], full_chroma], axis=-1).astype(float)
    rgb = (yuv - BT601_LIMITED_RANGE_OFFSETS) @ BT601_LIMITED_RANGE_TO_RGB.T
    return np.clip(np.round(rgb), 0, MAXIMUM_LEVEL).astype(np.uint8)


def largest_difference(first, second):
    return np.abs(first.astype(int) - second.astype(int)).max()


def test_nv12_converts_to_rgb_like_the_bt601_formula():
    torch = pytest.importorskip("torch")
    from backend.g2g.cuda_frames import nv12_to_rgb

    luma, chroma = synthetic_nv12(7, 9)
    rgb = nv12_to_rgb(
        torch.from_numpy(luma), torch.from_numpy(chroma), BT601_LIMITED_RANGE
    ).numpy()

    assert rgb.shape == (7, 9, 3)
    assert largest_difference(rgb, reference_rgb(luma, chroma)) <= CONVERSION_TOLERANCE


def test_nv12_converts_to_rgb_like_opencv():
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    from backend.g2g.cuda_frames import nv12_to_rgb

    height, width = 16, 24
    luma, chroma = synthetic_nv12(height, width)
    rgb = nv12_to_rgb(
        torch.from_numpy(luma), torch.from_numpy(chroma), BT601_LIMITED_RANGE
    ).numpy()
    nv12 = np.concatenate([luma, chroma.reshape(height // CHROMA_SUBSAMPLING, width)])

    expected = cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)
    assert largest_difference(rgb, expected) <= CONVERSION_TOLERANCE


def nv12_sample_to_rgb(sample, yuv_conversion):
    import torch
    from backend.g2g.cuda_frames import nv12_to_rgb

    luma_level, blue_level, red_level = sample
    luma = torch.full((CHROMA_SUBSAMPLING, CHROMA_SUBSAMPLING), luma_level)
    chroma = torch.tensor([[[blue_level, red_level]]])
    return nv12_to_rgb(
        luma.to(torch.uint8), chroma.to(torch.uint8), yuv_conversion
    ).numpy()[0, 0]


def test_bt709_converts_its_red_sample_to_pure_red():
    pytest.importorskip("torch")

    rgb = nv12_sample_to_rgb(BT709_RED_SAMPLE, BT709_LIMITED_RANGE)

    assert largest_difference(rgb, np.array(PURE_RED)) <= RED_SAMPLE_TOLERANCE


def test_bt601_reads_the_bt709_red_sample_as_a_duller_red():
    pytest.importorskip("torch")

    red, _, _ = nv12_sample_to_rgb(BT709_RED_SAMPLE, BT601_LIMITED_RANGE)

    assert red < BT601_RED_CEILING


CHROMA_CHANNELS = 2
SMOOTH_IMAGE_LEVELS = (40, 200)


def smooth_rgb(height, width):
    low, high = SMOOTH_IMAGE_LEVELS
    rows = np.linspace(low, high, height)[:, None]
    columns = np.linspace(low, high, width)[None, :]
    red = np.broadcast_to(rows, (height, width))
    green = np.broadcast_to(columns, (height, width))
    blue = (red + green) / 2
    return np.round(np.stack((red, green, blue), axis=-1)).astype(np.uint8)


BT709_ROUND_TRIP_TOLERANCE = 3


@pytest.mark.parametrize(
    "yuv_conversion, tolerance",
    [
        (BT601_LIMITED_RANGE, CONVERSION_TOLERANCE),
        (BT709_LIMITED_RANGE, BT709_ROUND_TRIP_TOLERANCE),
    ],
)
def test_rgb_converts_to_nv12_and_back_within_rounding(yuv_conversion, tolerance):
    torch = pytest.importorskip("torch")
    from backend.g2g.cuda_frames import nv12_to_rgb, rgb_to_nv12

    rgb = smooth_rgb(128, 192)
    luma, chroma = rgb_to_nv12(torch.from_numpy(rgb), yuv_conversion)
    round_trip = nv12_to_rgb(luma, chroma, yuv_conversion).numpy()

    assert luma.shape == (128, 192) and chroma.shape == (64, 96, CHROMA_CHANNELS)
    assert largest_difference(round_trip, rgb) <= tolerance


def test_rgb_converts_to_nv12_with_chroma_right_after_luma():
    torch = pytest.importorskip("torch")
    from backend.g2g.cuda_frames import rgb_to_nv12

    luma, chroma = rgb_to_nv12(
        torch.from_numpy(smooth_rgb(16, 24)), BT601_LIMITED_RANGE
    )

    assert luma.is_contiguous() and chroma.is_contiguous()
    assert chroma.data_ptr() == luma.data_ptr() + luma.numel()


def test_rgb_converts_to_nv12_like_opencv():
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    from backend.g2g.cuda_frames import rgb_to_nv12

    height, width = 16, 24
    rgb = smooth_rgb(height, width)
    luma, chroma = rgb_to_nv12(torch.from_numpy(rgb), BT601_LIMITED_RANGE)

    i420 = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
    chroma_height, chroma_width = (
        height // CHROMA_SUBSAMPLING,
        width // CHROMA_SUBSAMPLING,
    )
    chroma_plane_rows = chroma_height // CHROMA_SUBSAMPLING
    blue_plane = i420[height : height + chroma_plane_rows]
    red_plane = i420[height + chroma_plane_rows :]
    expected_chroma = np.stack(
        (
            blue_plane.reshape(chroma_height, chroma_width),
            red_plane.reshape(chroma_height, chroma_width),
        ),
        axis=-1,
    )
    assert largest_difference(luma.numpy(), i420[:height]) <= CONVERSION_TOLERANCE
    assert largest_difference(chroma.numpy(), expected_chroma) <= CONVERSION_TOLERANCE


def test_a_cuda_frame_reaches_process_frames_as_an_rgb_tensor_on_the_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from backend.g2g.cuda_frames import nv12_to_rgb

    class RecordFrame(VideoTransform):
        FRAME_CAPABILITIES = CUDA_INFERENCE_FRAME_CAPABILITIES

        def process_frames(self, frames, num_sources, fmt, target):
            self.received = frames, num_sources, fmt
            meta = analytics.add_relation_meta(target)
            analytics.add_object(meta, "person", 0, 0, 10, 10, 0.9)

    height, width = 6, 8
    luma, chroma = synthetic_nv12(height, width)
    element = RecordFrame()
    element.device = "cuda"
    sink = StubMetaSink()

    element.g2g_process_cuda(
        torch.from_numpy(luma).cuda(),
        torch.from_numpy(chroma).cuda(),
        width,
        height,
        BT601_LIMITED_RANGE,
        sink,
    )

    frames, num_sources, fmt = element.received
    assert frames.is_cuda and (num_sources, fmt) == (1, "RGB")
    expected = nv12_to_rgb(
        torch.from_numpy(luma), torch.from_numpy(chroma), BT601_LIMITED_RANGE
    )
    assert torch.equal(frames.cpu(), expected)
    assert len(sink.staged_objects) == 1


#: A decoder surface pads each row out to a pitch wider than the frame.
DECODER_ROW_PADDING = 64
BLACK_LUMA = 16
NEUTRAL_CHROMA = 128
PLAYER_FEET_REGION = (slice(405, 435), slice(580, 660))
HUD_REGION = (slice(0, 130), slice(0, 400))
EMPTY_PITCH_REGION = (slice(550, 700), slice(900, 1200))


def black_cuda_planes(height, width):
    import torch

    padded_luma = torch.full(
        (height, width + DECODER_ROW_PADDING), BLACK_LUMA, dtype=torch.uint8
    )
    chroma_shape = (
        height // CHROMA_SUBSAMPLING,
        width // CHROMA_SUBSAMPLING,
        CHROMA_CHANNELS,
    )
    chroma = torch.full(chroma_shape, NEUTRAL_CHROMA, dtype=torch.uint8)
    return padded_luma.cuda()[:, :width], chroma.cuda()


def test_anonymize_blacks_out_a_cuda_frame_and_returns_nv12():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from anonymize import Anonymize
    from backend.g2g.cuda_frames import nv12_to_rgb, rgb_to_nv12

    width, height = 96, 72
    original = torch.full((height, width, 3), 200, dtype=torch.uint8, device="cuda")
    luma, chroma = rgb_to_nv12(original, BT601_LIMITED_RANGE)
    person = {"label": "person", "x": 24, "y": 18, "w": 32, "h": 24, "score": 1.0}
    anonymizer = Anonymize()
    anonymizer.mode = "blackout"
    anonymizer.box_expansion = 0.0

    drawn_luma, drawn_chroma = anonymizer.g2g_process_cuda(
        luma,
        chroma,
        width,
        height,
        BT601_LIMITED_RANGE,
        StubMetaSink(upstream_objects=[person]),
    )

    pixels = nv12_to_rgb(drawn_luma, drawn_chroma, BT601_LIMITED_RANGE)
    assert drawn_luma.is_cuda and drawn_chroma.is_cuda
    assert pixels[22:38, 28:52].max() < 4
    assert (
        largest_difference(pixels[:12].cpu().numpy(), original[:12].cpu().numpy())
        <= BT709_ROUND_TRIP_TOLERANCE
    )


def football_overlay_on_a_cuda_frame():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    pytest.importorskip("cv2")
    from football_overlay import FootballOverlay

    overlay = FootballOverlay()
    overlay.team_colors = False
    width, height = 1280, 720
    player = {"label": "player_id_3", "x": 600, "y": 300, "w": 40, "h": 120}
    luma, chroma = black_cuda_planes(height, width)
    sink = StubMetaSink(upstream_objects=[{**player, "score": 1.0}])
    drawn_planes = overlay.g2g_process_cuda(
        luma, chroma, width, height, BT601_LIMITED_RANGE, sink
    )
    return overlay, (luma, chroma), drawn_planes


def test_football_overlay_draws_on_a_cuda_frame_and_hands_back_nv12_planes():
    from backend.g2g.cuda_frames import nv12_to_rgb

    _, (luma, chroma), drawn_planes = football_overlay_on_a_cuda_frame()

    drawn_luma, drawn_chroma = drawn_planes
    assert drawn_luma.is_cuda and drawn_luma.is_contiguous()
    assert drawn_chroma.is_cuda and drawn_chroma.is_contiguous()
    assert drawn_luma.shape == luma.shape and drawn_chroma.shape == chroma.shape
    assert drawn_luma.dtype == drawn_chroma.dtype == luma.dtype
    before = nv12_to_rgb(luma, chroma, BT601_LIMITED_RANGE).cpu().numpy()
    after = nv12_to_rgb(drawn_luma, drawn_chroma, BT601_LIMITED_RANGE).cpu().numpy()
    changed = np.any(before != after, axis=-1)
    assert changed[PLAYER_FEET_REGION].any(), "the ellipse at the player's feet"
    assert changed[HUD_REGION].any(), "the focal player's hud"
    assert not changed[EMPTY_PITCH_REGION].any(), "nothing drawn on the empty pitch"


def test_football_overlay_names_the_context_its_drawn_planes_live_in():
    from backend.cuda_driver import cuda_driver_library

    overlay, _, (drawn_luma, _) = football_overlay_on_a_cuda_frame()

    torch_context = ctypes.c_void_p()
    cuda_driver_library().cuCtxGetCurrent(ctypes.byref(torch_context))
    assert overlay.cuda_device == drawn_luma.device.index
    assert isinstance(overlay.cuda_context, int) and overlay.cuda_context != 0
    assert overlay.cuda_context == torch_context.value


def test_tracker_on_a_cuda_frame_tracks_and_leaves_the_frame_alone():
    pytest.importorskip("scipy")
    pytest.importorskip("torch")
    from tracker import TrackerTransform

    tracker = TrackerTransform()
    tracker.min_hits = 1
    width, height = 64, 48
    box = {"label": "player", "x": 10, "y": 10, "w": 8, "h": 16, "score": 0.9}
    sinks = [StubMetaSink(upstream_objects=[box]) for _ in range(2)]
    # the tracker never reads the planes
    returned = [
        tracker.g2g_process_cuda(None, None, width, height, BT601_LIMITED_RANGE, sink)
        for sink in sinks
    ]

    assert returned == [None, None]
    assert [index for index, _id in sinks[1].staged_upstream_tracking] == [0]
    assert sinks[1].staged_objects == []


SAMPLE_METADATA = PLUGIN_DIR.parent.parent / "data" / "sample_metadata.json"
KAFKA_MESSAGE_SCHEMA = PLUGIN_DIR.parent.parent / "data" / "mp_object_detector.json"
SAMPLE_METADATA_WIDTH, SAMPLE_METADATA_HEIGHT = 1280, 720
COUNTING_LINE_REGION = (slice(358, 363), slice(1000, 1280))
COUNTER_TEXT_REGION = (slice(30, 55), slice(0, 300))
EMPTY_CORNER_REGION = (slice(650, 720), slice(1000, 1280))


def overlay_counter_for_the_sample_metadata():
    pytest.importorskip("cairo")
    from overlay_counter import OverlayCounter

    counter = OverlayCounter()
    counter.meta_path = str(SAMPLE_METADATA)
    return counter


def test_overlay_counter_draws_the_counting_line_and_the_counts():
    counter = overlay_counter_for_the_sample_metadata()
    width, height = SAMPLE_METADATA_WIDTH, SAMPLE_METADATA_HEIGHT
    frame = bytearray(width * height * 4)
    counter.g2g_process(frame, width, height, "RGBA", StubMetaSink())

    pixels = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 4)
    assert pixels[COUNTING_LINE_REGION].any(), "the line across half the height"
    assert pixels[COUNTER_TEXT_REGION].any(), "the counter text"
    assert not pixels[EMPTY_CORNER_REGION].any(), "nothing drawn in the empty corner"


def test_overlay_counter_draws_on_a_cuda_frame():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from backend.g2g.cuda_frames import nv12_to_rgb

    counter = overlay_counter_for_the_sample_metadata()
    width, height = SAMPLE_METADATA_WIDTH, SAMPLE_METADATA_HEIGHT
    luma, chroma = black_cuda_planes(height, width)
    drawn_luma, drawn_chroma = counter.g2g_process_cuda(
        luma, chroma, width, height, BT601_LIMITED_RANGE, StubMetaSink()
    )

    before = nv12_to_rgb(luma, chroma, BT601_LIMITED_RANGE).cpu().numpy()
    after = nv12_to_rgb(drawn_luma, drawn_chroma, BT601_LIMITED_RANGE).cpu().numpy()
    changed = np.any(before != after, axis=-1)
    assert changed[COUNTING_LINE_REGION].any(), "the line across half the height"
    assert changed[COUNTER_TEXT_REGION].any(), "the counter text"
    assert not changed[EMPTY_CORNER_REGION].any(), "nothing drawn in the empty corner"


def test_a_frame_read_from_cuda_memory_refuses_a_write():
    from backend.g2g.frameio import DEVICE_FRAME

    with pytest.raises(ValueError, match="read-only"):
        frameio.write_frame(DEVICE_FRAME, np.zeros((2, 2, 3), dtype=np.uint8))


class RecordingProducer:
    def __init__(self, config):
        self.config = config
        self.produced = []
        self.flush_count = 0

    def produce(self, topic, value, callback=None):
        self.produced.append((topic, value))

    def flush(self):
        self.flush_count += 1


KAFKA_DETECTIONS = [
    {"label": "person", "x": 1, "y": 2, "w": 3, "h": 4, "score": 0.9},
    {"label": "car", "x": 10, "y": 20, "w": 30, "h": 40, "score": 0.5},
]


def test_kafkasink_sends_the_upstream_detections_and_flushes_at_eos(monkeypatch):
    import types

    producers = []

    def recording_producer(config):
        producers.append(RecordingProducer(config))
        return producers[-1]

    kafka_client = types.ModuleType("confluent_kafka")
    kafka_client.Producer = recording_producer
    monkeypatch.setitem(sys.modules, "confluent_kafka", kafka_client)
    from kafkasink import KafkaSink

    sink = KafkaSink()
    sink.broker = "localhost:29092"
    sink.topic = "detections"
    width, height = 4, 4
    result = sink.g2g_process(
        bytearray(width * height * 4),
        width,
        height,
        "RGBA",
        StubMetaSink(upstream_objects=KAFKA_DETECTIONS),
    )

    assert result is None
    (producer,) = producers
    assert producer.config["bootstrap.servers"] == "localhost:29092"
    ((topic, value),) = producer.produced
    assert topic == "detections"
    assert json.loads(value)["detections"] == [
        {
            "label": "person",
            "score": 0.9,
            "box": {"x1": 1, "y1": 2, "x2": 4, "y2": 6},
        },
        {
            "label": "car",
            "score": 0.5,
            "box": {"x1": 10, "y1": 20, "x2": 40, "y2": 60},
        },
    ]
    message = json.loads(value)
    schema = json.loads(KAFKA_MESSAGE_SCHEMA.read_text(encoding="utf-8"))
    detection_schema = schema["properties"]["detections"]["items"]
    assert set(schema["required"]) <= message.keys()
    assert isinstance(message["timestamp"], str)
    assert all(
        set(detection_schema["required"]) <= d.keys() for d in message["detections"]
    )
    flushes_before_eos = producer.flush_count
    sink.g2g_eos()
    assert producer.flush_count == flushes_before_eos + 1


def test_two_elements_on_their_own_threads_keep_their_own_sinks():
    """The host runs one thread per element, so a binding made on one thread must
    not redirect what another thread stages."""
    import threading

    sinks = {}
    staged = {}
    start = threading.Barrier(2)
    bound = threading.Barrier(2)

    def stage(name, label):
        sinks[name] = StubMetaSink()
        start.wait()
        analytics.bind(sinks[name])
        frameio.bind(sinks[name], "RGB")
        # Both threads have bound before either stages anything, so a shared
        # binding would send both records to whichever bound last.
        bound.wait()
        analytics.add_object(analytics.add_relation_meta(None), label, 0, 0, 1, 1, 1.0)
        frameio.append_blob(bytearray(4), "tag", label.encode())
        staged[name] = sinks[name].staged

    threads = [
        threading.Thread(target=stage, args=(name, label))
        for name, label in (("first", "person"), ("second", "handbag"))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(staged["first"]) == 2, "the first element lost a record to the second"
    assert len(staged["second"]) == 2
    assert sinks["first"].staged_blobs == [("tag", b"person")]
    assert sinks["second"].staged_blobs == [("tag", b"handbag")]


class RecordingLogger:
    """Stands in for the element's logger, keeping what it was told."""

    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)

    def info(self, message):
        pass

    def error(self, message):
        pass


def gst():
    pytest.importorskip("gi", reason="the gst driver and the pad templates need it")
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


def gst_payload_driver():
    """The gst backend's driver half, to mix into a leaf.

    This process selected the g2g backend, so a leaf's element base is the g2g
    one. Adding the driver gives a single instance both backends' entry points,
    which is what the seam claims: the same element runs under either.
    """
    gst()
    from backend.gst.aggregator import PayloadDriver

    return PayloadDriver


CHUNKED_OUTPUTS = [b"one", b"two", b"three"]


def chunking_leaf():
    """A payload element that answers one input buffer with several outputs."""
    driver = gst_payload_driver()

    class ChunkingLeaf(BaseAggregator, driver):
        def process_payload(self, payload):
            return list(CHUNKED_OUTPUTS)

    leaf = ChunkingLeaf()
    leaf.engine_name = None
    leaf.logger = RecordingLogger()
    return leaf


def translate_leaf(translate_text):
    """A real `BaseTranslate` whose model is the given text -> text function.

    Importing the element needs GStreamer even under the g2g backend, since the
    family still declares its pad templates with Gst types.
    """
    driver = gst_payload_driver()
    from base_translate import BaseTranslate

    class FakeTranslate(BaseTranslate, driver):
        def do_translate_text(self, text):
            return translate_text(text)

    leaf = FakeTranslate()
    leaf.engine_name = None  # nothing to load: the fake above is the model
    return leaf


class StubSrcPad:
    """Stands in for the element's src pad, for the family that pushes straight
    out of it instead of through the aggregator."""

    def __init__(self, pushed):
        self._pushed = pushed
        self.push_count = 0

    def push(self, buf):
        self._pushed.append(buf)
        self.push_count += 1
        return 0


def drive_gst_payload(leaf, payload, pts=1000, duration=500):
    """Run the gst driver over one input buffer, returning what it sent.

    Both send routes are stubbed, the aggregator's `finish_buffer` and the src
    pad, so `leaf.srcpad.push_count` says which one the element took.
    """
    Gst = gst()
    from backend.gst.aggregator import BaseAggregator as GstBaseAggregator

    pushed = []
    leaf.finish_buffer = pushed.append
    leaf.srcpad = StubSrcPad(pushed)

    inbuf = Gst.Buffer.new_allocate(None, len(payload), None)
    inbuf.fill(0, payload)
    inbuf.pts = pts
    inbuf.duration = duration

    ret = GstBaseAggregator.do_process(leaf, inbuf)
    return ret, pushed


def buffer_bytes(buf):
    Gst = gst()
    success, map_info = buf.map(Gst.MapFlags.READ)
    assert success
    data = bytes(map_info.data)
    buf.unmap(map_info)
    return data


def test_g2g_payload_driver_emits_the_translated_bytes():
    leaf = translate_leaf(lambda text: "hello" if text == "hola" else "")
    sink = StubMetaSink()

    leaf.g2g_process_payload([bytearray(b"hola")], "text/x-raw,format=utf8", sink)

    assert sink.emitted == [b"hello"]
    assert sink.emitted_durations == [None], "text keeps the input buffer's timing"


SD_IMAGE_SIDE = 512


def stablediffusion_leaf():
    from types import SimpleNamespace

    Image = pytest.importorskip("PIL.Image")

    driver = gst_payload_driver()
    from stablediffusion import StableDiffusion

    def fake_pipeline(prompt):
        image = Image.new("RGB", (SD_IMAGE_SIDE, SD_IMAGE_SIDE), "red")
        return SimpleNamespace(images=[image])

    class FakeStableDiffusion(StableDiffusion, driver):
        def get_model(self):
            return fake_pipeline

    leaf = FakeStableDiffusion()
    leaf.engine_name = None
    return leaf


def test_g2g_stablediffusion_emits_one_rgba_image():
    leaf = stablediffusion_leaf()
    sink = StubMetaSink()

    leaf.g2g_process_payload([bytearray(b"a cat")], "text/x-raw,format=utf8", sink)

    assert [len(payload) for payload in sink.emitted] == [
        SD_IMAGE_SIDE * SD_IMAGE_SIDE * 4
    ]
    assert sink.emitted_durations == [None]


def test_g2g_payload_driver_emits_nothing_when_the_element_has_no_output():
    leaf = translate_leaf(lambda text: "")
    sink = StubMetaSink()

    leaf.g2g_process_payload([bytearray(b"hola")], "text/x-raw,format=utf8", sink)

    assert sink.emitted == [], "an empty result must not reach the host"


def test_g2g_payload_driver_emits_every_payload():
    leaf = chunking_leaf()
    sink = StubMetaSink()

    leaf.g2g_process_payload([bytearray(b"in")], "audio/x-raw", sink)

    assert sink.emitted == CHUNKED_OUTPUTS
    assert leaf.logger.warnings == []


def test_gst_payload_driver_pushes_the_translated_buffer():
    Gst = gst()
    leaf = translate_leaf(lambda text: "hello" if text == "hola" else "")

    ret, pushed = drive_gst_payload(leaf, b"hola")

    assert ret == Gst.FlowReturn.OK
    assert len(pushed) == 1
    assert buffer_bytes(pushed[0]) == b"hello"
    assert pushed[0].pts == 1000 and pushed[0].duration == 500


def test_gst_payload_driver_pushes_nothing_when_the_element_has_no_output():
    Gst = gst()
    leaf = translate_leaf(lambda text: "")

    ret, pushed = drive_gst_payload(leaf, b"hola")

    assert ret == Gst.FlowReturn.OK
    assert pushed == []


def test_gst_payload_driver_pushes_every_payload_as_its_own_buffer():
    leaf = chunking_leaf()

    _, pushed = drive_gst_payload(leaf, b"in")

    assert [buffer_bytes(buf) for buf in pushed] == CHUNKED_OUTPUTS
    assert leaf.logger.warnings == []


class StoppedSrcPad:
    def push(self, buf):
        return gst().FlowReturn.FLUSHING


# an llm answer can finish after the src pad starts flushing
def test_gst_payload_push_after_the_pipeline_stopped_is_not_an_error():
    Gst = gst()
    from backend.gst.aggregator import BaseAggregator as GstBaseAggregator

    leaf = llm_leaf()
    leaf.srcpad = StoppedSrcPad()
    inbuf = Gst.Buffer.new_allocate(None, 5, None)
    inbuf.fill(0, b"hello")

    assert GstBaseAggregator.do_process(leaf, inbuf) == Gst.FlowReturn.OK


VAD_CHUNK_SAMPLES = 2400  # 150 ms at 16 kHz, so two silent chunks end a clip


def stub_vad(monkeypatch):
    """Stand in for the optional VAD package the transcribe family builds in its
    constructor. This one calls any non-zero sample speech, which lets a test
    write silence and speech as buffer contents."""
    import types

    class SpeechIsNonZero:
        def chunk_samples(self):
            return VAD_CHUNK_SAMPLES

        def process_chunk(self, chunk):
            return 1.0 if any(chunk) else 0.0

    module = types.ModuleType("pysilero_vad")
    module.SileroVoiceActivityDetector = SpeechIsNonZero
    monkeypatch.setitem(sys.modules, "pysilero_vad", module)


def transcribe_leaf(monkeypatch, transcript):
    """A real `BaseTranscribe` with a scripted VAD and transcriber."""
    driver = gst_payload_driver()
    stub_vad(monkeypatch)

    from base_transcribe import BaseTranscribe

    class FakeTranscribe(BaseTranscribe, driver):
        def do_transcribe(self, audio_data, task):
            return transcript.split()

    leaf = FakeTranscribe()
    leaf.engine_name = None
    return leaf


def speech(chunks=1):
    return np.full(VAD_CHUNK_SAMPLES * chunks, 1000, dtype=np.int16).tobytes()


def silence(chunks=1):
    return np.zeros(VAD_CHUNK_SAMPLES * chunks, dtype=np.int16).tobytes()


def separate_leaf(sample_rate=4):
    """A real `BaseSeparate` whose model returns the audio it was given."""
    driver = gst_payload_driver()
    from base_separate import BaseSeparate

    class PassThroughSeparate(BaseSeparate, driver):
        SAMPLE_RATE = sample_rate

        def do_separate(self, audio_data):
            return audio_data

    leaf = PassThroughSeparate()
    leaf.engine_name = None
    leaf.streaming = True  # a one second chunk, so four samples at this rate
    return leaf


class FakeEngine:
    def do_generate(self, text, system_prompt=None):
        return "answered " + text


def llm_leaf():
    """A real `BaseLlm` whose engine is a canned generator."""
    driver = gst_payload_driver()
    from base_llm import BaseLlm

    class FakeLlm(BaseLlm, driver):
        @property
        def engine(self):
            return self._fake_engine

        def get_tokenizer(self):
            return "tokenizer"

        def get_model(self):
            return "model"

    leaf = FakeLlm()
    leaf._fake_engine = FakeEngine()
    leaf.engine_name = None
    return leaf


PAYLOAD_FAMILY_BASES = [
    ("base_translate", "BaseTranslate"),
    ("base_transcribe", "BaseTranscribe"),
    ("base_llm", "BaseLlm"),
    ("base_separate", "BaseSeparate"),
    ("base_tts", "BaseTts"),
]


def declared_properties(cls):
    """Every property the class declares or inherits, mapped to its owner."""
    return {
        name: klass.__name__
        for klass in reversed(cls.__mro__)
        for name, value in vars(klass).items()
        if isinstance(value, GObject.Property)
    }


@pytest.mark.parametrize("module_name,class_name", PAYLOAD_FAMILY_BASES)
def test_every_declared_property_reads_before_anything_sets_it(
    monkeypatch, module_name, class_name
):
    """A freshly built element has to answer every property it declares.

    Nothing applies a declared default: GObject keeps it in the pspec and never
    routes it through the setter, so the constructor is the only thing that can
    create the backing attribute a custom getter reads.
    """
    gst()
    stub_vad(monkeypatch)
    element = getattr(importlib.import_module(module_name), class_name)()

    unreadable = []
    for name, owner in sorted(declared_properties(type(element)).items()):
        try:
            getattr(element, name)
        except AttributeError as exception:
            unreadable.append(f"{owner}.{name}: {exception}")

    assert unreadable == []


def test_g2g_transcribe_emits_only_once_the_clip_ends(monkeypatch):
    leaf = transcribe_leaf(monkeypatch, "hello world")
    sink = StubMetaSink()
    caps = "audio/x-raw,format=S16LE,rate=16000,channels=1"

    leaf.g2g_process_payload([bytearray(speech())], caps, sink)
    assert sink.emitted == [], "speech is still being accumulated"

    leaf.g2g_process_payload([bytearray(silence(3))], caps, sink)
    assert sink.emitted == [b"hello world"], "the silence should end the clip"


def test_gst_transcribe_pushes_only_once_the_clip_ends(monkeypatch):
    leaf = transcribe_leaf(monkeypatch, "hello world")

    _, pushed = drive_gst_payload(leaf, speech())
    assert pushed == [], "speech is still being accumulated"

    _, pushed = drive_gst_payload(leaf, silence(3))
    assert [buffer_bytes(buf) for buf in pushed] == [b"hello world"]
    assert pushed[0].pts == 1000 and pushed[0].duration == 500


def test_gst_transcribe_pushes_nothing_for_a_buffer_below_one_vad_chunk(monkeypatch):
    leaf = transcribe_leaf(monkeypatch, "hello world")

    _, pushed = drive_gst_payload(leaf, np.zeros(8, dtype=np.int16).tobytes())

    assert pushed == []


def test_gst_payload_excludes_metadata_memories_appended_upstream():
    Gst = gst()
    from backend.gst.aggregator import BaseAggregator as GstBaseAggregator

    driver = gst_payload_driver()
    seen = []

    class RecordingLeaf(BaseAggregator, driver):
        def process_payload(self, payload):
            seen.append(payload)
            return []

    leaf = RecordingLeaf()
    leaf.engine_name = None
    audio = speech()
    inbuf = Gst.Buffer.new_allocate(None, len(audio), None)
    inbuf.fill(0, audio)
    meta = Gst.Buffer.new_allocate(None, 12, None)
    meta.fill(0, b"GST-VAD:" + b"\x00\x00\x80\x3f")
    inbuf.append_memory(meta.get_memory(0))

    GstBaseAggregator.do_process(leaf, inbuf)

    assert seen == [audio]


def test_gst_transcribe_accumulates_buffers_below_one_vad_chunk(monkeypatch):
    leaf = transcribe_leaf(monkeypatch, "hello world")
    third = len(speech()) // 3

    for start in range(0, len(speech()), third):
        _, pushed = drive_gst_payload(leaf, speech()[start : start + third])
        assert pushed == [], "speech is still being accumulated"

    _, pushed = drive_gst_payload(leaf, silence(3))
    assert [buffer_bytes(buf) for buf in pushed] == [b"hello world"]


def test_gst_separate_pushes_one_buffer_per_whole_chunk():
    leaf = separate_leaf()
    samples = np.arange(1, 11, dtype=np.int16)  # ten samples, so two chunks of four

    _, pushed = drive_gst_payload(leaf, samples.tobytes())

    assert [buffer_bytes(buf) for buf in pushed] == [
        samples[:4].tobytes(),
        samples[4:8].tobytes(),
    ]
    assert len(leaf.clip_buffer) == 2, "the remainder waits for the next buffer"


def test_g2g_separate_emits_one_buffer_per_whole_chunk():
    leaf = separate_leaf()
    leaf.logger = RecordingLogger()
    sink = StubMetaSink()
    samples = np.arange(1, 11, dtype=np.int16)

    leaf.g2g_process_payload([bytearray(samples.tobytes())], "audio/x-raw", sink)

    assert sink.emitted == [samples[:4].tobytes(), samples[4:8].tobytes()]
    assert leaf.logger.warnings == []


def test_g2g_llm_emits_the_generated_text():
    leaf = llm_leaf()
    sink = StubMetaSink()

    leaf.g2g_process_payload([bytearray(b"question")], "text/x-raw,format=utf8", sink)

    assert sink.emitted == [b"answered question"]


class FakeOllamaResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return json.dumps(self.body).encode("utf-8")


def test_g2g_llm_remote_emits_the_server_reply(monkeypatch):
    import urllib.request

    from llm_remote import LlmRemote

    requests = []

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return FakeOllamaResponse({"response": "<reply>"})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    leaf = LlmRemote()
    sink = StubMetaSink()

    leaf.g2g_process_payload([bytearray(b"question")], "text/x-raw,format=utf8", sink)

    assert sink.emitted == [b"<reply>"]
    assert [(request["prompt"], request["model"]) for request in requests] == [
        ("question", "llama3")
    ]
    assert leaf.engine is None


def test_g2g_coalescehistory_emits_the_last_messages_joined():
    from coalescehistory import CoalesceHistory

    leaf = CoalesceHistory()
    leaf.history_length = 2
    sink = StubMetaSink()

    for message in (b"one", b"two", b"three"):
        leaf.g2g_process_payload([bytearray(message)], "text/x-raw,format=utf8", sink)

    assert sink.emitted == [b"one\n\n", b"one\n\ntwo\n\n", b"two\n\nthree\n\n"]
    assert leaf.engine is None


def test_gst_llm_pushes_out_of_the_src_pad_rather_than_the_aggregator():
    Gst = gst()
    leaf = llm_leaf()

    ret, pushed = drive_gst_payload(leaf, b"question")

    assert ret == Gst.FlowReturn.OK
    assert [buffer_bytes(buf) for buf in pushed] == [b"answered question"]
    assert leaf.srcpad.push_count == 1, "this family has never used finish_buffer"
    assert pushed[0].pts == 1000 and pushed[0].duration == 500


TTS_SAMPLE_RATE = 22050
TTS_SAMPLES = 8000


def tts_leaf():
    """A real `BaseTts` whose voice is a ramp of the right sample count."""
    pytest.importorskip("soundfile", reason="BaseTts encodes the voice with it")
    driver = gst_payload_driver()
    from base_tts import BaseTts

    class FakeTts(BaseTts, driver):
        def do_load_model(self):
            pass

        def do_generate_speech(self, transcript):
            return np.linspace(-0.5, 0.5, TTS_SAMPLES, dtype=np.float32)

        def do_get_sample_rate(self):
            return TTS_SAMPLE_RATE

    leaf = FakeTts()
    leaf.engine_name = None
    return leaf


def expected_tts_duration_ns():
    return int(TTS_SAMPLES / TTS_SAMPLE_RATE * 1_000_000_000)


def test_gst_tts_stamps_its_own_duration_and_no_presentation_time():
    Gst = gst()
    leaf = tts_leaf()

    ret, pushed = drive_gst_payload(leaf, b"speak this")

    assert ret == Gst.FlowReturn.OK
    assert len(pushed) == 1
    assert len(buffer_bytes(pushed[0])) == TTS_SAMPLES * 2, "S16LE, one channel"
    assert (
        pushed[0].pts == Gst.CLOCK_TIME_NONE
    ), "the text buffer's pts is not the audio's"
    assert pushed[0].dts == Gst.CLOCK_TIME_NONE
    assert pushed[0].duration == expected_tts_duration_ns()
    assert leaf.srcpad.push_count == 1, "this family has never used finish_buffer"


def test_g2g_tts_emits_the_duration_the_audio_actually_runs_for():
    leaf = tts_leaf()
    sink = StubMetaSink()

    leaf.g2g_process_payload([bytearray(b"speak this")], "text/x-raw,format=utf8", sink)

    assert len(sink.emitted) == 1
    assert len(sink.emitted[0]) == TTS_SAMPLES * 2
    assert sink.emitted_durations == [expected_tts_duration_ns()]


def test_tts_streaming_speaks_each_chunk_of_text_separately():
    """Streaming splits the text into 20 character chunks, one payload each."""
    gst()
    leaf = tts_leaf()
    leaf.streaming = True

    _, pushed = drive_gst_payload(leaf, b"x" * 45)

    assert len(pushed) == 3, "45 characters is three chunks"
    assert {buf.duration for buf in pushed} == {expected_tts_duration_ns()}


def test_g2g_tts_streaming_emits_each_chunk_of_text_separately():
    leaf = tts_leaf()
    leaf.streaming = True
    sink = StubMetaSink()

    leaf.g2g_process_payload([bytearray(b"x" * 45)], "text/x-raw,format=utf8", sink)

    assert len(sink.emitted) == 3, "45 characters is three chunks"
    assert sink.emitted_durations == [expected_tts_duration_ns()] * 3


def test_gst_driver_keeps_the_input_timing_including_dts():
    """base_llm timestamped its output with the input's dts; the driver, which
    now builds that buffer, has to keep doing it."""
    Gst = gst()
    leaf = llm_leaf()

    pushed = []
    leaf.finish_buffer = pushed.append
    leaf.srcpad = StubSrcPad(pushed)
    inbuf = Gst.Buffer.new_allocate(None, len(b"question"), None)
    inbuf.fill(0, b"question")
    inbuf.pts = 90
    inbuf.dts = 80
    inbuf.duration = 70

    from backend.gst.aggregator import BaseAggregator as GstBaseAggregator

    GstBaseAggregator.do_process(leaf, inbuf)

    assert (pushed[0].pts, pushed[0].dts, pushed[0].duration) == (90, 80, 70)


def test_a_model_that_cannot_load_raises_to_the_host():
    """On g2g there is no bus to post an error on, so the load failure reaches
    the host as the exception it was."""
    from engine.engine_factory import EngineFactory
    from engine.ml_engine import MLEngine

    model_name = "does-not-exist.pt"
    load_failure = "no weights at does-not-exist.pt"

    class FailingEngine(MLEngine):
        def do_load_model(self, model_name, **kwargs):
            raise FileNotFoundError(load_failure)

        def do_set_device(self, device):
            self.device = device

        def do_forward(self, frames):
            raise AssertionError("inference ran without a model")

        def do_generate(self, input_text, max_length=1000, system_prompt=None):
            raise AssertionError("generation ran without a model")

    class LoadFails(VideoTransform):
        def process_frames(self, frames, num_sources, fmt, target):
            raise AssertionError("a frame was processed without a model")

    EngineFactory.register("test_failing_engine", FailingEngine)
    elem = LoadFails()
    elem.engine_name = "test_failing_engine"
    elem.model_name = model_name

    with pytest.raises(FileNotFoundError, match=load_failure):
        elem.g2g_process(bytearray(8 * 8 * 3), 8, 8, "RGB", StubMetaSink())

    assert elem.engine.model is None


def test_error_helpers_raise_to_the_host():
    with pytest.raises(ValueError, match="no weights"):
        backend.post_model_load_error(None, "m.pt", ValueError("no weights"))
    with pytest.raises(ValueError, match="no weights"):
        backend.post_error(None, "summary", ValueError("no weights"))


def test_vad_gate_mutes_a_buffer_below_the_speech_threshold_in_place(monkeypatch):
    stub_vad(monkeypatch)
    from vad import VoiceActivityDetector

    vad = VoiceActivityDetector()
    vad.gate = True
    buf = bytearray(silence() + speech())
    sink = StubMetaSink()

    vad.g2g_process_payload([buf], vad.INPUT_CAPS, sink)

    assert not any(buf), "half speech scores 0.5, under the 0.7 threshold"
    assert sink.staged_blobs == [("GST-VAD:", struct.pack("f", 0.5))]


def test_vad_gate_leaves_speech_alone(monkeypatch):
    stub_vad(monkeypatch)
    from vad import VoiceActivityDetector

    vad = VoiceActivityDetector()
    vad.gate = True
    buf = bytearray(speech(2))
    sink = StubMetaSink()

    vad.g2g_process_payload([buf], vad.INPUT_CAPS, sink)

    assert bytes(buf) == speech(2)
    assert sink.staged_blobs == [("GST-VAD:", struct.pack("f", 1.0))]


class ScriptedClapEngine:
    def do_set_device(self, device):
        pass

    def do_load_model(self, model_name, labels):
        pass

    def do_forward(self, audio):
        return [("siren", 0.9), ("music", 0.2)]


def test_clap_stages_the_labels_scoring_over_its_threshold(monkeypatch):
    import clap

    monkeypatch.setattr(clap, "ClapEngine", ScriptedClapEngine)
    element = clap.ClapTransform()
    sink = StubMetaSink()
    audio = bytearray(np.ones(480, dtype=np.float32).tobytes())

    element.g2g_process_payload([audio], element.INPUT_CAPS, sink)

    assert sink.staged_blobs == [
        ("GST-CLAP:", json.dumps([{"label": "siren", "score": 0.9}]).encode())
    ]


def test_gst_in_place_audio_writes_the_samples_and_appends_the_blob_after_them():
    Gst = gst()
    from backend.gst.frameio import GstFrameIO
    from backend.gst.in_place_audio_transform import (
        InPlaceAudioTransform as GstInPlaceAudioTransform,
    )

    class MutingTransform(GstInPlaceAudioTransform):
        def process_in_place(self, samples, buf):
            samples[:] = 0
            GstFrameIO().append_blob(buf, b"GST-TEST:", b"muted")

    buf = Gst.Buffer.new_allocate(None, 4, None)
    buf.fill(0, b"\x01\x02\x03\x04")

    assert MutingTransform().do_transform_ip(buf) == Gst.FlowReturn.OK

    memories = []
    for index in range(buf.n_memory()):
        success, map_info = buf.peek_memory(index).map(Gst.MapFlags.READ)
        assert success
        memories.append(bytes(map_info.data))
        buf.peek_memory(index).unmap(map_info)
    assert memories == [bytes(4), b"GST-TEST:muted"]


BASEBALL_TRAIL_SIGHTINGS = 4


def baseball_trail_row_after(frames_without_ball):
    from baseball_overlay import BaseballOverlay

    overlay = BaseballOverlay()
    overlay.show_labels = False
    overlay.show_equipment = False
    width, height = 1280, 720
    batter = {"label": "person_id_1", "score": 1.0, **BATTER_BOX}
    for step in range(BASEBALL_TRAIL_SIGHTINGS + frames_without_ball):
        frame = bytearray(width * height * 4)
        upstream = [batter]
        if step < BASEBALL_TRAIL_SIGHTINGS:
            ball = {
                "label": "sports ball",
                "score": 0.9,
                "x": BALL_START_X + step * BALL_STEP_PIXELS,
                "y": BALL_Y,
                "w": BALL_SIZE,
                "h": BALL_SIZE,
            }
            upstream.append(ball)
        overlay.g2g_process(
            frame, width, height, "RGBA", StubMetaSink(upstream_objects=upstream)
        )
    pixels = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 4)
    trail_y = BALL_Y + BALL_SIZE // 2
    return pixels[trail_y - 2 : trail_y + 3, BALL_START_X : BALL_START_X + 40]


def test_baseball_overlay_clears_the_trail_once_the_ball_stays_unseen():
    pytest.importorskip("cv2")
    from baseball_overlay import BALL_TRAIL_MAX_GAP

    assert baseball_trail_row_after(BALL_TRAIL_MAX_GAP).any(), "trail kept in the gap"
    assert not baseball_trail_row_after(BALL_TRAIL_MAX_GAP + 1).any()


def baseball_ball_ring_drawn(ball_x, ball_y):
    from baseball_overlay import BaseballOverlay

    overlay = BaseballOverlay()
    overlay.show_labels = False
    overlay.show_equipment = False
    overlay.ball_trail = False
    width, height = 1280, 720
    upstream = [
        {"label": "person_id_1", "score": 1.0, **BATTER_BOX},
        {"label": "baseball bat", "score": 0.9, **BAT_BOX},
        {
            "label": "sports ball",
            "score": 0.9,
            "x": ball_x,
            "y": ball_y,
            "w": 20,
            "h": 20,
        },
    ]
    frame = bytearray(width * height * 4)
    overlay.g2g_process(
        frame, width, height, "RGBA", StubMetaSink(upstream_objects=upstream)
    )
    pixels = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 4)
    return pixels[ball_y - 5 : ball_y + 25, ball_x - 5 : ball_x + 25].any()


def test_baseball_overlay_ignores_a_ball_on_the_batter_or_the_bat():
    pytest.importorskip("cv2")
    assert baseball_ball_ring_drawn(900, 100), "a ball in the open is drawn"
    assert not baseball_ball_ring_drawn(
        BAT_BOX["x"] + 5, BAT_BOX["y"] + 5
    ), "ball on the bat"
    assert not baseball_ball_ring_drawn(
        BATTER_BOX["x"] + 10, BATTER_BOX["y"] + 200
    ), "ball on the batter"


def test_baseball_overlay_merges_the_parts_of_one_bat_into_one_box():
    from baseball_overlay import merge_bat_parts

    handle = {"box": (549, 252, 785, 394), "confidence": 0.8}
    barrel = {"box": (705, 90, 1115, 310), "confidence": 0.5}
    apart_barrel = {"box": (937, 112, 1226, 246), "confidence": 0.6}
    other_bat = {"box": (100, 600, 300, 660), "confidence": 0.4}
    assert merge_bat_parts([barrel, handle, other_bat]) == [
        {"box": (549, 90, 1115, 394), "confidence": 0.8},
        other_bat,
    ]
    assert merge_bat_parts([handle, apart_barrel, other_bat]) == [
        {"box": (549, 112, 1226, 394), "confidence": 0.8},
        other_bat,
    ]
