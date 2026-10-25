import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

from backend import frameio  # noqa: E402
from backend.gst.analytics import (  # noqa: E402
    ANALYTICS_STORE_VARIABLE,
    gstanalytics_importable,
    select_analytics,
)
from backend.gst.chunk_analytics import ChunkAnalytics  # noqa: E402
from utils.metadata import BATCH_SOURCE_INDEX_METADATA  # noqa: E402

VIDEO_FRAME_SIZE = 640 * 480 * 3
DEPTH_HEADER = b"GST-DEPTH:"
DEPTH_PAYLOAD = b'{"mean": 2.5}'
FRAME_FILL = 7
PERSON = {"label": "stream_0_person", "x": 10, "y": 20, "w": 30, "h": 40, "score": 0.9}
BALL = {"label": "stream_0_ball", "x": 1, "y": 2, "w": 3, "h": 4, "score": 0.5}
DOG = {"label": "stream_0_dog", "x": 5, "y": 6, "w": 7, "h": 8, "score": 0.25}


def add(analytics, meta, detection):
    return analytics.add_object(meta, *detection.values())


def read_back(detection, ref, track_id=None):
    stream, label = detection["label"].removeprefix("stream_").split("_", 1)
    return {
        **detection,
        "label": label,
        "stream": int(stream),
        "track_id": track_id,
        "ref": ref,
    }


def test_the_chunk_carries_objects_beside_another_blob_until_removed():
    analytics = ChunkAnalytics()
    buffer = Gst.Buffer.new_wrapped(bytes(VIDEO_FRAME_SIZE))

    meta = analytics.add_relation_meta(buffer)
    person = add(analytics, meta, PERSON)
    add(analytics, meta, BALL)
    analytics.add_classification(meta, 0, "stream_0_outdoor")
    tracking = analytics.add_tracking(meta, 7, 0)
    assert analytics.relate(meta, person, tracking)

    assert analytics.relation_length(meta) == 4
    tracked_person = read_back(PERSON, person.id, track_id=7)
    assert analytics.read_objects(meta) == [tracked_person, read_back(BALL, 1)]
    attached = analytics.get_relation_meta(buffer)
    assert analytics.read_objects(attached) == [tracked_person, read_back(BALL, 1)]
    assert attached.relations == [[person.id, tracking.id]]

    frameio.append_blob(buffer, DEPTH_HEADER, DEPTH_PAYLOAD)
    add(analytics, analytics.get_relation_meta(buffer), DOG)
    assert buffer.n_memory() == 3
    assert analytics.read_objects(analytics.get_relation_meta(buffer)) == [
        tracked_person,
        read_back(BALL, 1),
        read_back(DOG, 4),
    ]
    assert frameio.read_blobs(buffer)["depth"] == DEPTH_PAYLOAD

    assert analytics.remove_relation_meta(buffer)
    assert buffer.n_memory() == 2
    assert analytics.get_relation_meta(buffer) is None
    assert frameio.read_blobs(buffer) == {"depth": DEPTH_PAYLOAD}


def test_a_chunk_appended_after_the_batch_metadata_is_found():
    analytics = ChunkAnalytics()
    buffer = Gst.Buffer.new_wrapped(bytes(VIDEO_FRAME_SIZE))
    buffer.append_memory(Gst.Buffer.new_wrapped(bytes(VIDEO_FRAME_SIZE)).get_memory(0))
    BATCH_SOURCE_INDEX_METADATA.write(buffer, [(0,), (1,)])

    add(analytics, analytics.add_relation_meta(buffer), PERSON)
    add(analytics, analytics.get_relation_meta(buffer), BALL)

    assert buffer.n_memory() == 4
    assert analytics.read_objects(analytics.get_relation_meta(buffer)) == [
        read_back(PERSON, 0),
        read_back(BALL, 1),
    ]
    assert BATCH_SOURCE_INDEX_METADATA.read(buffer) == ((0,), (1,))


def test_the_environment_picks_the_store(monkeypatch):
    monkeypatch.setenv(ANALYTICS_STORE_VARIABLE, "chunk")
    assert isinstance(select_analytics(), ChunkAnalytics)

    monkeypatch.setenv(ANALYTICS_STORE_VARIABLE, "bogus")
    with pytest.raises(ImportError, match="'meta', 'chunk'"):
        select_analytics()

    if not gstanalytics_importable():
        pytest.skip("this GStreamer has no GstAnalytics")
    from backend.gst.relation_meta import RelationMetaAnalytics

    monkeypatch.setenv(ANALYTICS_STORE_VARIABLE, "meta")
    assert isinstance(select_analytics(), RelationMetaAnalytics)


def test_a_gstreamer_without_gstanalytics_gets_the_chunk(monkeypatch):
    real_require_version = gi.require_version

    def require_version(namespace, version):
        if namespace == "GstAnalytics":
            raise ValueError("Namespace GstAnalytics not available")
        return real_require_version(namespace, version)

    monkeypatch.delenv(ANALYTICS_STORE_VARIABLE, raising=False)
    monkeypatch.setattr(gi, "require_version", require_version)
    assert isinstance(select_analytics(), ChunkAnalytics)


def test_writing_the_frame_leaves_the_chunk_in_place():
    analytics = ChunkAnalytics()
    buffer = Gst.Buffer.new_wrapped(bytes(VIDEO_FRAME_SIZE))
    add(analytics, analytics.add_relation_meta(buffer), PERSON)

    np = pytest.importorskip("numpy")
    frameio.write_frame(buffer, np.full(VIDEO_FRAME_SIZE, FRAME_FILL, dtype=np.uint8))

    assert buffer.n_memory() == 2
    assert buffer.extract_dup(0, 1) == bytes([FRAME_FILL])
    assert analytics.read_objects(analytics.get_relation_meta(buffer)) == [
        read_back(PERSON, 0)
    ]


def test_metasink_writes_the_chunk_only_as_detections(monkeypatch):
    import metasink

    analytics = ChunkAnalytics()
    monkeypatch.setattr(metasink, "analytics", analytics)
    buffer = Gst.Buffer.new_wrapped(bytes(VIDEO_FRAME_SIZE))
    buffer.pts = 0
    add(analytics, analytics.add_relation_meta(buffer), PERSON)
    frameio.append_blob(buffer, DEPTH_HEADER, DEPTH_PAYLOAD)

    assert metasink.buffer_record(buffer, "video/x-raw") == {
        "pts": 0.0,
        "detections": [read_back(PERSON, 0)],
        "depth": {"mean": 2.5},
    }
