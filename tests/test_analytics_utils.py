import os
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
PLUGIN_DIR = BASE_DIR / "plugins" / "python"

os.environ["GST_PLUGIN_PATH"] = str(BASE_DIR / "plugins")
sys.path.insert(0, str(PLUGIN_DIR))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

from utils.analytics_utils import AnalyticsUtils  # noqa: E402


@pytest.mark.parametrize(
    ("full_label", "expected"),
    [
        ("person_id_7", (7, "person")),
        ("sports ball_id_3", (3, "sports ball")),
        ("id_5", (5, "id_5")),
        ("dog", (None, "dog")),
        ("whatever", (None, "whatever")),
    ],
)
def test_extract_id_from_label(full_label, expected):
    assert AnalyticsUtils().extract_id_from_label(full_label) == expected


def test_extract_analytics_metadata_reads_the_chunk(monkeypatch):
    from backend.gst.chunk_analytics import ChunkAnalytics
    import utils.analytics_utils

    chunk_analytics = ChunkAnalytics()
    monkeypatch.setattr(utils.analytics_utils, "analytics", chunk_analytics)
    buffer = Gst.Buffer.new_wrapped(bytes(16))
    meta = chunk_analytics.add_relation_meta(buffer)
    chunk_analytics.add_object(meta, "stream_0_person_id_7", 10, 20, 30, 40, 0.5)
    chunk_analytics.add_object(meta, "clip_red_car", 0, 0, 16, 1, 0.25)

    assert AnalyticsUtils().extract_analytics_metadata(buffer) == [
        {
            "label": "person",
            "track_id": 7,
            "confidence": 0.5,
            "box": {"x1": 10, "y1": 20, "x2": 40, "y2": 60},
        },
        {"type": "classification", "label": "red car", "confidence": 0.25},
    ]
