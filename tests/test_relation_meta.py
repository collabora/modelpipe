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

from backend.gst.analytics import gstanalytics_importable  # noqa: E402

pytestmark = pytest.mark.skipif(
    not gstanalytics_importable(), reason="needs the GstAnalytics bindings"
)


def test_a_copied_buffer_keeps_its_detections_and_track_ids():
    from backend.gst.analytics import relation_meta_analytics

    analytics = relation_meta_analytics()
    buffer = Gst.Buffer.new_wrapped(bytes(16))
    meta = analytics.add_relation_meta(buffer)
    analytics.add_object(meta, "person", 3, 4, 5, 6, 0.5)
    (person,) = analytics.read_objects(meta)
    analytics.track_object(meta, person, 7)

    copy = buffer.copy_deep()

    (copied_person,) = analytics.read_objects(analytics.get_relation_meta(copy))
    assert {key: copied_person[key] for key in ("label", "x", "y", "w", "h")} == {
        "label": "person",
        "x": 3,
        "y": 4,
        "w": 5,
        "h": 6,
    }
    assert copied_person["track_id"] == 7
