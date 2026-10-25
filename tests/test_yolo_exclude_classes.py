import os
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))
os.environ["YOLO_AUTOINSTALL"] = "false"

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
gi.require_version("GstAnalytics", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)
pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from backend import analytics  # noqa: E402
from yolo_reference import PORTRAIT, YOLO_MODEL  # noqa: E402


def decoded_labels(exclude_classes):
    import cv2
    from yolo import YOLOTransform

    element = YOLOTransform()
    element.set_property("model-name", YOLO_MODEL)
    element.set_property("engine-name", "pytorch")
    element.set_property("device", "cpu")
    element.set_property("exclude-classes", exclude_classes)
    element.do_load_model()
    frame = cv2.cvtColor(cv2.imread(str(PORTRAIT)), cv2.COLOR_BGR2RGB)
    result = element.do_forward(frame)
    buffer = Gst.Buffer.new_wrapped(frame.tobytes())
    element.do_decode(buffer, result)
    meta = analytics.get_relation_meta(buffer)
    return [found["label"] for found in analytics.read_objects(meta)]


def test_excluded_class_is_left_out_of_the_metadata():
    all_labels = decoded_labels("")
    persons = [label for label in all_labels if label.endswith("person")]
    assert persons

    kept = decoded_labels("person,ball")
    assert not any(label.endswith("person") for label in kept)
    assert len(kept) == len(all_labels) - len(persons)
