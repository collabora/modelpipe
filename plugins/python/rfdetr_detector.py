# RFDETRDetector
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from base_objectdetector import BaseObjectDetector
from backend import GObject
from engine.engine_factory import EngineFactory
from engine.rfdetr_engine import RFDETREngine
import backend


class RFDETRDetector(BaseObjectDetector):
    __gstmetadata__ = (
        "RF-DETR",
        "Transform",
        "Detects objects on video frames with an RF-DETR checkpoint",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    confidence = GObject.Property(
        type=float,
        default=0.25,
        minimum=0.0,
        maximum=1.0,
        nick="Confidence Threshold",
        blurb="Minimum detection confidence",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.task_engine_name = self.mgr.engine_name = "mp_rfdetr_engine"
        EngineFactory.register(self.mgr.engine_name, RFDETREngine)

    def do_forward(self, frames):
        self.task_engine.conf = self.confidence
        return super().do_forward(frames)


if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("mp_rfdetr", RFDETRDetector)
