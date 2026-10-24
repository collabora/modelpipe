# Yolo
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
from backend import GObject, CUDA_INFERENCE_FRAME_CAPABILITIES
import backend

CAN_REGISTER_ELEMENT = True
try:
    from base_objectdetector import BaseObjectDetector
    from tasks.yolo import YoloTask

    from engine.yolo_engine import ExportedYolo, YoloEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'yolo' element will not be available. Error {e}")


class YOLOTransform(BaseObjectDetector, YoloTask):
    """
    GStreamer element shell for YOLO model inference on video frames
    (detection, segmentation, and tracking). The result handling (do_decode)
    is inherited from the backend-agnostic YoloTask; this class supplies the
    engine wiring and registration.
    """

    __gstmetadata__ = (
        "YOLO",
        "Transform",
        "Performs object detection, segmentation, and tracking using YOLO on video frames",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )
    FRAME_CAPABILITIES = CUDA_INFERENCE_FRAME_CAPABILITIES

    imgsz = GObject.Property(
        type=int,
        default=640,
        minimum=32,
        maximum=4096,
        nick="Input Size",
        blurb="Longest side the frame is scaled to for the network; 1280 finds "
        "a ball a few pixels wide that 640 misses, at three times the cost",
        flags=GObject.ParamFlags.READWRITE,
    )

    confidence = GObject.Property(
        type=float,
        default=0.1,
        minimum=0.0,
        maximum=1.0,
        nick="Confidence Threshold",
        blurb="Minimum detection confidence (matches football_analyzer); kept "
        "low on purpose so the tracker can use weak boxes to continue tracks "
        "-- the tracker's new-track-confidence gates phantom tracks",
        flags=GObject.ParamFlags.READWRITE,
    )
    nms_iou = GObject.Property(
        type=float,
        default=0.7,
        minimum=0.0,
        maximum=1.0,
        nick="NMS IoU",
        blurb="NMS IoU threshold (matches football_analyzer's default); lower "
        "suppresses more overlap but can also drop genuinely close players",
        flags=GObject.ParamFlags.READWRITE,
    )
    agnostic_nms = GObject.Property(
        type=bool,
        default=False,
        nick="Class-Agnostic NMS",
        blurb="Suppress overlapping boxes across classes too; off by default "
        "(like football_analyzer) so two close players aren't merged",
        flags=GObject.ParamFlags.READWRITE,
    )
    exclude_classes = GObject.Property(
        type=str,
        default="",
        nick="Excluded Classes",
        blurb="Comma-separated class names whose boxes are dropped, so a second "
        "detector can supply that class instead",
        flags=GObject.ParamFlags.READWRITE,
    )
    tensorrt = GObject.Property(
        type=bool,
        default=False,
        nick="TensorRT",
        blurb="Run CUDA frames through a half precision TensorRT engine when "
        "track is off. The engine is built from the weights on the first frame, "
        "which takes a minute or two, and cached in ~/.cache/modelpipe/tensorrt",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.task_engine_name = self.mgr.engine_name = "mp_yolo_engine"
        EngineFactory.register(self.mgr.engine_name, YoloEngine)

    def export_model(self, model_name):
        return ExportedYolo(model_name, self.mgr.engine_name)

    def do_forward(self, frames):
        # Push NMS/confidence knobs to the engine before it runs the model.
        if self.task_engine:
            self.task_engine.imgsz = self.imgsz
            self.task_engine.conf = self.confidence
            self.task_engine.iou = self.nms_iou
            self.task_engine.agnostic_nms = self.agnostic_nms
            self.task_engine.tensorrt = self.tensorrt
        return super().do_forward(frames)


# The class is backend-agnostic: under g2g the host imports this module and
# instantiates YOLOTransform directly, so no GObject registration applies.
# GStreamer factory registration runs only under the gst backend.
if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("mp_yolo", YOLOTransform)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_yolo' element will not be registered because required modules are missing."
    )
