# Pose
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    import json

    from backend import analytics, frameio, GObject
    from base_objectdetector import BaseObjectDetector
    from engine.yolo_pose_engine import ExportedYoloPose, YoloPoseEngine
    from engine.engine_factory import EngineFactory
    from overlay_helper.overlay_torch import TorchOverlayGraphics
    from overlay_helper.overlay_utils_interface import Color
    from utils.coco_pose import DRAWN_KEYPOINT_CONFIDENCE, coco_skeleton

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'pose' element will not be available. Error {e}")

# Header prefix for keypoints buffer metadata
POSE_META_HEADER = b"GST-POSE:"

KEYPOINT_RADIUS = 4
RGB_CHANNELS = 3
FULL_INTENSITY = 255
OPAQUE = 1.0


# ultralytics paints its palette onto a bgr frame
def palette_colour(bgr_row):
    return Color(*(channel / FULL_INTENSITY for channel in bgr_row[::-1]))


class YOLOPoseTransform(BaseObjectDetector):
    """
    GStreamer element for human pose estimation using YOLO on video frames.

    Detects persons and estimates their 17 COCO body keypoints. Bounding boxes
    are attached as GstAnalytics metadata for use with mp_overlay. Keypoints
    are appended as a JSON blob in a GST-POSE: buffer memory chunk for use
    by downstream elements.

    When visualize=True (default), the skeleton and keypoints are drawn
    directly on the video frame using Ultralytics' built-in renderer. On a
    CUDA frame they are drawn in place on the GPU, without the boxes.

    Recommended model names: yolo11n-pose, yolo11s-pose, yolo11m-pose
    """

    __gstmetadata__ = (
        "YOLO Pose",
        "Transform",
        "Human pose estimation with COCO 17-keypoint skeleton using YOLO",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )
    FRAME_CAPABILITIES = backend.CUDA_INFERENCE_FRAME_CAPABILITIES

    visualize = GObject.Property(
        type=bool,
        default=True,
        nick="Visualize Skeleton",
        blurb="Draw skeleton and keypoints directly on the video frame",
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
        self.task_engine_name = self.mgr.engine_name = "mp_yolo_pose_engine"
        EngineFactory.register(self.mgr.engine_name, YoloPoseEngine)
        self.frames_on_the_device = False
        self.frame_format = None
        self.device_drawing_key = None
        self.device_renderer = None
        self.pose_style = None

    def export_model(self, model_name):
        return ExportedYoloPose(model_name, self.mgr.engine_name)

    # a hosted element has no pad to read the format from
    def process_frames(self, frames, num_sources, fmt, target):
        self.frame_format = fmt
        super().process_frames(frames, num_sources, fmt, target)

    def do_forward(self, frames):
        if self.task_engine:
            self.task_engine.tensorrt = self.tensorrt
        self.frames_on_the_device = getattr(frames, "is_cuda", False)
        return super().do_forward(frames)

    def do_decode(self, buf, result, stream_idx=0):
        boxes = result.boxes
        keypoints = result.keypoints

        if boxes is None or len(boxes) == 0:
            self.logger.info(f"Stream {stream_idx}: no persons detected")
            return

        # Attach person bounding boxes via GstAnalytics (compatible with mp_overlay)
        meta = analytics.add_relation_meta(buf)
        if not meta:
            self.logger.error("Failed to add analytics relation metadata")
            return

        kp_data_list = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i]
            score = boxes.conf[i].item()

            mtd = analytics.add_object(
                meta,
                f"stream_{stream_idx}_person",
                x1.item(),
                y1.item(),
                x2.item() - x1.item(),
                y2.item() - y1.item(),
                score,
            )
            if mtd is None:
                self.logger.error(f"Failed to add od_mtd for person {i}")
                continue

            # Collect keypoints for this person
            if keypoints is not None and i < len(keypoints):
                kp_xy = keypoints.xy[i].cpu().numpy().tolist()
                kp_conf = (
                    keypoints.conf[i].cpu().numpy().tolist()
                    if keypoints.conf is not None
                    else [1.0] * len(kp_xy)
                )
                kp_data_list.append(
                    {
                        "person": i,
                        "box": [x1.item(), y1.item(), x2.item(), y2.item()],
                        "score": score,
                        "keypoints": kp_xy,
                        "kp_conf": kp_conf,
                    }
                )

        if self.visualize:
            self._write_annotated_frame(buf, result)

        if kp_data_list:
            frameio.append_blob(
                buf, POSE_META_HEADER, json.dumps(kp_data_list).encode("utf-8")
            )
            self.logger.debug(
                f"Stream {stream_idx}: appended keypoints for "
                f"{len(kp_data_list)} persons"
            )

    def _write_annotated_frame(self, buf, result):
        """Write result.plot() (BGR) back into the GStreamer buffer."""
        if self.frames_on_the_device:
            self._draw_on_the_device(buf, result)
            return
        annotated_bgr = result.plot(kpt_line=True, kpt_radius=KEYPOINT_RADIUS)
        fmt = self.frame_format
        output = self._convert_bgr_to_format(annotated_bgr, fmt)
        if output is None:
            return
        frameio.write_frame(buf, output)

    def _draw_on_the_device(self, buf, result):
        keypoints = result.keypoints
        if keypoints is None or keypoints.conf is None:
            return
        fmt = self.frame_format
        renderer, style = self._device_pose_drawing(fmt)
        points = keypoints.xy.cpu().numpy()
        visible = keypoints.conf.cpu().numpy() >= DRAWN_KEYPOINT_CONFIDENCE
        keypoint_rows = [
            renderer.square_row({"x": x, "y": y}, style.keypoint_colours[index], OPAQUE)
            for person, seen in zip(points, visible)
            for index, (x, y) in enumerate(person)
            if seen[index]
        ]
        limbs = [
            (
                {"x": person[first][0], "y": person[first][1]},
                {"x": person[second][0], "y": person[second][1]},
                colour,
            )
            for person, seen in zip(points, visible)
            for (first, second), colour in zip(style.skeleton, style.limb_colours)
            if seen[first] and seen[second]
        ]
        # the inference map must close before the buffer is mapped for writing
        frameio.release_frames(buf)
        try:
            with renderer.drawing_stream():
                renderer.initialize(
                    frameio.read_frame_for_drawing(buf, self.width, self.height, fmt)
                )
                renderer.draw_squares(keypoint_rows, 2 * KEYPOINT_RADIUS + 1)
                renderer.draw_segments(limbs, style.line_width)
                renderer.finalize()
        finally:
            frameio.release_frames(buf)

    def _device_pose_drawing(self, fmt):
        key = (self.width, self.height, fmt)
        if self.device_drawing_key == key:
            return self.device_renderer, self.pose_style
        import numpy as np
        from types import SimpleNamespace
        from ultralytics.utils.plotting import Annotator

        annotator = Annotator(
            np.zeros((self.height, self.width, RGB_CHANNELS), np.uint8)
        )
        self.pose_style = SimpleNamespace(
            line_width=annotator.lw,
            skeleton=coco_skeleton(),
            keypoint_colours=[palette_colour(row) for row in annotator.kpt_color],
            limb_colours=[palette_colour(row) for row in annotator.limb_color],
        )
        self.device_renderer = TorchOverlayGraphics(self.width, self.height, fmt)
        self.device_drawing_key = key
        return self.device_renderer, self.pose_style

    @staticmethod
    def _convert_bgr_to_format(bgr, fmt):
        """Convert a BGR numpy array to the target GStreamer video format."""
        import numpy as np
        import cv2

        if fmt == "RGB":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif fmt == "BGR":
            return bgr
        elif fmt == "RGBA":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        elif fmt == "BGRA":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        elif fmt == "ARGB":
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
            return np.roll(rgba, 1, axis=-1)  # RGBA -> ARGB
        elif fmt == "ABGR":
            bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            return np.roll(bgra, 1, axis=-1)  # BGRA -> ABGR
        else:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_yolo_pose", YOLOPoseTransform
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_yolo_pose' element will not be registered because required modules are missing."
    )
