# Super Resolution
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    from video_transform import VideoTransform
    from utils.format_converter import FormatConverter
    from engine.super_res_engine import ExportedSuperRes, SuperResEngine
    from engine.engine_factory import EngineFactory
    from backend import frameio
    from tasks.superres import SuperResTask

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'superres' element will not be available. Error {e}")


DEFAULT_MODEL_NAME = "real-esrgan-x4"


class SuperResTransform(VideoTransform, SuperResTask):
    """
    GStreamer element for image super-resolution using Real-ESRGAN.

    Set model-name to a Real-ESRGAN variant: real-esrgan-x4 or real-esrgan-x2.

    The upscaled frame is resized back to the original buffer dimensions
    (in-place transform). This provides enhanced detail while maintaining
    pipeline compatibility. Use frame-stride to reduce compute load.
    """

    __gstmetadata__ = (
        "Super Resolution",
        "Transform",
        "Image super-resolution using Real-ESRGAN",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )
    FRAME_CAPABILITIES = backend.CUDA_INFERENCE_FRAME_CAPABILITIES

    def __init__(self):
        super().__init__()
        self.task_engine_name = self.mgr.engine_name = "mp_superres_engine"
        EngineFactory.register(self.mgr.engine_name, SuperResEngine)
        self.format_converter = FormatConverter()
        self.model_name = DEFAULT_MODEL_NAME

    def export_model(self, model_name):
        return ExportedSuperRes(model_name, self.mgr.engine_name)

    def process_frames(self, frames, num_sources, fmt, target):
        """Upscale the primary frame and write it back at the original size."""
        frame = frames[0] if frames.ndim == 4 else frames
        upscaled = self.forward(frame)
        if upscaled is None:
            return

        output, _blob = self.decode(upscaled, fmt)
        frameio.write_result(target, output)


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_superres", SuperResTransform
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_superres' element will not be registered because required modules are missing."
    )
