# Depth
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
    from engine.depth_anything_engine import (
        DepthAnythingEngine,
        ExportedDepthAnything,
        KerasHubDepthAnything,
    )
    from engine.engine_factory import EngineFactory
    from backend import GObject
    from tasks.depth import DepthTask

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'depth' element will not be available. Error {e}")

# Header prefix for depth map buffer metadata
DEPTH_META_HEADER = b"GST-DEPTH:"


class DepthTransform(VideoTransform, DepthTask):
    """
    GStreamer element for monocular depth estimation using DepthAnything V2.

    Set model-name to a HuggingFace model ID, e.g.:
      depth-anything/Depth-Anything-V2-Small-hf

    When visualize=True (default), the video frame is replaced with a
    colorized depth map. Use a tee element upstream to preserve the original
    video alongside the depth visualization.

    A uint8 normalized depth map is always appended to the buffer as a
    GST-DEPTH: memory chunk for downstream elements:
      for i in range(buf.n_memory()):
          data = bytes(buf.peek_memory(i).map(Gst.MapFlags.READ).data)
          if data.startswith(b"GST-DEPTH:"):
              depth = np.frombuffer(data[10:], dtype=np.uint8).reshape(H, W)

    Use frame-stride to skip frames and reduce compute:
      mp_depth model-name=depth-anything/Depth-Anything-V2-Small-hf frame-stride=2
    """

    META_HEADER = DEPTH_META_HEADER
    FRAME_CAPABILITIES = backend.CUDA_INFERENCE_FRAME_CAPABILITIES

    __gstmetadata__ = (
        "Depth",
        "Transform",
        "Monocular depth estimation using DepthAnything V2",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    visualize = GObject.Property(
        type=bool,
        default=True,
        nick="Visualize Depth",
        blurb="Replace video frame with a colorized depth map",
        flags=GObject.ParamFlags.READWRITE,
    )

    colormap = GObject.Property(
        type=str,
        default="inferno",
        nick="Colormap",
        blurb="Colormap for depth visualization: inferno, jet, viridis, plasma, magma",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.task_engine_name = self.mgr.engine_name = "mp_depth_engine"
        EngineFactory.register(self.mgr.engine_name, DepthAnythingEngine)
        self.format_converter = FormatConverter()

    def export_model(self, model_name):
        if self.mgr.engine_name == EngineFactory.JAX_ENGINE:
            return KerasHubDepthAnything(model_name, self.mgr.engine_name)
        return ExportedDepthAnything(model_name, self.mgr.engine_name)


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("mp_depth", DepthTransform)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_depth' element will not be registered because required modules are missing."
    )
