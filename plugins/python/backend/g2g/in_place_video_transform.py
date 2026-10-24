# InPlaceVideoTransform (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from backend.cuda_driver import primary_cuda_context
from backend.g2g.analytics import analytics
from backend.g2g.cuda_frames import rgb_to_nv12, rgba_device_frame
from backend.g2g.frameio import DEVICE_FRAME, frameio
from backend.g2g.shims import declared_property_names
from backend.video_frame import SYSTEM_DRAW_FRAME_CAPABILITIES
from log.logger_factory import LoggerFactory

DEVICE_DRAWING_FORMAT = "RGBA"
COLOUR_CHANNEL_COUNT = 3


class InPlaceVideoTransform:
    FRAME_CAPABILITIES = SYSTEM_DRAW_FRAME_CAPABILITIES

    def __init__(self):
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.width = 0
        self.height = 0
        self.format = None

    def g2g_properties(self):
        return declared_property_names(type(self))

    def g2g_process(self, buf, width, height, fmt, sink):
        self.width = width
        self.height = height
        self.format = (fmt or "RGB").upper()
        frameio.bind(sink, fmt)
        analytics.bind(sink)
        frame = None
        if self.FRAME_CAPABILITIES.reads_pixels:
            frame = frameio.read_frame(buf, None, width, height)
        self.process_in_place(frame, self.format, buf)
        return None

    def g2g_process_cuda(self, luma, chroma, width, height, yuv_conversion, sink):
        import torch

        self.width = width
        self.height = height
        self.format = DEVICE_DRAWING_FORMAT
        frameio.bind(sink, self.format)
        analytics.bind(sink)
        if not self.FRAME_CAPABILITIES.reads_pixels:
            self.process_in_place(None, self.format, DEVICE_FRAME)
            return None
        if not self.FRAME_CAPABILITIES.accepts_cuda:
            raise ValueError(
                f"{type(self).__name__} draws only on system memory, put "
                "videoconvert between the decoder and it"
            )
        frame = rgba_device_frame(luma, chroma, yuv_conversion)
        self.process_in_place(frame, self.format, DEVICE_FRAME)
        drawn_planes = rgb_to_nv12(frame[..., :COLOUR_CHANNEL_COUNT], yuv_conversion)
        self.cuda_device = frame.device.index
        self.cuda_context = primary_cuda_context(self.cuda_device)
        # downstream reads the drawn planes on its own stream once this returns
        torch.cuda.current_stream().synchronize()
        return drawn_planes
