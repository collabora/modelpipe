# BaseSink (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from backend.g2g.analytics import analytics
from backend.g2g.frameio import DEVICE_FRAME, frameio
from backend.g2g.in_place_video_transform import DEVICE_DRAWING_FORMAT
from backend.g2g.shims import declared_property_names
from backend.video_frame import SYSTEM_READ_FRAME_CAPABILITIES
from log.logger_factory import LoggerFactory


class BaseSink:
    FRAME_CAPABILITIES = SYSTEM_READ_FRAME_CAPABILITIES

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
        self.render(frame, self.format, buf)
        return None

    def g2g_process_cuda(self, luma, chroma, width, height, yuv_conversion, sink):
        self.width = width
        self.height = height
        self.format = DEVICE_DRAWING_FORMAT
        frameio.bind(sink, self.format)
        analytics.bind(sink)
        if self.FRAME_CAPABILITIES.reads_pixels:
            raise ValueError(
                f"{type(self).__name__} reads only system memory, put "
                "videoconvert between the decoder and it"
            )
        self.render(None, self.format, DEVICE_FRAME)
        return None

    def g2g_eos(self):
        self.stop()

    def stop(self):
        pass
