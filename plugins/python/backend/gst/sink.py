# BaseSink (GStreamer backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstBase, GstVideo  # noqa: E402

from backend.gst.errors import post_error  # noqa: E402
from backend.gst.frameio import mapped_frame  # noqa: E402
from backend.gst.in_place_video_transform import (  # noqa: E402
    caps_for_frame_capabilities,
)
from backend.gst.video_transform import (  # noqa: E402
    SYSTEM_MEMORY_CAPS,
)
from backend.video_frame import SYSTEM_READ_FRAME_CAPABILITIES  # noqa: E402
from log.logger_factory import LoggerFactory  # noqa: E402


def sink_template(caps):
    return Gst.PadTemplate.new(
        "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, caps
    )


class BaseSink(GstBase.BaseSink):
    __gsttemplates__ = sink_template(Gst.Caps.from_string(SYSTEM_MEMORY_CAPS))

    FRAME_CAPABILITIES = SYSTEM_READ_FRAME_CAPABILITIES

    # runs before pygobject registers the subclass and reads its templates
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "FRAME_CAPABILITIES" not in cls.__dict__:
            return
        cls.__gsttemplates__ = sink_template(
            caps_for_frame_capabilities(cls.FRAME_CAPABILITIES)
        )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.set_sync(False)
        self.set_async_enabled(False)
        self.width = 0
        self.height = 0
        self.format = None
        self._bytes_per_pixel = 0

    def do_set_caps(self, caps):
        info = GstVideo.VideoInfo.new_from_caps(caps)
        self.width = info.width
        self.height = info.height
        self.format = info.finfo.name
        self._bytes_per_pixel = info.finfo.pixel_stride[0]
        return True

    def do_render(self, buf):
        if not self.FRAME_CAPABILITIES.reads_pixels:
            return self._run(None, buf)
        with mapped_frame(
            buf, Gst.MapFlags.READ, self.width, self.height, self._bytes_per_pixel
        ) as frame:
            if frame is None:
                self.logger.error("failed to map buffer for reading")
                return Gst.FlowReturn.ERROR
            return self._run(frame, buf)

    def do_stop(self):
        self.stop()
        return True

    def stop(self):
        pass

    def _run(self, frame, buf):
        try:
            self.render(frame, self.format, buf)
            return Gst.FlowReturn.OK
        except Exception as exception:
            post_error(self, "render error", exception)
            return Gst.FlowReturn.ERROR
