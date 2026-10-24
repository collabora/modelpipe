# InPlaceVideoTransform (GStreamer backend)
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

from backend.gst.cuda_memory import is_cuda_buffer  # noqa: E402
from backend.gst.errors import post_error  # noqa: E402
from backend.gst.frameio import mapped_frame  # noqa: E402
from backend.gst.video_transform import CUDA_MEMORY_FEATURE, pad_templates  # noqa: E402
from backend.video_frame import (  # noqa: E402
    ANY_PIXEL_FORMAT,
    SYSTEM_DRAW_FRAME_CAPABILITIES,
)
from log.logger_factory import LoggerFactory  # noqa: E402

ANY_VIDEO_CAPS = "video/x-raw"


def with_cuda_memory(system_caps):
    caps = Gst.Caps.new_empty()
    cuda_features = Gst.CapsFeatures.new_single(CUDA_MEMORY_FEATURE)
    for features in (cuda_features, None):
        for index in range(system_caps.get_size()):
            caps.append_structure_full(
                system_caps.get_structure(index).copy(),
                features.copy() if features else system_caps.get_features(index).copy(),
            )
    return caps


def caps_for_frame_capabilities(capabilities):
    if capabilities.pixel_formats == (ANY_PIXEL_FORMAT,):
        system_caps = Gst.Caps.from_string(ANY_VIDEO_CAPS)
    else:
        formats = ", ".join(capabilities.pixel_formats)
        system_caps = Gst.Caps.from_string(
            f"video/x-raw, format=(string){{ {formats} }}"
        )
    return with_cuda_memory(system_caps) if capabilities.accepts_cuda else system_caps


class InPlaceVideoTransform(GstBase.BaseTransform):
    VIDEO_CAPS = Gst.Caps.from_string(ANY_VIDEO_CAPS)
    __gsttemplates__ = pad_templates(VIDEO_CAPS)

    FRAME_CAPABILITIES = SYSTEM_DRAW_FRAME_CAPABILITIES

    # runs before pygobject registers the subclass and reads its templates
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "FRAME_CAPABILITIES" not in cls.__dict__:
            return
        cls.__gsttemplates__ = pad_templates(
            caps_for_frame_capabilities(cls.FRAME_CAPABILITIES)
        )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.set_in_place(True)
        self.set_passthrough(not self.FRAME_CAPABILITIES.reads_pixels)
        self.width = 0
        self.height = 0
        self.format = None
        self._bytes_per_pixel = 0

    def do_set_caps(self, incaps, outcaps):
        info = GstVideo.VideoInfo.new_from_caps(incaps)
        self.width = info.width
        self.height = info.height
        self.format = info.finfo.name
        self._bytes_per_pixel = info.finfo.pixel_stride[0]
        return True

    def do_transform_ip(self, buf):
        if not self.FRAME_CAPABILITIES.reads_pixels:
            return self._run(None, buf)
        if self.FRAME_CAPABILITIES.accepts_cuda and is_cuda_buffer(buf):
            return self._run_on_device_frame(buf)
        with mapped_frame(
            buf, Gst.MapFlags.WRITE, self.width, self.height, self._bytes_per_pixel
        ) as frame:
            if frame is None:
                self.logger.error("failed to map buffer for writing")
                return Gst.FlowReturn.ERROR
            return self._run(frame, buf)

    def _run_on_device_frame(self, buf):
        import backend

        try:
            frame = backend.frameio.read_frame_for_drawing(
                buf, self.width, self.height, self.format
            )
        except ValueError as exception:
            post_error(self, "cannot draw on the frame", exception)
            return Gst.FlowReturn.ERROR
        try:
            return self._run(frame, buf)
        finally:
            backend.frameio.release_frames(buf)

    def _run(self, frame, buf):
        try:
            self.process_in_place(frame, self.format, buf)
            return Gst.FlowReturn.OK
        except Exception as exception:
            post_error(self, "transform error", exception)
            return Gst.FlowReturn.ERROR
