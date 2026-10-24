# VideoTransform (GStreamer backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstBase  # noqa: E402

from backend.core import FrameProcessingMixin  # noqa: E402
from backend.gst.errors import post_error  # noqa: E402
from backend.gst.transform import BaseTransform  # noqa: E402
from backend.video_frame import SYSTEM_INFERENCE_FRAME_CAPABILITIES  # noqa: E402

# GST_BASE_TRANSFORM_FLOW_DROPPED, which pygobject does not expose
FLOW_DROPPED = Gst.FlowReturn.CUSTOM_SUCCESS

VIDEO_FIELDS = (
    "format=(string){ RGB, RGBA, ARGB, BGRA, ABGR },"
    "width=(int)[1,2147483647],height=(int)[1,2147483647]"
)
SYSTEM_MEMORY_CAPS = f"video/x-raw,{VIDEO_FIELDS}"
CUDA_MEMORY_FEATURE = "memory:CUDAMemory"
CUDA_MEMORY_CAPS = f"video/x-raw({CUDA_MEMORY_FEATURE}),{VIDEO_FIELDS}"


def frame_caps(capabilities):
    formats = ", ".join(capabilities.pixel_formats)
    fields = (
        f"format=(string){{ {formats} }},"
        "width=(int)[1,2147483647],height=(int)[1,2147483647]"
    )
    system_caps = f"video/x-raw,{fields}"
    if not capabilities.accepts_cuda:
        return Gst.Caps.from_string(system_caps)
    return Gst.Caps.from_string(
        f"video/x-raw({CUDA_MEMORY_FEATURE}),{fields};{system_caps}"
    )


def pad_templates(caps):
    return (
        Gst.PadTemplate.new("src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, caps),
        Gst.PadTemplate.new(
            "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, caps
        ),
    )


def caps_without_cuda_memory(caps):
    system_caps = Gst.Caps.new_empty()
    for index in range(caps.get_size()):
        features = caps.get_features(index)
        if features.contains(CUDA_MEMORY_FEATURE):
            continue
        system_caps.append_structure_full(
            caps.get_structure(index).copy(), features.copy()
        )
    return system_caps


class VideoTransform(BaseTransform, FrameProcessingMixin):
    """
    GStreamer element for video transformation using a PyTorch model.
    """

    VIDEO_CAPS = Gst.Caps.from_string(SYSTEM_MEMORY_CAPS)
    __gsttemplates__ = pad_templates(VIDEO_CAPS)
    FRAME_CAPABILITIES = SYSTEM_INFERENCE_FRAME_CAPABILITIES

    # runs before pygobject registers the subclass and reads its templates
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "FRAME_CAPABILITIES" not in cls.__dict__:
            return
        cls.__gsttemplates__ = pad_templates(frame_caps(cls.FRAME_CAPABILITIES))

    def __init__(self):
        super().__init__()
        self.set_qos_enabled(True)
        self._sink_running_time_ns = None
        self._sink_clock_then_ns = None
        self._last_run_ns = 0

    def do_src_event(self, event):
        if event.type == Gst.EventType.QOS:
            _, _, diff, timestamp = event.parse_qos()
            clock = self.get_clock()
            if diff > 0 and clock is not None:
                self._sink_running_time_ns = timestamp + diff
                self._sink_clock_then_ns = clock.get_time()
        return GstBase.BaseTransform.do_src_event(self, event)

    def do_sink_event(self, event):
        if event.type == Gst.EventType.FLUSH_STOP:
            self._sink_running_time_ns = None
        return GstBase.BaseTransform.do_sink_event(self, event)

    def would_reach_the_sink_late(self, buf):
        if self._sink_running_time_ns is None or buf.pts == Gst.CLOCK_TIME_NONE:
            return False
        running_time = self.segment.to_running_time(Gst.Format.TIME, buf.pts)
        if running_time == Gst.CLOCK_TIME_NONE:
            return False
        elapsed_ns = self.get_clock().get_time() - self._sink_clock_then_ns
        due_by = self._sink_running_time_ns + elapsed_ns + self._last_run_ns
        return running_time < due_by

    def do_transform_caps(self, direction, caps, filter_caps):
        # the identity transform, the gi chain-up refuses a None filter
        transformed = (
            caps
            if filter_caps is None
            else filter_caps.intersect_full(caps, Gst.CapsIntersectMode.FIRST)
        )
        # an exported model reads host arrays only
        if not self.runs_exported_model:
            return transformed
        return caps_without_cuda_memory(transformed)

    def do_set_caps(self, incaps, outcaps):
        struct = incaps.get_structure(0)
        self.width = struct.get_int("width").value
        self.height = struct.get_int("height").value

        return True

    def do_transform_ip(self, buf):
        """GStreamer per-frame driver: extract the frame(s) through the backend
        frame I/O, run the element's `process_frames`, and map the outcome to a
        `Gst.FlowReturn`. Elements supply `process_frames`, not this."""
        # Imported lazily: the frameio singleton lives in backend.gst, which is
        # still being constructed when this module is imported.
        from backend import frameio

        if self._only_on and not self.carries_only_on(buf):
            return Gst.FlowReturn.OK
        if self.is_qos_enabled() and self.would_reach_the_sink_late(buf):
            return FLOW_DROPPED
        try:
            # a device frame's producer wait is queued on the current stream
            with self.engine.stream_context():
                frames, num_sources, fmt = frameio.read_frames(
                    buf,
                    self.sinkpad,
                    self.width,
                    self.height,
                    (
                        getattr(self, "framerate_num", 30),
                        getattr(self, "framerate_denom", 1),
                    ),
                )
                if frames is None:
                    self.logger.error("Failed to extract frames")
                    return Gst.FlowReturn.ERROR
                started = time.monotonic_ns()
                self.process_frames(frames, num_sources, fmt, buf)
                self._last_run_ns = time.monotonic_ns() - started
                return Gst.FlowReturn.OK
        except Exception as exception:
            post_error(self, "transform error", exception)
            return Gst.FlowReturn.ERROR
        finally:
            frameio.release_frames(buf)
