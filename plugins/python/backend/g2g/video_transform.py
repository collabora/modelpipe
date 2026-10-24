# VideoTransform (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""g2g backend for video transform elements.

The host calls `g2g_process(buf, w, h, fmt, sink)` once per frame. This base
binds the frame's sink onto the shared `frameio` / `analytics` (so leaf task code
that calls them through `from backend import frameio, analytics` reaches this
frame's buffer and sink), applies `only-on` to the upstream metadata the sink
carries, extracts the frame, then hands off to `process_frames`,
the framework-agnostic per-frame hook the leaf element supplies (the same hook the
gst backend drives from `do_transform_ip`).
"""

from backend.g2g.aggregator import FrameBatchProcessingMixin
from backend.g2g.analytics import analytics
from backend.g2g.cuda_frames import rgb_device_frame
from backend.g2g.frameio import DEVICE_FRAME, as_rgb, frameio
from backend.core import FrameProcessingMixin
from backend.g2g.transform import BaseTransform
from backend.video_frame import SYSTEM_INFERENCE_FRAME_CAPABILITIES


DEVICE_FRAME_FORMAT = "RGB"


class VideoTransform(BaseTransform, FrameProcessingMixin, FrameBatchProcessingMixin):
    """Base for g2g video transform elements."""

    FRAME_CAPABILITIES = SYSTEM_INFERENCE_FRAME_CAPABILITIES

    def __init__(self):
        super().__init__()
        self.width = 0
        self.height = 0

    def g2g_process(self, buf, width, height, fmt, sink):
        self.width = width
        self.height = height
        # Route this frame's pixels (buf) and metadata sink to the shared I/O the
        # leaf task code uses.
        frameio.bind(sink, fmt)
        analytics.bind(sink)
        if self._only_on and not self.carries_only_on(buf):
            return None
        self._ensure_model()
        self._ensure_started()
        frames, num_sources, fmt = frameio.read_frames(buf, None, width, height)
        if frames is None:
            return None
        self.process_frames(as_rgb(frames, fmt), num_sources, fmt, buf)
        # Blobs/detections are staged on the sink; no return payload needed.
        return None

    def g2g_process_cuda(self, luma, chroma, width, height, yuv_conversion, sink):
        import torch

        if not self.FRAME_CAPABILITIES.accepts_cuda:
            raise ValueError(f"{type(self).__name__} reads only system memory")
        self.width = width
        self.height = height
        frameio.bind(sink, DEVICE_FRAME_FORMAT)
        analytics.bind(sink)
        if self._only_on and not self.carries_only_on(DEVICE_FRAME):
            return None
        self._ensure_model()
        self._ensure_started()
        with self.engine.stream_context():
            frame = rgb_device_frame(luma, chroma, yuv_conversion)
            if self.runs_exported_model:
                frame = frame.cpu().numpy()
            self.process_frames(frame, 1, DEVICE_FRAME_FORMAT, DEVICE_FRAME)
            # downstream may unmap the decoder's surface once this returns
            torch.cuda.current_stream().synchronize()
        return None
