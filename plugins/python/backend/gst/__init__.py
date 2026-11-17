# GStreamer backend
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""GStreamer backend: element bases plus the analytics metadata implementation.

Element registration metadata (`__gstmetadata__`, `__gsttemplates__`, and the
`__gstelementfactory__` / `GObject.type_register` calls in each leaf plugin) is
GStreamer-specific and stays in this backend.
"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject  # noqa: E402

# gstreamer segfaults on element type registration before Gst.init
Gst.init(None)

from backend.gst.transform import BaseTransform  # noqa: E402
from backend.gst.aggregator import BaseAggregator  # noqa: E402
from backend.gst.video_transform import VideoTransform  # noqa: E402
from backend.gst.in_place_video_transform import InPlaceVideoTransform  # noqa: E402
from backend.gst.in_place_audio_transform import InPlaceAudioTransform  # noqa: E402
from backend.gst.sink import BaseSink  # noqa: E402
from backend.gst.analytics import select_analytics  # noqa: E402
from backend.gst.frameio import GstFrameIO  # noqa: E402
from backend.gst.errors import post_error, post_model_load_error  # noqa: E402,F401

#: The analytics metadata implementation for this backend.
analytics = select_analytics()

#: The frame buffer I/O implementation for this backend.
frameio = GstFrameIO()

# Framework primitives exposed to leaf elements so their task code imports them
# from the backend rather than touching `gi` directly. `FlowReturn` is the
# process()/transform return type; `GObject` carries the property declarations.
FlowReturn = Gst.FlowReturn

__all__ = [
    "BaseTransform",
    "BaseAggregator",
    "VideoTransform",
    "InPlaceVideoTransform",
    "InPlaceAudioTransform",
    "BaseSink",
    "analytics",
    "frameio",
    "FlowReturn",
    "GObject",
    "post_error",
    "post_model_load_error",
]
