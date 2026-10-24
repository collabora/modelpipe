# Pluggable element backend
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Pluggable multimedia-framework backend for the ML elements.

The ML logic (engines, tasks, base classes) is framework-agnostic. The pieces
that DO depend on the framework are isolated behind this package:

  * the element base classes `BaseTransform` / `BaseAggregator` / `VideoTransform`
  * the `analytics` metadata interface (add/get/remove detections)

Select the active backend with the `MP_BACKEND` environment variable
(default ``"gst"``). To add another backend, create a sibling package exposing
the same names (reusing `backend.core.MLEngineMixin` and implementing
`backend.analytics.AnalyticsBackend`) and add a branch below.
"""

from backend.analytics import AnalyticsBackend  # noqa: F401  (re-exported)
from backend.frameio import FrameIO  # noqa: F401  (re-exported)
from backend.core import MLEngineMixin  # noqa: F401  (re-exported)
from backend.video_frame import (  # noqa: F401  (re-exported)
    VideoFrameCapabilities,
    SYSTEM_INFERENCE_FRAME_CAPABILITIES,
    CUDA_INFERENCE_FRAME_CAPABILITIES,
    SYSTEM_READ_FRAME_CAPABILITIES,
    SYSTEM_DRAW_FRAME_CAPABILITIES,
    CUDA_DRAW_FRAME_CAPABILITIES,
    CUDA_RGBA_DRAW_FRAME_CAPABILITIES,
    METADATA_FRAME_CAPABILITIES,
)

from utils.selected_backend import BACKEND

if BACKEND == "gst":
    from backend.gst import (
        BaseTransform,
        BaseAggregator,
        VideoTransform,
        InPlaceVideoTransform,
        InPlaceAudioTransform,
        BaseSink,
        analytics,
        frameio,
        FlowReturn,
        GObject,
        post_error,
        post_model_load_error,
    )
elif BACKEND == "g2g":
    from backend.g2g import (
        BaseTransform,
        BaseAggregator,
        VideoTransform,
        InPlaceVideoTransform,
        InPlaceAudioTransform,
        BaseSink,
        analytics,
        frameio,
        FlowReturn,
        GObject,
        post_error,
        post_model_load_error,
    )
else:
    raise ImportError(
        f"Unknown MP_BACKEND={BACKEND!r}; supported backends: 'gst', 'g2g'"
    )


def register_gst_element(name, cls, type_name=None):
    """Register `cls` with the GObject type system and return its factory tuple.

    Leaf elements call this behind a `BACKEND == "gst"` guard: under g2g the
    element base classes are plain Python objects, so type_register would fail.
    """
    from backend.gst import Gst, GObject

    GObject.type_register(cls, type_name)
    return (name, Gst.Rank.NONE, cls)


__all__ = [
    "BACKEND",
    "register_gst_element",
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
    "AnalyticsBackend",
    "FrameIO",
    "MLEngineMixin",
    "VideoFrameCapabilities",
    "SYSTEM_INFERENCE_FRAME_CAPABILITIES",
    "CUDA_INFERENCE_FRAME_CAPABILITIES",
    "SYSTEM_READ_FRAME_CAPABILITIES",
    "SYSTEM_DRAW_FRAME_CAPABILITIES",
    "CUDA_DRAW_FRAME_CAPABILITIES",
    "CUDA_RGBA_DRAW_FRAME_CAPABILITIES",
    "METADATA_FRAME_CAPABILITIES",
]
