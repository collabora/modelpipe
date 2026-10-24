# OverlayCounter
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import backend
from backend import GObject, InPlaceVideoTransform, CUDA_DRAW_FRAME_CAPABILITIES
from overlay_helper.overlay_utils import draw_frame_objects, load_metadata
from overlay_helper.overlay_utils_interface import (
    DEFAULT_RENDERER,
    GRAPHICS_TYPE_BY_RENDERER,
    Color,
    GraphicsType,
    OverlayGraphicsFactory,
    TrackingDisplay,
)
from utils.analytics_utils import AnalyticsUtils

COUNTER_COLOR = Color(1.0, 0.0, 0.0, 1.0)
COUNTING_LINE_WIDTH = 5
COUNTER_TEXT_X = 0
COUNTER_TEXT_Y = 50
COUNTER_TEXT_FONT_SIZE = 20


class OverlayCounter(InPlaceVideoTransform):
    __gstmetadata__ = (
        "OverlayCounter",
        "Filter/Effect/Video",
        "Counts objects and displays count",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    FRAME_CAPABILITIES = CUDA_DRAW_FRAME_CAPABILITIES

    meta_path = GObject.Property(
        type=str,
        default=None,
        nick="Metadata File Path",
        blurb="Path to the JSON file containing frame metadata with bounding boxes and tracking data",
        flags=GObject.ParamFlags.READWRITE,
    )
    tracking = GObject.Property(
        type=bool,
        default=True,
        nick="Enable Tracking Display",
        blurb="Enable or disable tracking display",
        flags=GObject.ParamFlags.READWRITE,
    )

    @GObject.Property(
        type=str,
        default=DEFAULT_RENDERER,
        nick="Renderer",
        blurb=f"Renderer for system memory frames: {', '.join(GRAPHICS_TYPE_BY_RENDERER)}",
    )
    def renderer(self):
        return self.selected_renderer

    @renderer.setter
    def renderer(self, value):
        if value not in GRAPHICS_TYPE_BY_RENDERER:
            raise ValueError(
                f"Unknown renderer {value}, allowed: {', '.join(GRAPHICS_TYPE_BY_RENDERER)}"
            )
        self.selected_renderer = value

    def __init__(self):
        super().__init__()
        self.selected_renderer = DEFAULT_RENDERER
        self.analytics_utils = AnalyticsUtils()
        self.tracking_display = TrackingDisplay()
        self.metadata_by_frame_index = None
        self.frame_counter = 0
        self.overlay_graphics = None
        self.overlay_graphics_key = None

    def frame_metadata(self, buf):
        if not self.meta_path:
            return self.analytics_utils.extract_analytics_metadata(buf)
        if self.metadata_by_frame_index is None:
            self.metadata_by_frame_index = load_metadata(self.meta_path, self.logger)
        return self.metadata_by_frame_index.get(self.frame_counter, [])

    def overlay_graphics_for(self, graphics_type, fmt):
        key = (graphics_type, self.width, self.height, fmt)
        if key != self.overlay_graphics_key:
            self.overlay_graphics = OverlayGraphicsFactory.create(
                graphics_type, self.width, self.height, video_format=fmt
            )
            self.overlay_graphics_key = key
        return self.overlay_graphics

    def process_in_place(self, frame, fmt, buf):
        import numpy as np

        frame_metadata = self.frame_metadata(buf)
        self.frame_counter += 1
        if isinstance(frame, np.ndarray):
            graphics_type = GRAPHICS_TYPE_BY_RENDERER[self.selected_renderer]
            self.draw(
                self.overlay_graphics_for(graphics_type, fmt), frame, frame_metadata
            )
            return
        import torch

        overlay_graphics = self.overlay_graphics_for(GraphicsType.TORCH, fmt)
        # the producer wait was queued on the stream current at map time
        overlay_graphics.stream.wait_stream(torch.cuda.current_stream())
        with overlay_graphics.drawing_stream():
            self.draw(overlay_graphics, frame, frame_metadata)

    def draw(self, overlay_graphics, frame, frame_metadata):
        overlay_graphics.initialize(frame)
        draw_frame_objects(
            overlay_graphics,
            frame_metadata,
            self.tracking_display if self.tracking else None,
        )
        line_y = self.height / 2
        self.tracking_display.set_y_line(line_y)
        overlay_graphics.draw_line(
            start={"x": 0, "y": line_y},
            end={"x": self.width, "y": line_y},
            color=COUNTER_COLOR,
            width=COUNTING_LINE_WIDTH,
        )
        approaching, receding = self.tracking_display.count_objects()
        overlay_graphics.draw_text(
            f"Objects receding : {receding}, approaching : {approaching}",
            COUNTER_TEXT_X,
            COUNTER_TEXT_Y,
            COUNTER_COLOR,
            COUNTER_TEXT_FONT_SIZE,
        )
        overlay_graphics.finalize()


if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_overlay_counter", OverlayCounter
    )
