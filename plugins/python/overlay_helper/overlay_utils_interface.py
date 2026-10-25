# overlay_utils_interface.py
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from abc import ABC, abstractmethod
from enum import Enum


class Color:
    def __init__(self, r, g, b, a=1.0):
        self.r = r
        self.g = g
        self.b = b
        self.a = a


BOX_COLOR = Color(1, 0, 0, 1)
BOX_LINE_WIDTH = 2
BOX_LABEL_FONT_SIZE = 12
BOX_LABEL_OFFSET = 10
TRACKING_POINT_SIZE = 10
CLASSIFICATION_PADDING = 10
CLASSIFICATION_LINE_HEIGHT = 22
CLASSIFICATION_FONT_SIZE = 14
CLASSIFICATION_LABEL_WIDTH = 220
CLASSIFICATION_BACKGROUND_MARGIN = 4
CLASSIFICATION_BACKGROUND_COLOR = Color(0, 0, 0, 0.6)
CLASSIFICATION_TEXT_COLOR = Color(0, 1, 0, 1)


class TrackingDisplay:
    def __init__(self, max_history_length=5000):
        self.history = []
        self.max_history_length = max_history_length
        self.id_color_map = {}
        self.color_palette = [
            Color(1.0, 0.0, 0.0, 1.0),  # Red
            Color(0.0, 1.0, 0.0, 1.0),  # Green
            Color(0.0, 0.0, 1.0, 1.0),  # Blue
            Color(1.0, 1.0, 0.0, 1.0),  # Yellow
            Color(1.0, 0.0, 1.0, 1.0),  # Magenta
            Color(0.0, 1.0, 1.0, 1.0),  # Cyan
            Color(1.0, 0.5, 0.0, 1.0),  # Orange
            Color(0.5, 0.0, 1.0, 1.0),  # Purple
            Color(0.5, 1.0, 0.0, 1.0),  # Lime
            Color(0.0, 0.5, 1.0, 1.0),  # Light Blue
            Color(1.0, 0.3, 0.3, 1.0),  # Light Red
            Color(0.3, 1.0, 0.3, 1.0),  # Light Green
            Color(0.3, 0.3, 1.0, 1.0),  # Light Blue
            Color(1.0, 1.0, 0.3, 1.0),  # Light Yellow
            Color(1.0, 0.3, 1.0, 1.0),  # Pink
            Color(0.3, 1.0, 1.0, 1.0),  # Aqua
            Color(0.5, 0.2, 0.0, 1.0),  # Brown
            Color(0.2, 0.5, 0.0, 1.0),  # Olive
            Color(0.5, 0.5, 0.5, 1.0),  # Grey
            Color(1.0, 0.6, 0.4, 1.0),  # Peach
        ]
        self.processed_ids = set()  # Track IDs that have already been counted
        self.total_top_to_bottom = 0  # Total objects crossing from top to bottom
        self.total_bottom_to_top = 0  # Total objects crossing from bottom to top
        self.y_line = None  # The y-coordinate of the horizontal line

    def get_color_for_id(self, track_id):
        if track_id not in self.id_color_map:
            color_index = len(self.id_color_map) % len(self.color_palette)
            self.id_color_map[track_id] = self.color_palette[color_index]
        return self.id_color_map[track_id]

    def add_tracking_point(self, center, track_id):
        color = self.get_color_for_id(track_id)
        self.history.append(
            {"center": center, "color": color, "track_id": track_id, "opacity": 1.0}
        )

        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length :]

    def fade_history(self):
        for point in self.history:
            point["opacity"] *= 0.9
        self.history = [point for point in self.history if point["opacity"] > 0.1]

    def set_y_line(self, y_line):
        """Set the y-coordinate of the horizontal line for calculations."""
        self.y_line = y_line

    def count_objects(self):
        """Count cars crossing the horizontal line set by `set_y_line`.

        Returns:
            tuple: (count_top_to_bottom, count_bottom_to_top)
        """
        if self.y_line is None:
            raise ValueError("The y_line must be set before counting crossings.")

        count_top_to_bottom = 0
        count_bottom_to_top = 0

        track_last_positions = {}

        for point in self.history:
            track_id = point["track_id"]
            if track_id in self.processed_ids:
                continue  # Skip tracks that have already been processed

            current_y = point["center"]["y"]

            if track_id in track_last_positions:
                last_y = track_last_positions[track_id]

                if last_y < self.y_line <= current_y:  # Crossed from top to bottom
                    count_top_to_bottom += 1
                    self.total_top_to_bottom += 1
                    self.processed_ids.add(track_id)
                elif last_y > self.y_line >= current_y:  # Crossed from bottom to top
                    count_bottom_to_top += 1
                    self.total_bottom_to_top += 1
                    self.processed_ids.add(track_id)

            track_last_positions[track_id] = current_y

        return self.total_top_to_bottom, self.total_bottom_to_top


class OverlayGraphics(ABC):
    @abstractmethod
    def initialize(self, buffer_data):
        """Initialize the graphics context and prepare for rendering."""
        pass

    @abstractmethod
    def draw_metadata(self, metadata, tracking_display):
        """Draw metadata and tracking points on the current frame."""
        pass

    @abstractmethod
    def finalize(self):
        """Finalize the rendering and clean up the context."""
        pass

    @abstractmethod
    def draw_bounding_box(self, box):
        """Draw a bounding box on the current frame."""
        pass

    @abstractmethod
    def draw_text(self, label, x, y, colour, font_size):
        """Draw a label at the specified position on the current frame."""
        pass

    @abstractmethod
    def draw_tracking_point(self, center, color, opacity):
        """Draw a tracking point with specified color and opacity."""
        pass

    @abstractmethod
    def draw_line(self, start, end, color, width):
        """Draw a line from start to end with specified color and width."""
        pass

    def draw_polyline(self, points, color, width):
        for start, end in zip(points, points[1:]):
            self.draw_line(start, end, color, width)


class GraphicsType(Enum):
    CAIRO = "cairo"
    SKIA = "skia"
    OPENGL = "opengl"
    VULKAN = "vulkan"
    TORCH = "torch"


DEFAULT_RENDERER = "cairo"
GRAPHICS_TYPE_BY_RENDERER = {
    "cairo": GraphicsType.CAIRO,
    "skia": GraphicsType.SKIA,
}


class OverlayGraphicsFactory:
    @staticmethod
    def create(graphics_type, width, height, **kwargs):
        """Factory method to create an OverlayGraphics object based on type."""
        if graphics_type == GraphicsType.CAIRO:
            from .overlay_cairo import CairoOverlayGraphics

            return CairoOverlayGraphics(width, height)
        elif graphics_type == GraphicsType.SKIA:
            from .overlay_skia import SkiaOverlayGraphics

            return SkiaOverlayGraphics(width, height, kwargs["video_format"])
        elif graphics_type == GraphicsType.TORCH:
            from .overlay_torch import TorchOverlayGraphics

            return TorchOverlayGraphics(width, height, kwargs["video_format"])
        elif graphics_type == GraphicsType.OPENGL:
            from .overlay_opengl import OpenGLOverlayGraphics

            return OpenGLOverlayGraphics(width, height)
        elif graphics_type == GraphicsType.VULKAN:
            from .overlay_vulkan import VulkanOverlayGraphics

            return VulkanOverlayGraphics(
                width, height, kwargs["video_format"], kwargs["vulkan_queue"]
            )
        else:
            raise ValueError(f"Unknown graphics type: {graphics_type}")
