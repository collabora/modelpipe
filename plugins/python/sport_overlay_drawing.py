# SportOverlayDrawing
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import math
from contextlib import contextmanager

CHANNEL_ORDER = {
    "RGBA": (0, 1, 2, 3),
    "ARGB": (3, 0, 1, 2),
    "BGRA": (2, 1, 0, 3),
    "ABGR": (3, 2, 1, 0),
}
FULL_INTENSITY = 255
COLOUR_CHANNEL_COUNT = 3

JERSEY_TOP_FRACTION = 0.15
JERSEY_BOTTOM_FRACTION = 0.55
JERSEY_LEFT_FRACTION = 0.25
JERSEY_RIGHT_FRACTION = 0.75
JERSEY_MIN_SIDE = 3
JERSEY_MIN_SATURATION = 80
JERSEY_MIN_VALUE = 50
# opencv 8-bit hue runs 0..179
RED_HUE_LOW_MAX = 10
RED_HUE_HIGH_MIN = 170
BLUE_HUE_MIN = 100
BLUE_HUE_MAX = 130
# referee kits skip grass green (40..90) and both team bands
YELLOW_HUE_MIN = 18
YELLOW_HUE_MAX = 34
PINK_HUE_MIN = 145
PINK_HUE_MAX = 165
JERSEY_VOTE_NAMES = ("red", "blue", "ref")
JERSEY_MIN_PIXELS = 20
JERSEY_MIN_PIXEL_FRACTION = 0.02

# opencv's rgb to hsv works in 12-bit fixed point
HSV_SHIFT = 12
HSV_ROUNDING = 1 << (HSV_SHIFT - 1)
HUE_RANGE = 180
HUE_SECTORS = 6

# cv2.getTextSize height per unit of hershey scale matches the pil font ascent
HERSHEY_SCALE_TO_FONT_PIXELS = 27


def channel_order(video_format):
    return CHANNEL_ORDER.get(video_format, CHANNEL_ORDER["RGBA"])


# drawings: (implementation, width, height, format) -> drawing, kept across frames
def drawing_for(drawings, frame, fmt):
    import numpy as np

    height, width = frame.shape[:2]
    if isinstance(frame, np.ndarray):
        key = (OpenCVSportDrawing, width, height, fmt)
        if key not in drawings:
            drawings[key] = OpenCVSportDrawing(fmt)
        return drawings[key]
    key = (TorchSportDrawing, width, height, fmt)
    if key not in drawings:
        drawings[key] = TorchSportDrawing(width, height, fmt)
    return drawings[key]


def jersey_window(box, frame_width, frame_height):
    x1, y1, x2, y2 = (int(value) for value in box)
    box_height, box_width = y2 - y1, x2 - x1
    if box_height <= 0 or box_width <= 0:
        return None
    top = max(0, y1 + int(JERSEY_TOP_FRACTION * box_height))
    bottom = min(frame_height, y1 + int(JERSEY_BOTTOM_FRACTION * box_height))
    left = max(0, x1 + int(JERSEY_LEFT_FRACTION * box_width))
    right = min(frame_width, x1 + int(JERSEY_RIGHT_FRACTION * box_width))
    if bottom - top < JERSEY_MIN_SIDE or right - left < JERSEY_MIN_SIDE:
        return None
    return top, bottom, left, right


def jersey_hue_masks(hue, saturation, value):
    coloured = (saturation > JERSEY_MIN_SATURATION) & (value > JERSEY_MIN_VALUE)
    red = ((hue <= RED_HUE_LOW_MAX) | (hue >= RED_HUE_HIGH_MIN)) & coloured
    blue = (hue >= BLUE_HUE_MIN) & (hue <= BLUE_HUE_MAX) & coloured
    yellow = (hue >= YELLOW_HUE_MIN) & (hue <= YELLOW_HUE_MAX)
    pink = (hue >= PINK_HUE_MIN) & (hue <= PINK_HUE_MAX)
    referee = (yellow | pink) & coloured
    return {"red": red, "blue": blue, "ref": referee}


def jersey_vote(counts, pixel_count):
    min_pixels = max(JERSEY_MIN_PIXELS, int(JERSEY_MIN_PIXEL_FRACTION * pixel_count))
    best = max(counts, key=counts.get)
    if counts[best] < min_pixels:
        return None
    return best


class OpenCVSportDrawing:
    def __init__(self, video_format):
        self.order = channel_order(video_format)
        self.inverse_order = [self.order.index(channel) for channel in range(4)]
        self.frame = None

    @contextmanager
    def drawing_on(self, frame):
        self.frame = frame
        try:
            yield self
        finally:
            self.frame = None

    def buffer_colour(self, rgba):
        return tuple(rgba[index] for index in self.order)

    def draw_ellipse_arc(self, center, axes, start_angle, end_angle, rgba, thickness):
        import cv2

        cv2.ellipse(
            self.frame,
            center,
            axes,
            0.0,
            start_angle,
            end_angle,
            self.buffer_colour(rgba),
            thickness,
            cv2.LINE_AA,
        )

    def draw_triangle(self, points, fill_rgba, outline_rgba, outline_thickness):
        import cv2
        import numpy as np

        contour = np.array(points, dtype=np.int32)
        cv2.drawContours(
            self.frame, [contour], 0, self.buffer_colour(fill_rgba), cv2.FILLED
        )
        cv2.drawContours(
            self.frame,
            [contour],
            0,
            self.buffer_colour(outline_rgba),
            outline_thickness,
        )

    def draw_polyline(self, points, rgba, thickness):
        import cv2
        import numpy as np

        contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            self.frame,
            [contour],
            False,
            self.buffer_colour(rgba),
            thickness,
            cv2.LINE_AA,
        )

    def draw_segments(self, segments, rgba, thickness):
        import cv2

        colour = self.buffer_colour(rgba)
        for start, end in segments:
            cv2.line(self.frame, start, end, colour, thickness, cv2.LINE_AA)

    def draw_text(self, text, origin, scale, rgba, thickness):
        import cv2

        cv2.putText(
            self.frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            self.buffer_colour(rgba),
            thickness,
            cv2.LINE_AA,
        )

    def fill_rectangle(self, top_left, bottom_right, rgba):
        import cv2

        cv2.rectangle(
            self.frame, top_left, bottom_right, self.buffer_colour(rgba), cv2.FILLED
        )

    def draw_rectangle(self, top_left, bottom_right, rgba, thickness):
        import cv2

        cv2.rectangle(
            self.frame, top_left, bottom_right, self.buffer_colour(rgba), thickness
        )

    def blit(self, image, left, top):
        frame_height, frame_width = self.frame.shape[:2]
        image_height, image_width = image.shape[:2]
        right = min(frame_width, left + image_width)
        bottom = min(frame_height, top + image_height)
        if right <= left or bottom <= top:
            return
        self.frame[top:bottom, left:right] = image[: bottom - top, : right - left]

    def classify_jersey(self, box):
        import cv2
        import numpy as np

        frame_height, frame_width = self.frame.shape[:2]
        window = jersey_window(box, frame_width, frame_height)
        if window is None:
            return None
        top, bottom, left, right = window
        rgb = np.ascontiguousarray(
            self.frame[top:bottom, left:right][:, :, self.inverse_order[:3]]
        )
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int32)
        masks = jersey_hue_masks(hsv[..., 0], hsv[..., 1], hsv[..., 2])
        counts = {name: int(mask.sum()) for name, mask in masks.items()}
        return jersey_vote(counts, rgb.shape[0] * rgb.shape[1])


class TorchSportDrawing:
    def __init__(self, width, height, video_format):
        from overlay_helper.overlay_torch import TorchOverlayGraphics

        self.graphics = TorchOverlayGraphics(width, height, video_format)
        order = channel_order(video_format)
        self.logical_rgb_channels = [order.index(channel) for channel in range(3)]
        self.frame = None
        self.jersey_masks = None
        self.uploaded_image = None
        self.uploaded_pixels = None
        # every draw_entries call scans a frame-sized table
        self.queued_shapes = []
        # antialiased text blends over a shape only when drawn in a later call
        self.queued_labels = []

    @contextmanager
    def drawing_on(self, frame):
        import torch

        # the producer wait was queued on the stream current at map time
        self.graphics.stream.wait_stream(torch.cuda.current_stream())
        with self.graphics.drawing_stream():
            self.graphics.initialize(frame)
            self.frame = frame
            try:
                yield self
                self.draw_queued()
            finally:
                self.queued_shapes = []
                self.queued_labels = []
                self.graphics.finalize()
                self.frame = None
                self.jersey_masks = None

    @staticmethod
    def colour(rgba):
        from overlay_helper.overlay_utils_interface import Color

        red, green, blue, alpha = (value / FULL_INTENSITY for value in rgba)
        return Color(red, green, blue, alpha)

    def draw_ellipse_arc(self, center, axes, start_angle, end_angle, rgba, thickness):
        import numpy as np

        center_x, center_y = center
        horizontal_axis, vertical_axis = axes
        sweep = math.radians(end_angle - start_angle)
        # one sample per pixel of arc along the longer axis
        sample_count = math.ceil(max(horizontal_axis, vertical_axis) * sweep) + 1
        angles = np.linspace(
            math.radians(start_angle), math.radians(end_angle), sample_count
        )
        colour = self.colour(rgba)
        rows = np.column_stack(
            [
                center_x + horizontal_axis * np.cos(angles),
                center_y + vertical_axis * np.sin(angles),
                np.tile((colour.r, colour.g, colour.b, colour.a), (sample_count, 1)),
            ]
        )
        self.queued_shapes.append(self.graphics.square_entries(rows, thickness))

    def queue_segments(self, segments, thickness):
        from overlay_helper.overlay_torch import segment_rows

        self.queued_shapes.append(
            self.graphics.square_entries(segment_rows(segments), thickness)
        )

    def draw_queued(self):
        from overlay_helper.overlay_torch import concatenated_entries

        for queue in (self.queued_shapes, self.queued_labels):
            if queue:
                self.graphics.draw_entries(concatenated_entries(queue))
        self.queued_shapes = []
        self.queued_labels = []

    def draw_triangle(self, points, fill_rgba, outline_rgba, outline_thickness):
        import numpy as np
        from overlay_helper.overlay_torch import PixelEntries

        corners = np.array(points, dtype=np.float32)
        left, top = np.floor(corners.min(0)).astype(np.int64)
        right, bottom = np.ceil(corners.max(0)).astype(np.int64)
        lines, columns = np.meshgrid(
            np.arange(top, bottom + 1), np.arange(left, right + 1), indexing="ij"
        )
        lines, columns = lines.reshape(-1), columns.reshape(-1)
        inside = self.inside_triangle(
            corners, columns.astype(np.float32), lines.astype(np.float32)
        )
        fill_pixel_count = int(inside.sum())
        fill = np.array(
            self.graphics.channel_values(self.colour(fill_rgba)), np.float32
        )
        self.queued_shapes.append(
            PixelEntries(
                lines[inside],
                columns[inside],
                np.broadcast_to(fill, (fill_pixel_count, COLOUR_CHANNEL_COUNT)),
                np.ones(fill_pixel_count, np.float32),
            )
        )
        outline = self.colour(outline_rgba)
        vertices = [{"x": x, "y": y} for x, y in points]
        self.queue_segments(
            [
                (start, end, outline)
                for start, end in zip(vertices, vertices[1:] + vertices[:1])
            ],
            outline_thickness,
        )

    @staticmethod
    def inside_triangle(corners, columns, lines):
        import numpy as np

        sides = []
        for index in range(3):
            start_x, start_y = corners[index]
            end_x, end_y = corners[(index + 1) % 3]
            sides.append(
                (end_x - start_x) * (lines - start_y)
                - (end_y - start_y) * (columns - start_x)
            )
        sides = np.stack(sides)
        return (sides >= 0).all(0) | (sides <= 0).all(0)

    def draw_polyline(self, points, rgba, thickness):
        from overlay_helper.overlay_torch import polyline_segments

        self.queue_segments(
            polyline_segments([{"x": x, "y": y} for x, y in points], self.colour(rgba)),
            thickness,
        )

    def draw_segments(self, segments, rgba, thickness):
        colour = self.colour(rgba)
        self.queue_segments(
            [
                ({"x": start_x, "y": start_y}, {"x": end_x, "y": end_y}, colour)
                for (start_x, start_y), (end_x, end_y) in segments
            ],
            thickness,
        )

    def draw_text(self, text, origin, scale, rgba, thickness):
        x, y = origin
        font_size = round(scale * HERSHEY_SCALE_TO_FONT_PIXELS)
        self.queued_labels.append(
            self.graphics.text_entries([(text, x, y)], self.colour(rgba), font_size, 0)
        )

    def fill_rectangle(self, top_left, bottom_right, rgba):
        self.draw_queued()
        left, top = top_left
        right, bottom = bottom_right
        # opencv includes the bottom right corner
        self.graphics.blend_rectangle(
            left, top, right + 1, bottom + 1, self.colour(rgba)
        )

    def draw_rectangle(self, top_left, bottom_right, rgba, thickness):
        left, top = top_left
        right, bottom = bottom_right
        corners = [
            {"x": left, "y": top},
            {"x": right, "y": top},
            {"x": right, "y": bottom},
            {"x": left, "y": bottom},
        ]
        colour = self.colour(rgba)
        self.queue_segments(
            [
                (start, end, colour)
                for start, end in zip(corners, corners[1:] + corners[:1])
            ],
            thickness,
        )

    def blit(self, image, left, top):
        import torch

        self.draw_queued()
        if image is not self.uploaded_image:
            self.uploaded_pixels = torch.from_numpy(image).to(self.frame.device)
            self.uploaded_image = image
        frame_height, frame_width = self.frame.shape[:2]
        image_height, image_width = image.shape[:2]
        right = min(frame_width, left + image_width)
        bottom = min(frame_height, top + image_height)
        if right <= left or bottom <= top:
            return
        self.frame[top:bottom, left:right] = self.uploaded_pixels[
            : bottom - top, : right - left
        ]

    def classify_jersey(self, box):
        frame_height, frame_width = self.frame.shape[:2]
        window = jersey_window(box, frame_width, frame_height)
        if window is None:
            return None
        top, bottom, left, right = window
        window_masks = self.whole_frame_jersey_masks()[:, top:bottom, left:right]
        totals = window_masks.sum((1, 2)).tolist()
        counts = dict(zip(JERSEY_VOTE_NAMES, totals))
        return jersey_vote(counts, (bottom - top) * (right - left))

    # one pass over the frame costs less than one per box
    def whole_frame_jersey_masks(self):
        import torch

        if self.jersey_masks is None:
            self.draw_queued()
            rgb = self.frame[..., self.logical_rgb_channels]
            masks = jersey_hue_masks(*opencv_hsv(rgb))
            self.jersey_masks = torch.stack([masks[name] for name in JERSEY_VOTE_NAMES])
        return self.jersey_masks


def opencv_hsv(rgb):
    import torch

    red, green, blue = rgb.to(torch.int32).unbind(-1)
    value = torch.maximum(torch.maximum(red, green), blue)
    difference = value - torch.minimum(torch.minimum(red, green), blue)
    saturation_divisor = torch.where(
        value > 0,
        ((FULL_INTENSITY << HSV_SHIFT) / value.clamp(min=1)).round(),
        0,
    ).to(torch.int32)
    saturation = (difference * saturation_divisor + HSV_ROUNDING) >> HSV_SHIFT
    hue_divisor = torch.where(
        difference > 0,
        ((HUE_RANGE << HSV_SHIFT) / (HUE_SECTORS * difference.clamp(min=1))).round(),
        0,
    ).to(torch.int32)
    hue = torch.where(
        value == red,
        green - blue,
        torch.where(
            value == green, blue - red + 2 * difference, red - green + 4 * difference
        ),
    )
    hue = (hue * hue_divisor + HSV_ROUNDING) >> HSV_SHIFT
    hue = torch.where(hue < 0, hue + HUE_RANGE, hue)
    return hue, saturation, value
