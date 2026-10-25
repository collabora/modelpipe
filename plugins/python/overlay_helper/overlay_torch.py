# overlay_torch.py
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import math
from typing import NamedTuple

from utils.format_converter import packed_format_layout

from .overlay_utils_interface import (
    BOX_COLOR,
    BOX_LABEL_FONT_SIZE,
    BOX_LABEL_OFFSET,
    BOX_LINE_WIDTH,
    CLASSIFICATION_BACKGROUND_COLOR,
    CLASSIFICATION_BACKGROUND_MARGIN,
    CLASSIFICATION_FONT_SIZE,
    CLASSIFICATION_LABEL_WIDTH,
    CLASSIFICATION_LINE_HEIGHT,
    CLASSIFICATION_PADDING,
    CLASSIFICATION_TEXT_COLOR,
    TRACKING_POINT_SIZE,
    OverlayGraphics,
)

COLOUR_CHANNEL_COUNT = 3
FULL_INTENSITY = 255
# a box outline is four filled slabs
SLABS_PER_BOX = 4
# the 2d prefix sum of these corner marks is one inside the rectangle
CORNER_SIGNS = (1, -1, -1, 1)


class TorchOverlayGraphics(OverlayGraphics):
    def __init__(self, width, height, video_format):
        import torch

        self.width = width
        self.height = height
        channel_count, rgb_slice = packed_format_layout(video_format)
        rgb_channels = range(channel_count)[rgb_slice[-1]]
        first_channel = min(rgb_channels)
        self.colour_channels = slice(
            first_channel, first_channel + COLOUR_CHANNEL_COUNT
        )
        self.reverses_colour = rgb_channels.step < 0
        self.stream = torch.cuda.Stream()
        self.glyph_atlases = {}
        self.device_colours = {}
        self.pixels = None

    def drawing_stream(self):
        import torch

        return torch.cuda.stream(self.stream)

    def initialize(self, buffer_data):
        self.pixels = buffer_data[..., self.colour_channels]

    def finalize(self):
        # downstream waits on the producer's stream, never on this one
        self.stream.synchronize()
        self.pixels = None

    def draw_metadata(self, metadata, tracking_display):
        if tracking_display and tracking_display.history:
            self.draw_squares(
                [
                    self.square_row(point["center"], point["color"], point["opacity"])
                    for point in tracking_display.history
                ],
                TRACKING_POINT_SIZE,
            )

        classifications = [d for d in metadata if d.get("type") == "classification"]
        for index, item in enumerate(classifications):
            self.draw_classification_label(index, item)

        detections = [d for d in metadata if d.get("type") != "classification"]
        if not detections:
            return
        self.draw_bounding_boxes([data["box"] for data in detections])
        self.draw_texts(
            [
                (data.get("label", ""), data["box"]["x1"], data["box"]["y1"])
                for data in detections
            ],
            BOX_COLOR,
            BOX_LABEL_FONT_SIZE,
            BOX_LABEL_OFFSET,
        )
        if not tracking_display:
            return
        for data in detections:
            track_id = data.get("track_id")
            if track_id is None:
                continue
            box = data["box"]
            center = {
                "x": (box["x1"] + box["x2"]) / 2,
                "y": (box["y1"] + box["y2"]) / 2,
            }
            tracking_display.add_tracking_point(center, track_id)

    def draw_classification_label(self, index, item):
        x = CLASSIFICATION_PADDING
        y = CLASSIFICATION_PADDING + (index + 1) * CLASSIFICATION_LINE_HEIGHT
        left = x - CLASSIFICATION_BACKGROUND_MARGIN
        top = y - CLASSIFICATION_LINE_HEIGHT + CLASSIFICATION_BACKGROUND_MARGIN
        self.blend_rectangle(
            left,
            top,
            left + CLASSIFICATION_LABEL_WIDTH,
            top + CLASSIFICATION_LINE_HEIGHT,
            CLASSIFICATION_BACKGROUND_COLOR,
        )
        text = f"{item['label']}: {item['confidence']:.1%}"
        self.draw_text(text, x, y, CLASSIFICATION_TEXT_COLOR, CLASSIFICATION_FONT_SIZE)

    def draw_bounding_box(self, box):
        self.draw_bounding_boxes([box])

    def draw_bounding_boxes(self, boxes):
        import numpy as np
        import torch

        corners = np.array(
            [(box["x1"], box["y1"], box["x2"], box["y2"]) for box in boxes]
        )
        half_width = BOX_LINE_WIDTH // 2
        left, top = (corners[:, :2].round().astype(np.int64) - half_width).T
        right, bottom = (corners[:, 2:].round().astype(np.int64) + half_width).T
        slab_tops = np.concatenate([top, bottom - BOX_LINE_WIDTH, top, top])
        slab_bottoms = np.concatenate([top + BOX_LINE_WIDTH, bottom, bottom, bottom])
        slab_lefts = np.concatenate([left, left, left, right - BOX_LINE_WIDTH])
        slab_rights = np.concatenate([right, right, left + BOX_LINE_WIDTH, right])
        slab_tops = slab_tops.clip(0, self.height)
        slab_bottoms = np.maximum(slab_bottoms.clip(0, self.height), slab_tops)
        slab_lefts = slab_lefts.clip(0, self.width)
        slab_rights = np.maximum(slab_rights.clip(0, self.width), slab_lefts)
        rows = np.concatenate([slab_tops, slab_tops, slab_bottoms, slab_bottoms])
        columns = np.concatenate([slab_lefts, slab_rights, slab_lefts, slab_rights])
        signs = np.repeat(CORNER_SIGNS, len(boxes) * SLABS_PER_BOX)
        table = torch.from_numpy(np.stack([rows, columns, signs])).to(
            self.pixels.device
        )
        corner_marks = torch.zeros(
            self.height + 1, self.width + 1, dtype=torch.int32, device=table.device
        )
        corner_marks.index_put_(
            (table[0], table[1]), table[2].to(torch.int32), accumulate=True
        )
        coverage = corner_marks.cumsum(0, dtype=torch.int32).cumsum(
            1, dtype=torch.int32
        )
        covered = coverage[: self.height, : self.width, None] > 0
        colour = self.device_colour(BOX_COLOR).to(torch.uint8)
        self.pixels.copy_(torch.where(covered, colour, self.pixels))

    def draw_text(self, label, x, y, color, font_size):
        self.draw_texts([(label, x, y)], color, font_size, 0)

    def draw_texts(self, labels, color, font_size, baseline_offset):
        if font_size not in self.glyph_atlases:
            self.glyph_atlases[font_size] = GlyphAtlas(font_size, self.pixels.device)
        atlas = self.glyph_atlases[font_size]
        placed_glyphs = []
        for text, x, y in labels:
            line = round(y) - baseline_offset - atlas.ascent
            column = x
            for character in text:
                glyph = atlas.glyph(character)
                placed_glyphs.append((glyph, line, round(column)))
                column += atlas.advances[glyph]
        if not placed_glyphs:
            return
        inked = atlas.inked_pixels(placed_glyphs)
        colours = self.device_colour(color).expand(
            len(inked.alphas), COLOUR_CHANNEL_COUNT
        )
        self.draw_pixels(inked.lines, inked.columns, colours, inked.alphas)

    def draw_tracking_point(self, center, color, opacity):
        self.draw_squares(
            [self.square_row(center, color, opacity)], TRACKING_POINT_SIZE
        )

    def draw_line(self, start, end, color, width):
        self.draw_segments([(start, end, color)], width)

    def draw_polyline(self, points, color, width):
        pairs = list(zip(points, points[1:])) or [(points[0], points[0])]
        self.draw_segments([(start, end, color) for start, end in pairs], width)

    def draw_segments(self, segments, width):
        import numpy as np

        rows = []
        for start, end, color in segments:
            first = np.array((start["x"], start["y"]), float)
            last = np.array((end["x"], end["y"]), float)
            steps = max(1, math.ceil(np.abs(last - first).max()))
            fractions = np.arange(0, steps + 1)[:, None] / steps
            rows.extend(
                self.square_row({"x": x, "y": y}, color, color.a)
                for x, y in first + (last - first) * fractions
            )
        self.draw_squares(rows, width)

    def square_row(self, center, color, alpha):
        return (center["x"], center["y"], color.r, color.g, color.b, alpha)

    def draw_squares(self, rows, size):
        import numpy as np
        import torch

        if not rows:
            return
        device = self.pixels.device
        table = torch.from_numpy(np.array(rows, dtype=np.float32)).to(device)
        centre_columns, centre_lines, colours, alphas = table.split(
            [1, 1, COLOUR_CHANNEL_COUNT, 1], dim=1
        )
        channel_values = colours * FULL_INTENSITY
        if self.reverses_colour:
            channel_values = channel_values.flip(1)
        offsets = torch.arange(size, device=device) - size // 2
        columns = centre_columns.round().long()[:, :, None] + offsets[None, None, :]
        lines = centre_lines.round().long()[:, :, None] + offsets[None, :, None]
        columns, lines = torch.broadcast_tensors(columns, lines)
        pixels_per_square = size * size
        self.draw_pixels(
            lines.reshape(-1),
            columns.reshape(-1),
            channel_values.repeat_interleave(pixels_per_square, dim=0),
            alphas.repeat_interleave(pixels_per_square),
        )

    # only the last entry over a pixel is blended
    def draw_pixels(self, lines, columns, colours, alphas):
        import torch

        inside = (
            (columns >= 0)
            & (columns < self.width)
            & (lines >= 0)
            & (lines < self.height)
        )
        lines = lines.clamp(0, self.height - 1)
        columns = columns.clamp(0, self.width - 1)
        pixel_index = lines * self.width + columns
        # zero marks a pixel no entry inside the frame covers
        entry_number = torch.arange(1, len(lines) + 1, device=lines.device) * inside
        topmost = torch.zeros(
            self.width * self.height, dtype=torch.long, device=lines.device
        )
        topmost.scatter_reduce_(0, pixel_index, entry_number, "amax")
        winner = topmost[pixel_index]
        colours = torch.cat([colours.new_zeros(1, COLOUR_CHANNEL_COUNT), colours])
        alphas = torch.cat([alphas.new_zeros(1), alphas.reshape(-1)])
        background = self.pixels[lines, columns].float()
        blended = background.lerp_(colours[winner], alphas[winner, None])
        self.pixels[lines, columns] = blended.round_().to(torch.uint8)

    def blend_rectangle(self, left, top, right, bottom, color):
        rows = slice(max(top, 0), min(bottom, self.height))
        columns = slice(max(left, 0), min(right, self.width))
        if rows.start >= rows.stop or columns.start >= columns.stop:
            return
        region = self.pixels[rows, columns]
        blended = region.float().lerp_(self.device_colour(color), color.a)
        region.copy_(blended.round_())

    def channel_values(self, color):
        values = [
            color.r * FULL_INTENSITY,
            color.g * FULL_INTENSITY,
            color.b * FULL_INTENSITY,
        ]
        return values[::-1] if self.reverses_colour else values

    def device_colour(self, color):
        import torch

        key = (color.r, color.g, color.b)
        if key not in self.device_colours:
            self.device_colours[key] = torch.tensor(
                self.channel_values(color),
                dtype=torch.float32,
                device=self.pixels.device,
            )
        return self.device_colours[key]


class InkedPixels(NamedTuple):
    lines: object
    columns: object
    alphas: object


class GlyphAtlas:
    def __init__(self, font_size, device):
        from PIL import ImageFont

        self.font = ImageFont.load_default(size=font_size)
        self.ascent, self.descent = self.font.getmetrics()
        self.device = device
        self.glyph_of_character = {}
        self.advances = []
        self.pixel_counts = []
        self.glyph_tables = []
        self.uploaded_glyph_count = 0

    def glyph(self, character):
        if character not in self.glyph_of_character:
            self.glyph_of_character[character] = self.rasterize(character)
        return self.glyph_of_character[character]

    def rasterize(self, character):
        import numpy as np
        from PIL import Image, ImageDraw

        left, _, right, _ = self.font.getbbox(character)
        image = Image.new("L", (max(right - left, 1), self.ascent + self.descent))
        ImageDraw.Draw(image).text(
            (-left, 0), character, fill=FULL_INTENSITY, font=self.font
        )
        alpha = np.asarray(image)
        lines, columns = np.nonzero(alpha)
        self.glyph_tables.append(
            np.stack([lines, columns + left, alpha[lines, columns]]).astype(np.int64)
        )
        self.advances.append(self.font.getlength(character))
        self.pixel_counts.append(len(lines))
        return len(self.advances) - 1

    def upload_new_glyphs(self):
        import numpy as np
        import torch

        if self.uploaded_glyph_count == len(self.glyph_tables):
            return
        table = torch.from_numpy(np.concatenate(self.glyph_tables, axis=1)).to(
            self.device
        )
        self.pixel_offsets = table[:2]
        self.pixel_alphas = table[2].float() / FULL_INTENSITY
        counts = torch.tensor(self.pixel_counts, device=self.device)
        self.glyph_pixel_counts = counts
        self.glyph_first_pixels = counts.cumsum(0) - counts
        self.uploaded_glyph_count = len(self.glyph_tables)

    # placed_glyphs holds (glyph, line, column) rows
    def inked_pixels(self, placed_glyphs):
        import numpy as np
        import torch

        self.upload_new_glyphs()
        pixel_total = sum(self.pixel_counts[glyph] for glyph, _, _ in placed_glyphs)
        # a list of tuples converts to a tensor far slower than through numpy
        glyphs, line_origins, column_origins = (
            torch.from_numpy(np.array(placed_glyphs, dtype=np.int64)).to(self.device).T
        )
        counts = self.glyph_pixel_counts[glyphs]
        placement_of_pixel = torch.repeat_interleave(
            torch.arange(len(placed_glyphs), device=self.device),
            counts,
            output_size=pixel_total,
        )
        first_output_pixel = counts.cumsum(0) - counts
        source = (
            self.glyph_first_pixels[glyphs][placement_of_pixel]
            + torch.arange(pixel_total, device=self.device)
            - first_output_pixel[placement_of_pixel]
        )
        lines, columns = self.pixel_offsets[:, source]
        return InkedPixels(
            lines=lines + line_origins[placement_of_pixel],
            columns=columns + column_origins[placement_of_pixel],
            alphas=self.pixel_alphas[source],
        )
