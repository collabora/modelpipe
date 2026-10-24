# Anonymize
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import math

import backend
from backend import GObject, InPlaceVideoTransform, CUDA_RGBA_DRAW_FRAME_CAPABILITIES
from utils.analytics_utils import AnalyticsUtils

BLUR_MODE = "blur"
PIXELATE_MODE = "pixelate"
BLACKOUT_MODE = "blackout"
ANONYMIZE_MODES = (BLUR_MODE, PIXELATE_MODE, BLACKOUT_MODE)
REDUCED_PIXEL_SIZE = 12
COLOUR_CHANNEL_COUNT = 3


def reduced_dimensions(height, width):
    return (
        max(1, math.ceil(height / REDUCED_PIXEL_SIZE)),
        max(1, math.ceil(width / REDUCED_PIXEL_SIZE)),
    )


def numpy_effect(pixels, mode):
    import numpy as np

    if mode == BLACKOUT_MODE:
        return np.zeros_like(pixels)
    height, width = pixels.shape[:2]
    reduced_height, reduced_width = reduced_dimensions(height, width)
    reduced = np.empty(
        (reduced_height, reduced_width, pixels.shape[2]), dtype=np.float32
    )
    for reduced_row in range(reduced_height):
        top = reduced_row * REDUCED_PIXEL_SIZE
        bottom = min(height, top + REDUCED_PIXEL_SIZE)
        for reduced_column in range(reduced_width):
            left = reduced_column * REDUCED_PIXEL_SIZE
            right = min(width, left + REDUCED_PIXEL_SIZE)
            reduced[reduced_row, reduced_column] = pixels[top:bottom, left:right].mean(
                axis=(0, 1)
            )
    if mode == PIXELATE_MODE:
        return (
            reduced.repeat(REDUCED_PIXEL_SIZE, axis=0)
            .repeat(REDUCED_PIXEL_SIZE, axis=1)[:height, :width]
            .round()
            .astype(np.uint8)
        )
    row_positions = np.linspace(0, reduced_height - 1, height)
    column_positions = np.linspace(0, reduced_width - 1, width)
    top_rows = np.floor(row_positions).astype(int)
    bottom_rows = np.minimum(top_rows + 1, reduced_height - 1)
    left_columns = np.floor(column_positions).astype(int)
    right_columns = np.minimum(left_columns + 1, reduced_width - 1)
    row_weights = (row_positions - top_rows)[:, None, None]
    column_weights = (column_positions - left_columns)[None, :, None]
    top = (
        reduced[top_rows[:, None], left_columns[None, :]] * (1.0 - column_weights)
        + reduced[top_rows[:, None], right_columns[None, :]] * column_weights
    )
    bottom = (
        reduced[bottom_rows[:, None], left_columns[None, :]] * (1.0 - column_weights)
        + reduced[bottom_rows[:, None], right_columns[None, :]] * column_weights
    )
    return (top * (1.0 - row_weights) + bottom * row_weights).round().astype(np.uint8)


def torch_effect(pixels, mode):
    import torch
    import torch.nn.functional as functional

    if mode == BLACKOUT_MODE:
        return torch.zeros_like(pixels)
    height, width = pixels.shape[:2]
    reduced_height, reduced_width = reduced_dimensions(height, width)
    channels_first = pixels.permute(2, 0, 1).unsqueeze(0).float()
    reduced = functional.interpolate(
        channels_first, size=(reduced_height, reduced_width), mode="area"
    )
    if mode == BLUR_MODE:
        transformed = functional.interpolate(
            reduced, size=(height, width), mode="bilinear", align_corners=False
        )
    else:
        transformed = functional.interpolate(
            reduced, size=(height, width), mode="nearest"
        )
    return transformed.squeeze(0).permute(1, 2, 0).round_().to(torch.uint8)


def fade_weights(height, width, edge_fade, reference):
    import numpy as np

    if isinstance(reference, np.ndarray):
        rows = np.arange(height)
        columns = np.arange(width)
        row_distance = np.minimum(rows, height - 1 - rows)
        column_distance = np.minimum(columns, width - 1 - columns)
        distance = np.minimum(row_distance[:, None], column_distance[None, :])
        return np.clip(distance / edge_fade, 0.0, 1.0)[..., None]

    import torch

    rows = torch.arange(height, device=reference.device)
    columns = torch.arange(width, device=reference.device)
    row_distance = torch.minimum(rows, height - 1 - rows)
    column_distance = torch.minimum(columns, width - 1 - columns)
    distance = torch.minimum(row_distance[:, None], column_distance[None, :])
    return (distance / edge_fade).clamp_(0.0, 1.0)[..., None]


def apply_effect(region, mode, edge_fade):
    import numpy as np

    pixels = region[..., :COLOUR_CHANNEL_COUNT]
    transformed = (
        numpy_effect(pixels, mode)
        if isinstance(region, np.ndarray)
        else torch_effect(pixels, mode)
    )
    if edge_fade == 0:
        if isinstance(region, np.ndarray):
            np.copyto(pixels, transformed)
        else:
            pixels.copy_(transformed)
        return
    weights = fade_weights(region.shape[0], region.shape[1], edge_fade, region)
    if isinstance(region, np.ndarray):
        blended = (
            pixels.astype(np.float32) * (1.0 - weights)
            + transformed.astype(np.float32) * weights
        )
        np.copyto(pixels, np.rint(blended).astype(np.uint8))
    else:
        blended = pixels.float() * (1.0 - weights) + transformed.float() * weights
        pixels.copy_(blended.round_().to(pixels.dtype))


class Anonymize(InPlaceVideoTransform):
    __gstmetadata__ = (
        "Anonymize",
        "Filter/Effect/Video",
        "Anonymizes regions from analytics detections",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    FRAME_CAPABILITIES = CUDA_RGBA_DRAW_FRAME_CAPABILITIES

    @GObject.Property(
        type=str,
        default=BLUR_MODE,
        nick="Mode",
        blurb=f"Effect applied to detections: {', '.join(ANONYMIZE_MODES)}",
    )
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in ANONYMIZE_MODES:
            raise ValueError(
                f"Unknown anonymize mode {value}, allowed: {', '.join(ANONYMIZE_MODES)}"
            )
        self._mode = value

    labels = GObject.Property(
        type=str,
        default="",
        nick="Labels",
        blurb="Comma-separated detection labels to anonymize, empty selects all",
        flags=GObject.ParamFlags.READWRITE,
    )
    box_expansion = GObject.Property(
        type=float,
        default=0.1,
        minimum=0.0,
        maximum=10.0,
        nick="Box Expansion",
        blurb="Fraction of the detection width and height added on every side",
        flags=GObject.ParamFlags.READWRITE,
    )
    edge_fade = GObject.Property(
        type=int,
        default=0,
        minimum=0,
        maximum=2147483647,
        nick="Edge Fade",
        blurb="Pixels over which the effect fades in from each region edge",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self._mode = BLUR_MODE
        self.analytics_utils = AnalyticsUtils()

    def selected_labels(self):
        return {label.strip() for label in self.labels.split(",") if label.strip()}

    def expanded_region(self, box):
        width = box["x2"] - box["x1"]
        height = box["y2"] - box["y1"]
        left = max(0, math.floor(box["x1"] - self.box_expansion * width))
        top = max(0, math.floor(box["y1"] - self.box_expansion * height))
        right = min(self.width, math.ceil(box["x2"] + self.box_expansion * width))
        bottom = min(self.height, math.ceil(box["y2"] + self.box_expansion * height))
        if right <= left or bottom <= top:
            return None
        return top, bottom, left, right

    def process_in_place(self, frame, fmt, buf):
        selected_labels = self.selected_labels()
        for detection in self.analytics_utils.extract_analytics_metadata(buf):
            if "box" not in detection:
                continue
            if selected_labels and detection["label"] not in selected_labels:
                continue
            region = self.expanded_region(detection["box"])
            if region is None:
                continue
            top, bottom, left, right = region
            apply_effect(frame[top:bottom, left:right], self.mode, self.edge_fade)


if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("mp_anonymize", Anonymize)
