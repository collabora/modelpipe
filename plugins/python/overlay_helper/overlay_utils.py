# overlay_utils.py
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import json

from utils.analytics_utils import TRAIL_LABEL_SUFFIX
from .overlay_utils_interface import Color

TRAIL_COLOR = Color(1.0, 1.0, 0.0, 1.0)
TRAIL_LINE_WIDTH = 3


def load_metadata(meta_path, logger):
    """Load JSON metadata from a file and return a dictionary indexed by frame index.

    Args:
        meta_path (str): Path to the JSON metadata file.

    Returns:
        dict: Metadata indexed by frame index.
    """
    if not meta_path:
        logger.error("Frame metadata file path not set.")
        return {}

    if not os.path.exists(meta_path):
        logger.error(f"JSON file not found: {meta_path}")
        return {}

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
            frame_data = all_data.get("frames", [])
            # Store metadata indexed by frame_index
            metadata = {
                frame.get("frame_index"): frame.get("objects", [])
                for frame in frame_data
            }
            logger.info(f"Loaded metadata for {len(metadata)} frames.")
            return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error while loading metadata: {e}")
        return {}


def draw_frame_objects(overlay_graphics, frame_metadata, tracking_display):
    trails = {}
    boxes = []
    for entry in frame_metadata:
        label = entry.get("label", "")
        if label.endswith(TRAIL_LABEL_SUFFIX):
            trails.setdefault(label, []).append(entry["box"])
        else:
            boxes.append(entry)
    for points in trails.values():
        overlay_graphics.draw_polyline(
            [{"x": point["x1"], "y": point["y1"]} for point in points],
            TRAIL_COLOR,
            TRAIL_LINE_WIDTH,
        )
    overlay_graphics.draw_metadata(boxes, tracking_display)
    if tracking_display is not None:
        tracking_display.fade_history()
