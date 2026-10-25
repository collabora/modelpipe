# AnalyticsUtils
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import re

from backend import analytics
from log.logger_factory import LoggerFactory

# zero-size objects sharing one such label are the points of a polyline, oldest first
TRAIL_LABEL_SUFFIX = "_trail"


class AnalyticsUtils:
    """
    Utility class for extracting analytics metadata from GStreamer buffers.

    This class provides methods to retrieve and parse object detection (OD) and tracking metadata
    attached to video buffers through `backend.analytics`. It is designed to work with
    metadata added by object detection and tracking elements, such as YOLO-based transformers.

    Supported Label Formats:
    - Named tracking: '<name>_id_<id>' → label='<name>', track_id=<id>
    - Simpler tracking: 'id_<id>' → label='id_<id>', track_id=<id>
    - Any other format, such as a class name or '<name>_trail' → label='<full_label>', track_id=None

    The extracted metadata includes labels, track IDs, confidence scores, and bounding box coordinates.
    """

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)

    def extract_analytics_metadata(self, buffer):
        metadata = []
        meta = analytics.get_relation_meta(buffer)
        if not meta:
            self.logger.info("No analytics relation metadata found on buffer")
            return metadata

        count = analytics.relation_length(meta)
        self.logger.info(f"Found {count} analytics relations in metadata")
        for obj in analytics.read_objects(meta):
            full_label = obj["label"]

            # CLIP/SigLIP zero-shot classification metadata (clip_<label>)
            clip_match = re.match(r"clip_(.+)", full_label)
            if clip_match:
                label = clip_match.group(1).replace("_", " ")
                metadata.append(
                    {
                        "type": "classification",
                        "label": label,
                        "confidence": obj["score"],
                    }
                )
                self.logger.debug(f"Added classification entry: {metadata[-1]}")
                continue

            # Standard object detection metadata
            track_id, label = self.extract_id_from_label(full_label)
            if obj["track_id"] is not None:
                track_id = obj["track_id"]
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            metadata.append(
                {
                    "label": label,
                    "track_id": track_id,
                    "confidence": obj["score"],
                    "box": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
                }
            )
            self.logger.debug(f"Added detection entry: {metadata[-1]}")
        return metadata

    def extract_id_from_label(self, full_label):
        match = re.match(r"(.+)_id_(\d+)", full_label)
        if match:
            track_id = int(match.group(2))
            label = match.group(1)
            self.logger.debug(
                f"Extracted track_id {track_id} and label '{label}' from '{full_label}' (named tracking)"
            )
            return track_id, label

        match = re.match(r"id_(\d+)", full_label)
        if match:
            track_id = int(match.group(1))
            label = f"id_{track_id}"
            self.logger.debug(
                f"Extracted track_id {track_id} and label '{label}' from '{full_label}' (simpler tracking)"
            )
            return track_id, label

        return None, full_label
