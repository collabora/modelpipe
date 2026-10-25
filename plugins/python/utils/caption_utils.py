# caption_utils.py
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import json


def load_captions(caption_path, logger):
    """
    Load JSON caption data from a file and return a dictionary indexed by frame index.

    Args:
        caption_path (str): Path to the JSON caption file.
        logger: Logger instance for logging messages.

    Returns:
        dict: Captions indexed by frame index (e.g., {0: "caption text", 1: "other text"}).
    """
    if not caption_path:
        logger.error("Caption file path not set.")
        return {}

    if not os.path.exists(caption_path):
        logger.error(f"JSON file not found: {caption_path}")
        return {}

    try:
        with open(caption_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
            frame_data = all_data.get("frames", [])
            # Store captions indexed by frame_index
            captions = {
                frame.get("frame_index"): frame.get("caption", "")
                for frame in frame_data
            }
            logger.info(f"Loaded captions for {len(captions)} frames.")
            return captions
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error while loading captions: {e}")
        return {}
