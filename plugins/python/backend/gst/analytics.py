# Analytics store selection
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

import gi

gi.require_version("Gst", "1.0")

from log.logger_factory import LoggerFactory  # noqa: E402

ANALYTICS_STORE_VARIABLE = "MP_ANALYTICS"
RELATION_META_STORE = "meta"
CHUNK_STORE = "chunk"


def gstanalytics_importable():
    try:
        gi.require_version("GstAnalytics", "1.0")
        from gi.repository import GstAnalytics  # noqa: F401
    except (ImportError, ValueError):
        return False
    return True


def relation_meta_analytics():
    from backend.gst.detection_copy_patch import keep_detections_on_buffer_copy
    from backend.gst.relation_meta import RelationMetaAnalytics

    keep_detections_on_buffer_copy()
    return RelationMetaAnalytics()


def chunk_analytics():
    from backend.gst.chunk_analytics import ChunkAnalytics

    return ChunkAnalytics()


def select_analytics():
    logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
    store = os.environ.get(ANALYTICS_STORE_VARIABLE, "").lower()
    if store == RELATION_META_STORE:
        logger.info(
            f"analytics in the relation meta, forced by {ANALYTICS_STORE_VARIABLE}"
        )
        return relation_meta_analytics()
    if store == CHUNK_STORE:
        logger.info(
            f"analytics in a buffer chunk, forced by {ANALYTICS_STORE_VARIABLE}"
        )
        return chunk_analytics()
    if store:
        raise ImportError(
            f"Unknown {ANALYTICS_STORE_VARIABLE}={store!r}, supported stores: "
            f"{RELATION_META_STORE!r}, {CHUNK_STORE!r}"
        )
    if gstanalytics_importable():
        logger.info("analytics in the relation meta, GstAnalytics found")
        return relation_meta_analytics()
    logger.info("analytics in a buffer chunk, GstAnalytics missing")
    return chunk_analytics()
