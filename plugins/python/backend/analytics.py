# AnalyticsBackend
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Framework-agnostic interface for analytics metadata.

Detection / classification / tracking results are attached to a buffer as
analytics metadata. GStreamer expresses this via `GstAnalytics` relation
metadata. Elements should go through this interface instead of calling the
framework API directly, so the same element code can run on a different backend
later.

The `meta` handle and the detection handles returned by `add_*` are opaque to
callers: pass them back into the same backend (e.g. into `relate`). The read
side (`read_objects`) returns plain dicts so consumers need no framework types.
"""

import re
from abc import ABC, abstractmethod

STREAM_LABEL = re.compile(r"stream_(\d+)_(.+)$")


# a batched element labels a detection `stream_<i>_<class>`, `i` the frame's place in the batch
def split_stream_label(label):
    match = STREAM_LABEL.match(label)
    if match is None:
        return None, label
    return int(match.group(1)), match.group(2)


def stream_label(stream, label):
    if stream is None:
        return label
    return f"stream_{stream}_{label}"


class AnalyticsBackend(ABC):
    """Attach to and read analytics metadata on a buffer/frame."""

    @abstractmethod
    def add_relation_meta(self, buf):
        """Attach (or fetch the existing) analytics relation meta. Returns an
        opaque meta handle, or None on failure."""

    @abstractmethod
    def get_relation_meta(self, buf):
        """Return the attached relation meta handle, or None if absent."""

    @abstractmethod
    def remove_relation_meta(self, buf):
        """Remove any attached relation meta. Returns True if something was
        removed."""

    @abstractmethod
    def relation_length(self, meta):
        """Number of relations (detections) currently held by `meta`."""

    @abstractmethod
    def quark(self, label):
        """Intern a string label into the backend's id space (a GQuark for
        GStreamer). Accepts a str; ints pass through unchanged."""

    @abstractmethod
    def add_object(self, meta, label, x, y, w, h, score):
        """Add an object-detection box (x, y, w, h) with a class `label` and
        confidence `score`. Returns an opaque detection handle, or None."""

    @abstractmethod
    def add_classification(self, meta, index, label):
        """Add a single-class classification result tagged with a stream
        `index` and a `label`. Returns an opaque handle, or None."""

    @abstractmethod
    def add_tracking(self, meta, track_id, timestamp=None):
        """Add a tracking record for `track_id`. `timestamp` defaults to the
        backend's current running time. Returns an opaque handle, or None."""

    @abstractmethod
    def relate(self, meta, src, dst):
        """Relate two handles (e.g. a detection to its tracking record).
        Returns True on success."""

    @abstractmethod
    def read_objects(self, meta):
        """Read back the object detections held by `meta`. Returns a list of
        dicts with keys: ``label`` (str, without the ``stream_<i>_`` prefix),
        ``stream`` (the batch position from that prefix, or None), ``x``, ``y``,
        ``w``, ``h``, ``score``, ``track_id`` (the related tracking id, or None) and ``ref`` (what
        `track_object` takes). Only present (valid-location) detections are
        returned."""

    @abstractmethod
    def track_object(self, meta, obj, track_id): ...
