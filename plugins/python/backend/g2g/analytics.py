# G2gAnalyticsBackend (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Analytics metadata over the g2g host's `MetaSink`.

GStreamer attaches a `GstAnalyticsRelationMeta` to the buffer and relates
detections, classifications and tracking records. The g2g host instead hands the
element a flat `MetaSink` per frame with `add_object(label, x, y, w, h, score)` /
`add_classification(label, score)` / `add_blob(...)` to stage into and
`objects()` / `blobs()` to read the frame's incoming metadata back; the host then
materializes what was staged into the frame's `AnalyticsMeta`. This backend maps the rich
`AnalyticsBackend` interface the leaf task code uses onto that flat sink:

  * the "relation meta" handle is a thin wrapper over the bound sink that counts
    how many records were staged (so `relation_length` works);
  * string labels are interned to the `u32` ids the sink expects (`quark`);
  * `add_*` return the sink's own staging handle, which `relate` passes back to
    pair a detection with its tracking id;
  * the host fills the sink with the frame's incoming detections before the
    call, so `read_objects` returns what an upstream element attached; what this
    element stages is appended to those on the way out.
"""

import threading

from backend.analytics import AnalyticsBackend


class _RelationMeta:
    """The g2g stand-in for a buffer's relation meta: the bound sink plus a count
    of staged records (GstAnalytics tracks relations; the flat sink only counts)."""

    def __init__(self, sink):
        self.sink = sink
        self.count = 0


class _Bound:
    """What one element has staged: the frame's sink, and its label id space."""

    def __init__(self):
        self.sink = None
        self.meta = None
        self.upstream_names = []
        self.upstream_ids = {}
        self.own_names = []
        self.own_ids = {}
        self.published = False


class G2gAnalyticsBackend(AnalyticsBackend):
    """`AnalyticsBackend` mapping detections/classifications onto a `MetaSink`."""

    def __init__(self):
        # Per-thread because the host runs one thread per element: one shared
        # binding would stage an element's detections on another's frame.
        self._threads = threading.local()

    @property
    def _bound(self):
        bound = getattr(self._threads, "bound", None)
        if bound is None:
            bound = _Bound()
            self._threads.bound = bound
        return bound

    def bind(self, sink):
        """Bind this frame's sink (called per frame by the g2g element bases).
        A fresh relation-meta is created lazily on first `add_relation_meta`."""
        bound = self._bound
        bound.sink = sink
        bound.meta = None
        # the frame carries one name table for every element
        bound.upstream_names = list(sink.class_names()) if sink is not None else []
        bound.upstream_ids = {name: i for i, name in enumerate(bound.upstream_names)}
        bound.published = False

    def quark(self, label):
        """Intern a string label into the `u32` id space the sink expects; ints
        pass through unchanged (matches the GStreamer GQuark contract)."""
        if isinstance(label, int):
            return label
        bound = self._bound
        qid = bound.upstream_ids.get(label)
        if qid is not None:
            return qid
        own = bound.own_ids.get(label)
        if own is None:
            own = len(bound.own_names)
            bound.own_names.append(label)
            bound.own_ids[label] = own
            bound.published = False
        return len(bound.upstream_names) + own

    def _publish_class_names(self):
        bound = self._bound
        if bound.sink is None or bound.published:
            return
        bound.sink.set_class_names(bound.upstream_names + bound.own_names)
        bound.published = True

    def add_relation_meta(self, buf):
        bound = self._bound
        if bound.sink is None:
            return None
        if bound.meta is None:
            bound.meta = _RelationMeta(bound.sink)
        return bound.meta

    def get_relation_meta(self, buf):
        # An upstream detection is reason enough for a handle: the element does
        # not have to have staged anything of its own first.
        bound = self._bound
        if bound.meta is None and bound.sink is not None and bound.sink.objects():
            bound.meta = _RelationMeta(bound.sink)
        return bound.meta

    def remove_relation_meta(self, buf):
        bound = self._bound
        had = bound.meta is not None
        bound.meta = None
        return had

    def relation_length(self, meta):
        return meta.count if meta else 0

    def add_object(self, meta, label, x, y, w, h, score):
        if meta is None:
            return None
        meta.count += 1
        qid = self.quark(label)
        self._publish_class_names()
        return meta.sink.add_object(
            qid, float(x), float(y), float(w), float(h), float(score)
        )

    def add_classification(self, meta, index, label):
        if meta is None:
            return None
        # The flat sink's add_classification is (label, score); the gst `index`
        # (stream id) has no place in it and is dropped.
        meta.count += 1
        qid = self.quark(label)
        self._publish_class_names()
        return meta.sink.add_classification(qid, 1.0)

    def add_tracking(self, meta, track_id, timestamp=None):
        # The host stamps its own arrival time, so the gst `timestamp` is dropped.
        if meta is None:
            return None
        meta.count += 1
        return meta.sink.add_tracking(int(track_id))

    def relate(self, meta, src, dst):
        if meta is None or src is None or dst is None:
            return False
        meta.sink.relate(int(src), int(dst))
        return True

    def read_objects(self, meta):
        # The frame's incoming detections. What this element stages is not read back.
        if not meta:
            return []
        objects = meta.sink.objects()
        for index, (obj, track_id) in enumerate(zip(objects, meta.sink.tracking_ids())):
            # the host has already split the batch and dropped the prefix
            obj["stream"] = None
            obj["track_id"] = track_id
            obj["ref"] = index
        return objects

    def track_object(self, meta, obj, track_id):
        meta.count += 1
        meta.sink.track_upstream(obj["ref"], int(track_id))


#: Defined here, like `frameio`, to avoid a circular import through `backend`.
analytics = G2gAnalyticsBackend()
