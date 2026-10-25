# RelationMetaAnalytics
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstAnalytics", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GstAnalytics, GLib  # noqa: E402

from backend.analytics import AnalyticsBackend  # noqa: E402


class RelationMetaAnalytics(AnalyticsBackend):
    def add_relation_meta(self, buf):
        return GstAnalytics.buffer_add_analytics_relation_meta(buf)

    def get_relation_meta(self, buf):
        return GstAnalytics.buffer_get_analytics_relation_meta(buf)

    def remove_relation_meta(self, buf):
        meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
        if meta is None:
            return False
        # GstAnalytics relation meta is an add-once accumulator: detections are
        # added into it as *_mtd entries. The Python bindings expose no path to
        # detach it (the RelationMeta wrapper is not a Gst.Meta, so it cannot be
        # passed to Gst.Buffer.remove_meta). Attempt it for bindings that do
        # accept it; otherwise report that nothing was removed.
        try:
            return bool(buf.remove_meta(meta))
        except (TypeError, AttributeError):
            return False

    def relation_length(self, meta):
        return GstAnalytics.relation_get_length(meta)

    def quark(self, label):
        if isinstance(label, int):
            return label
        return GLib.quark_from_string(str(label))

    def add_object(self, meta, label, x, y, w, h, score):
        qk = self.quark(label)
        ret, mtd = meta.add_od_mtd(qk, x, y, w, h, score)
        return mtd if ret else None

    def add_classification(self, meta, index, label):
        qk = self.quark(label)
        ret, mtd = meta.add_one_cls_mtd(index, qk)
        return mtd if ret else None

    def add_tracking(self, meta, track_id, timestamp=None):
        if timestamp is None:
            timestamp = Gst.util_get_timestamp()
        ret, mtd = meta.add_tracking_mtd(track_id, timestamp)
        return mtd if ret else None

    def relate(self, meta, src, dst):
        return bool(
            GstAnalytics.RelationMeta.set_relation(
                meta, GstAnalytics.RelTypes.RELATE_TO, src.id, dst.id
            )
        )

    def read_objects(self, meta):
        objects = []
        count = GstAnalytics.relation_get_length(meta)
        tracks = []
        for index in range(count):
            ret, tracking_mtd = meta.get_tracking_mtd(index)
            if ret:
                tracks.append(tracking_mtd)
        for index in range(count):
            ret, od_mtd = meta.get_od_mtd(index)
            if not ret or od_mtd is None:
                continue
            label = GLib.quark_to_string(od_mtd.get_obj_type())
            presence, x, y, w, h, score = od_mtd.get_location()
            if not presence:
                continue
            objects.append(
                {
                    "label": label,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "score": score,
                    "track_id": related_track_id(meta, od_mtd, tracks),
                    "ref": od_mtd,
                }
            )
        return objects

    def track_object(self, meta, obj, track_id):
        self.relate(meta, obj["ref"], self.add_tracking(meta, track_id))


# get_direct_related prints a PyGObject warning per call, so the relation is tested pairwise
def related_track_id(meta, od_mtd, tracks):
    for tracking_mtd in tracks:
        if (
            meta.get_relation(od_mtd.id, tracking_mtd.id)
            & GstAnalytics.RelTypes.RELATE_TO
        ):
            return tracking_mtd.get_info()[1]
    return None
