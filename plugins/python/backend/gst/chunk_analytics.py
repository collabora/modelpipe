# ChunkAnalytics
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json
from collections import namedtuple

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from backend.analytics import AnalyticsBackend, split_stream_label  # noqa: E402
from utils.metadata import first_memory_with_prefix  # noqa: E402

ANALYTICS_CHUNK_HEADER = b"GST-ANALYTICS:"
OBJECT_FIELDS = ("label", "x", "y", "w", "h", "score")

ChunkEntry = namedtuple("ChunkEntry", "id")


class AnalyticsChunk:
    def __init__(self, buffer, document):
        self.buffer = buffer
        self.objects = document["objects"]
        self.classifications = document["classifications"]
        self.tracking = document["tracking"]
        self.relations = document["relations"]

    def length(self):
        return len(self.objects) + len(self.classifications) + len(self.tracking)

    def document(self):
        return {
            "objects": self.objects,
            "classifications": self.classifications,
            "tracking": self.tracking,
            "relations": self.relations,
        }


def write_chunk(chunk):
    payload = ANALYTICS_CHUNK_HEADER + json.dumps(chunk.document()).encode()
    # Gst.Memory.new_wrapped aborts in pygobject
    memory = Gst.Buffer.new_wrapped(payload).get_memory(0)
    index = first_memory_with_prefix(chunk.buffer, ANALYTICS_CHUNK_HEADER)
    if index is None:
        chunk.buffer.append_memory(memory)
        return
    chunk.buffer.replace_memory(index, memory)


class ChunkAnalytics(AnalyticsBackend):
    def add_relation_meta(self, buf):
        chunk = self.get_relation_meta(buf)
        if chunk is not None:
            return chunk
        chunk = AnalyticsChunk(
            buf, {"objects": [], "classifications": [], "tracking": [], "relations": []}
        )
        write_chunk(chunk)
        return chunk

    def get_relation_meta(self, buf):
        index = first_memory_with_prefix(buf, ANALYTICS_CHUNK_HEADER)
        if index is None:
            return None
        with buf.peek_memory(index).map(Gst.MapFlags.READ) as info:
            payload = bytes(info.data[len(ANALYTICS_CHUNK_HEADER) :])
        return AnalyticsChunk(buf, json.loads(payload))

    def remove_relation_meta(self, buf):
        index = first_memory_with_prefix(buf, ANALYTICS_CHUNK_HEADER)
        if index is None:
            return False
        buf.remove_memory(index)
        return True

    def relation_length(self, meta):
        return meta.length()

    def quark(self, label):
        return label

    def add_object(self, meta, label, x, y, w, h, score):
        entry = ChunkEntry(meta.length())
        # the relation meta stores integer boxes
        meta.objects.append(
            {
                "id": entry.id,
                "label": label,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "score": float(score),
            }
        )
        write_chunk(meta)
        return entry

    def add_classification(self, meta, index, label):
        entry = ChunkEntry(meta.length())
        meta.classifications.append({"id": entry.id, "index": index, "label": label})
        write_chunk(meta)
        return entry

    def add_tracking(self, meta, track_id, timestamp=None):
        if timestamp is None:
            timestamp = Gst.util_get_timestamp()
        entry = ChunkEntry(meta.length())
        meta.tracking.append(
            {"id": entry.id, "track_id": track_id, "timestamp": timestamp}
        )
        write_chunk(meta)
        return entry

    def relate(self, meta, src, dst):
        meta.relations.append([src.id, dst.id])
        write_chunk(meta)
        return True

    def read_objects(self, meta):
        track_of_entry = {entry["id"]: entry["track_id"] for entry in meta.tracking}
        track_of_object = {
            src: track_of_entry[dst]
            for src, dst in meta.relations
            if dst in track_of_entry
        }
        objects = []
        for obj in meta.objects:
            stream, label = split_stream_label(obj["label"])
            objects.append(
                {
                    **{field: obj[field] for field in OBJECT_FIELDS},
                    "label": label,
                    "stream": stream,
                    "track_id": track_of_object.get(obj["id"]),
                    "ref": obj["id"],
                }
            )
        return objects

    def track_object(self, meta, obj, track_id):
        self.relate(meta, ChunkEntry(obj["ref"]), self.add_tracking(meta, track_id))
