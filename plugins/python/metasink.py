# MetaSink
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    import json
    import sys

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402

    from log.logger_factory import LoggerFactory  # noqa: E402
    from backend import analytics, GObject  # noqa: E402
    from utils.blobs import blob_name, read_blobs  # noqa: E402
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'mp_metasink' element will not be available. Error: {e}"
    )

BUS_MESSAGE_NAME = "mp-metadata"
BUS_MESSAGE_FIELD = "json"

TEXT_MEDIA_PREFIX = "text/"


def buffer_record(buffer, media_type):
    from backend.gst.chunk_analytics import ANALYTICS_CHUNK_HEADER

    record = {}
    if buffer.pts != Gst.CLOCK_TIME_NONE:
        record["pts"] = buffer.pts / Gst.SECOND
    if media_type and media_type.startswith(TEXT_MEDIA_PREFIX):
        with buffer.map(Gst.MapFlags.READ) as info:
            record["text"] = bytes(info.data).decode("utf-8", errors="replace")
        return record
    meta = analytics.get_relation_meta(buffer)
    # an empty list tells a frame with no detections from a dropped frame
    record["detections"] = analytics.read_objects(meta) if meta else []
    blobs = read_blobs(buffer)
    # the detections above already carry the analytics chunk
    blobs.pop(blob_name(ANALYTICS_CHUNK_HEADER), None)
    for name, payload in blobs.items():
        # embedding blobs are binary, not json
        try:
            record[name] = json.loads(payload)
        except ValueError:
            continue
    return record


class MetaSink(GstBase.BaseSink):
    GST_PLUGIN_NAME = "mp_metasink"

    __gstmetadata__ = (
        "Metadata Sink",
        "Sink",
        "Writes the analytics metadata, blobs and text of each buffer as one JSON line",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    if backend.BACKEND == "gst":
        __gsttemplates__ = Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.new_any(),
        )

    location = GObject.Property(
        type=str,
        default="",
        nick="Location",
        blurb="File to append the JSON lines to, stdout when empty",
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.set_sync(False)
        # a sporadic feed like a caption pad must not hold the pipeline in preroll
        self.set_async_enabled(False)
        self._output = None
        self._media_type = None

    def do_start(self):
        self._output = (
            open(self.location, "a", encoding="utf-8") if self.location else sys.stdout
        )
        return True

    def do_stop(self):
        if self._output is not sys.stdout:
            self._output.close()
        self._output = None
        return True

    def do_set_caps(self, caps):
        self._media_type = caps.get_structure(0).get_name()
        return True

    def do_render(self, buffer):
        record = buffer_record(buffer, self._media_type)
        line = json.dumps(record)
        self._output.write(line + "\n")
        self._output.flush()
        structure = Gst.Structure.new_empty(BUS_MESSAGE_NAME)
        structure.set_value(BUS_MESSAGE_FIELD, line)
        self.post_message(Gst.Message.new_application(self, structure))
        return Gst.FlowReturn.OK


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        MetaSink.GST_PLUGIN_NAME, MetaSink
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_metasink' element will not be registered because required modules are missing."
    )
