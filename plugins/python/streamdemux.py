# StreamDemux
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import backend
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GObject  # noqa: E402
from backend import analytics  # noqa: E402
from backend.analytics import stream_label  # noqa: E402
from log.logger_factory import LoggerFactory  # noqa: E402
from utils.metadata import BATCH_SOURCE_INDEX_METADATA  # noqa: E402
from collections import defaultdict  # noqa: E402


class StreamDemux(Gst.Element):
    __gstmetadata__ = (
        "StreamDemux",
        "Demuxer",
        "Custom stream demuxer with metadata splitting",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    if backend.BACKEND == "gst":
        __gsttemplates__ = (
            Gst.PadTemplate.new(
                "sink",
                Gst.PadDirection.SINK,
                Gst.PadPresence.ALWAYS,
                Gst.Caps.from_string("video/x-raw"),
            ),
            Gst.PadTemplate.new(
                "src_%u",
                Gst.PadDirection.SRC,
                Gst.PadPresence.REQUEST,
                Gst.Caps.from_string("video/x-raw"),
            ),
        )

    __gproperties__ = {
        "max-queue-size": (
            GObject.TYPE_UINT,
            "Maximum queue size",
            "Maximum number of buffers to queue per source pad",
            1,
            1000,
            10,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.sinkpad = Gst.Pad.new_from_template(self.get_pad_template("sink"), "sink")
        self.sinkpad.set_event_function_full(self.event)
        self.sinkpad.set_chain_function_full(self.chain)
        self.add_pad(self.sinkpad)
        self.pad_count = 0
        self.buffer_queues = defaultdict(list)
        self.src_pads = {}
        self.max_queue_size = 10
        self.pads_requested = False  # Track initial pad creation

    def do_set_property(self, prop, value):
        if prop.name == "max-queue-size":
            self.max_queue_size = value
        else:
            raise AttributeError(f"Unknown property: {prop.name}")

    def do_get_property(self, prop):
        if prop.name == "max-queue-size":
            return self.max_queue_size
        raise AttributeError(f"Unknown property: {prop.name}")

    def do_request_new_pad(self, template, name, caps):
        if name is None:
            name = f"src_{self.pad_count}"
            self.pad_count += 1
        self.logger.debug(f"Requesting new pad: {name}")
        if "src_" in name:
            pad = Gst.Pad.new_from_template(template, name)
            self.add_pad(pad)
            if self.sinkpad.has_current_caps():
                caps = self.sinkpad.get_current_caps()
                self.logger.info(f"Setting caps on {pad.get_name()}: {caps}")
                pad.set_caps(caps)
            self.src_pads[name] = pad
            return pad
        return None

    def do_release_pad(self, pad):
        pad_name = pad.get_name()
        self.logger.debug(f"Releasing pad: {pad_name}")
        self.buffer_queues[pad_name].clear()
        self.remove_pad(pad)
        del self.src_pads[pad_name]

    def process_src_pad(self, buffer, memory_chunk, batch_position, source_index):
        out_buffer = Gst.Buffer.new()
        out_buffer.append_memory(memory_chunk)
        out_buffer.pts = buffer.pts
        out_buffer.duration = buffer.duration
        out_buffer.dts = buffer.dts
        out_buffer.offset = buffer.offset
        meta = analytics.get_relation_meta(buffer)
        if meta:
            out_meta = analytics.add_relation_meta(out_buffer)
            objects = analytics.read_objects(meta)
            self.logger.info(
                f"Processing {len(objects)} analytics relations for stream_{source_index}"
            )
            # elements label by batch position, not source index
            for obj in objects:
                if obj["stream"] == batch_position:
                    new_od_mtd = analytics.add_object(
                        out_meta,
                        stream_label(source_index, obj["label"]),
                        obj["x"],
                        obj["y"],
                        obj["w"],
                        obj["h"],
                        obj["score"],
                    )
                    if new_od_mtd is None:
                        self.logger.error(f"Failed to attach metadata: {obj['label']}")
        return out_buffer

    def push_buffer(self, pad, buffer, pad_name):
        self.logger.debug(f"Pushing buffer on {pad_name}")
        ret = pad.push(buffer)
        if ret != Gst.FlowReturn.OK:
            self.logger.error(f"Failed to push buffer on {pad_name}: {ret}")
            if ret not in (Gst.FlowReturn.FLUSHING, Gst.FlowReturn.ERROR):
                if len(self.buffer_queues[pad_name]) < self.max_queue_size:
                    self.buffer_queues[pad_name].append(buffer)
                else:
                    self.logger.warning(
                        f"Dropping buffer on {pad_name}: queue full (size={self.max_queue_size})"
                    )
            return False
        self.logger.debug(f"Successfully pushed buffer on {pad_name}")
        return True

    def chain(self, pad, parent, buffer):
        self.logger.debug("Processing buffer in chain function")
        source_indices = []
        if buffer.n_memory() > 0:
            try:
                source_indices = [
                    index for (index,) in BATCH_SOURCE_INDEX_METADATA.read(buffer)
                ]
                self.logger.info(f"Decoded source indices: {source_indices}")
                # Request all pads upfront on first buffer
                if not self.pads_requested:
                    for idx in source_indices:
                        pad_name = f"src_{idx}"
                        if pad_name not in self.src_pads:
                            self.request_pad(
                                self.get_pad_template("src_%u"), pad_name, None
                            )
                    self.pads_requested = True
                    # Ensure caps are set on all pads
                    if self.sinkpad.has_current_caps():
                        caps = self.sinkpad.get_current_caps()
                        for src_pad in self.src_pads.values():
                            if not src_pad.has_current_caps():
                                self.logger.info(
                                    f"Setting caps on {src_pad.get_name()}: {caps}"
                                )
                                src_pad.set_caps(caps)
            except ValueError as e:
                self.logger.error(str(e))

        for batch_position, idx in enumerate(source_indices):
            memory_chunk = buffer.peek_memory(batch_position)
            pad_name = f"src_{idx}"
            src_pad = self.get_static_pad(pad_name)
            if src_pad is None:
                src_pad = self.request_pad(
                    self.get_pad_template("src_%u"), pad_name, None
                )
                if src_pad is None:
                    self.logger.error(f"Failed to request pad: {pad_name}")
                    continue
            if not hasattr(src_pad, "stream_started"):
                src_pad.push_event(Gst.Event.new_stream_start(f"demux-stream-{idx}"))
                src_pad.stream_started = True
            if not src_pad.has_current_caps() and self.sinkpad.has_current_caps():
                caps = self.sinkpad.get_current_caps()
                self.logger.info(f"Setting CAPS on {src_pad.get_name()}: {caps}")
                src_pad.set_caps(caps)
            if not hasattr(src_pad, "segment_pushed"):
                segment = Gst.Segment()
                segment.init(Gst.Format.TIME)
                segment.start = buffer.pts
                self.logger.info(
                    f"Sending SEGMENT event on {src_pad.get_name()} with start={segment.start}"
                )
                src_pad.push_event(Gst.Event.new_segment(segment))
                src_pad.segment_pushed = True

            out_buffer = self.process_src_pad(
                buffer, memory_chunk, batch_position, source_index=idx
            )
            while self.buffer_queues[pad_name]:
                if self.push_buffer(
                    src_pad, self.buffer_queues[pad_name].pop(0), pad_name
                ):
                    continue
                break
            self.push_buffer(src_pad, out_buffer, pad_name)
        return Gst.FlowReturn.OK

    def event(self, pad, parent, event):
        self.logger.debug(f"Received event: {event.type}")
        # the src pads push their own stream-start, caps and segment
        if event.type in (
            Gst.EventType.EOS,
            Gst.EventType.FLUSH_START,
            Gst.EventType.FLUSH_STOP,
        ):
            return Gst.Pad.event_default(pad, parent, event)
        return True


if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("mp_streamdemux", StreamDemux)
