# mp-mcp
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json
import os
import re
import sys
import tempfile
import threading
import time
from collections import deque
from importlib.metadata import version
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image

import mp_launch

# the plugin loader reads GST_PLUGIN_PATH at Gst.init
if mp_launch.CHECKOUT_PLUGINS:
    os.environ["GST_PLUGIN_PATH"] = os.pathsep.join(
        filter(
            None, [str(mp_launch.CHECKOUT_PLUGINS), os.environ.get("GST_PLUGIN_PATH")]
        )
    )
sys.path.insert(0, str(mp_launch.plugin_dir()))

import gi  # noqa: E402

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import GLib, GObject, Gst, GstVideo  # noqa: E402

Gst.init(None)

import metasink  # noqa: E402
import documented_pipelines  # noqa: E402
from alertrecorder import Clip, DEFAULT_ENCODER  # noqa: E402
from embedding_index import EmbeddingIndex  # noqa: E402
from engine import model_engines  # noqa: E402
from log.logger_factory import LoggerFactory  # noqa: E402

APPLICATION_NAME = "modelpipe"
MODEL_DEVICE = os.environ.get("MP_MCP_DEVICE", "")
VLM_MODEL = os.environ.get("MP_MCP_VLM_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct")
ELEMENT_PREFIX = "mp_"
RECENT_RECORDS = 1000
RECENT_BUS_MESSAGES = 1000
RECENT_LATENCIES = 30
PENDING_LATENCY_BUFFERS = 64
QUEUE_NAME_PREFIX = "queue"
SNAPSHOT_IMAGE_FORMAT = "jpeg"
FRAME_CONVERT_SECONDS = 5
RGB_CAPS = "video/x-raw,format=RGB"
RGB_BYTES_PER_PIXEL = 3
# greedy, so the same frame captions the same way twice
VLM_TEMPERATURE = 0.0
CLIP_DECODE_PIPELINE = (
    "filesrc name=source ! decodebin ! videoconvert ! video/x-raw,format=I420 "
    "! appsink name=frames sync=false"
)
PREROLL_SECONDS = 10
CLIP_FINISH_SECONDS = 20
PIPELINES_PATH = (
    mp_launch.CHECKOUT_PLUGINS.parent / "PIPELINES.md"
    if mp_launch.CHECKOUT_PLUGINS
    else None
)

server = MCPServer(
    APPLICATION_NAME,
    version=version("modelpipe"),
    instructions=(
        "Pipelines are gst-launch descriptions. End one with mp_metasink to read "
        "its detections, transcripts and alerts back through latest_metadata."
    ),
)


class Session:
    def __init__(self):
        self.pipeline = None
        self.records = deque(maxlen=RECENT_RECORDS)
        # the deque drops its oldest entries
        self.posted = 0
        self.errors = []
        self.bus_messages = deque(maxlen=RECENT_BUS_MESSAGES)
        self.buffer_flows = {}
        self.ended = False
        self.loop_clip = False
        self._duration = 0
        self._stopped = threading.Event()
        self._replay = threading.Event()
        self._replayer = None
        self.lock = threading.Condition()

    def _segment_seek(self, flush):
        pipeline = self.pipeline
        if pipeline is None or self._duration <= 0:
            return False
        flags = Gst.SeekFlags.SEGMENT
        if flush:
            flags |= Gst.SeekFlags.FLUSH
        return pipeline.seek(
            1.0,
            Gst.Format.TIME,
            flags,
            Gst.SeekType.SET,
            0,
            Gst.SeekType.SET,
            self._duration,
        )

    def _arm_segment(self):
        pipeline = self.pipeline
        for _ in range(50):
            if self._stopped.is_set() or pipeline is None:
                return
            ok, duration = pipeline.query_duration(Gst.Format.TIME)
            if ok and duration > 0:
                self._duration = duration
                # A flush at the end stalls the picture.
                self._segment_seek(flush=True)
                return
            time.sleep(0.1)

    def replay(self):
        self._arm_segment()
        while not self._stopped.is_set():
            if not self._replay.wait(0.2):
                continue
            if self._stopped.is_set():
                return
            self._replay.clear()
            pipeline = self.pipeline
            if pipeline is None or self._stopped.is_set():
                continue
            if self._segment_seek(flush=False):
                continue
            # The flush seek waits until the streaming-thread EOS handler returns.
            time.sleep(0.05)
            if pipeline.seek_simple(
                Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0
            ):
                continue
            with self.lock:
                self.ended = True
                self.lock.notify_all()

    def keep_bus_message(self, kind, message, text):
        with self.lock:
            self.bus_messages.append(
                {"type": kind, "source": message.src.get_name(), "text": text}
            )

    def on_message(self, bus, message):
        if message.type == Gst.MessageType.APPLICATION:
            structure = message.get_structure()
            if structure.get_name() == metasink.BUS_MESSAGE_NAME:
                record = json.loads(structure.get_string(metasink.BUS_MESSAGE_FIELD))
                with self.lock:
                    self.records.append(record)
                    self.posted += 1
                    self.lock.notify_all()
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            with self.lock:
                self.errors.append(
                    f"{message.src.get_name()}: {with_debug(error, debug)}"
                )
                self.lock.notify_all()
        elif message.type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            self.keep_bus_message("warning", message, with_debug(warning, debug))
        elif message.type == Gst.MessageType.QOS:
            _format, processed, dropped = message.parse_qos_stats()
            jitter, _proportion, _quality = message.parse_qos_values()
            self.keep_bus_message(
                "qos",
                message,
                f"processed {processed}, dropped {dropped}, "
                f"jitter {jitter // Gst.MSECOND} ms",
            )
        elif message.type == Gst.MessageType.ELEMENT:
            self.keep_bus_message(
                "element", message, message.get_structure().to_string()
            )
        elif message.type in (
            Gst.MessageType.EOS,
            Gst.MessageType.SEGMENT_DONE,
        ):
            if (
                self.loop_clip
                and self.pipeline is not None
                and not self._stopped.is_set()
            ):
                self._replay.set()
            else:
                with self.lock:
                    self.ended = True
                    self.lock.notify_all()
        # nothing else pops this bus
        return Gst.BusSyncReply.DROP


def with_debug(error, debug):
    return f"{error.message} ({debug})" if debug else error.message


class ElementLatency:
    def __init__(self):
        self.entered = {}
        self.recent = deque(maxlen=RECENT_LATENCIES)
        # input and output probes run on different threads behind a queue
        self.lock = threading.Lock()

    def enter(self, _pad, info):
        pts = info.get_buffer().pts
        if pts == Gst.CLOCK_TIME_NONE:
            return Gst.PadProbeReturn.OK
        with self.lock:
            # an element that changes timestamps never matches its input
            if len(self.entered) >= PENDING_LATENCY_BUFFERS:
                del self.entered[next(iter(self.entered))]
            self.entered[pts] = time.monotonic()
        return Gst.PadProbeReturn.OK

    def leave(self, pts, now):
        with self.lock:
            entered = self.entered.pop(pts, None)
            if entered is not None:
                self.recent.append(now - entered)

    def milliseconds(self):
        with self.lock:
            if not self.recent:
                return None
            return round(1000 * sum(self.recent) / len(self.recent), 1)


class BufferFlow:
    def __init__(self, latency):
        self.latency = latency
        self.buffers = 0
        self.first = None
        self.last = None

    def count(self, _pad, info):
        now = time.monotonic()
        if self.first is None:
            self.first = now
        self.buffers += 1
        self.last = now
        self.latency.leave(info.get_buffer().pts, now)
        return Gst.PadProbeReturn.OK

    def report(self, pad, now):
        if self.last is None:
            return {"pad": pad, "buffers": 0}
        flowing_seconds = self.last - self.first
        per_second = (self.buffers - 1) / flowing_seconds if flowing_seconds else 0
        report = {
            "pad": pad,
            "buffers": self.buffers,
            "per_second": round(per_second, 1),
            "seconds_since_last": round(now - self.last, 1),
        }
        latency_ms = self.latency.milliseconds()
        if latency_ms is not None:
            report["latency_ms"] = latency_ms
        return report


def watch_pad(element, pad, latency):
    if pad.get_direction() == Gst.PadDirection.SINK:
        pad.add_probe(Gst.PadProbeType.BUFFER, latency.enter)
        return
    flow = BufferFlow(latency)
    session.buffer_flows[f"{element.get_name()}.{pad.get_name()}"] = flow
    pad.add_probe(Gst.PadProbeType.BUFFER, flow.count)


def watch_buffer_flow(pipeline):
    for element in pipeline.iterate_elements():
        latency = ElementLatency()
        for pad in element.iterate_pads():
            watch_pad(element, pad, latency)
        # decodebin and demuxers add their source pads once data arrives
        element.connect("pad-added", watch_pad, latency)


session = Session()


def running_pipeline():
    if session.pipeline is None:
        raise ToolError("no pipeline is running, call start_pipeline first")
    return session.pipeline


def metadata_sinks(pipeline):
    for element in pipeline.iterate_elements():
        if element.get_factory().get_name() == metasink.MetaSink.GST_PLUGIN_NAME:
            yield element


def named_element(name):
    element = running_pipeline().get_by_name(name)
    if element is None:
        raise ToolError(f"the pipeline has no element named {name!r}")
    return element


def serialized(spec, value):
    if value is None:
        return ""
    # an enum reads as its nick, as on a gst-launch line
    if spec.value_type == GObject.TYPE_STRING:
        return value
    # Gst.value_serialize spells 0.99 as 0.98999999999999999
    if spec.value_type in (GObject.TYPE_DOUBLE, GObject.TYPE_FLOAT):
        return f"{value:g}"
    holder = GObject.Value(spec.value_type)
    holder.set_value(value)
    spelled = Gst.value_serialize(holder)
    return str(value) if spelled is None else spelled


@server.tool(
    description="Start a pipeline from a gst-launch description, stopping any running one first. "
    "End it with mp_metasink so its results reach latest_metadata."
)
def start_pipeline(pipeline: str, loop: bool = False) -> dict:
    stop_pipeline()
    try:
        parsed = Gst.parse_launch(pipeline)
    except GLib.Error as error:
        raise ToolError(error.message) from error
    session.__init__()
    session.pipeline = parsed
    session.loop_clip = loop
    # records already arrive over the bus
    for sink in metadata_sinks(parsed):
        if not sink.get_property("location"):
            sink.set_property("location", os.devnull)
    parsed.get_bus().set_sync_handler(session.on_message)
    watch_buffer_flow(parsed)
    if parsed.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        errors = ", ".join(session.errors) or "the pipeline refused to start"
        stop_pipeline()
        raise ToolError(errors)
    if loop:
        session._replayer = threading.Thread(target=session.replay, daemon=True)
        session._replayer.start()
    return pipeline_status()


@server.tool(
    description="The running pipeline's state, whether it reached end of stream, its errors, "
    "and how many metadata records it has posted."
)
def pipeline_status() -> dict:
    if session.pipeline is None:
        return {"state": "none"}
    _result, state, _pending = session.pipeline.get_state(0)
    with session.lock:
        return {
            "state": state.value_nick,
            "ended": session.ended,
            "errors": list(session.errors),
            "records": len(session.records),
        }


@server.tool(description="Stop and release the running pipeline.")
def stop_pipeline() -> dict:
    pipeline = session.pipeline
    # EOS lets the muxer finish the mp4.
    record = pipeline is not None and pipeline.get_by_name("record") is not None
    session._stopped.set()
    session._replay.set()
    thread = session._replayer
    if record:
        pipeline.send_event(Gst.Event.new_eos())
        with session.lock:
            session.lock.wait(timeout=8)
    if session.pipeline is not None:
        session.pipeline.set_state(Gst.State.NULL)
        session.pipeline = None
    if thread is not None:
        thread.join(timeout=1)
        session._replayer = None
    return {"state": "none"}


@server.tool(
    description="The newest records mp_metasink posted, oldest first: each has a pts in "
    "seconds plus detections, text, or a blob such as alert or depth."
)
def latest_metadata(count: int = 10) -> list[dict]:
    with session.lock:
        return list(session.records)[-count:]


@server.tool(
    description="The newest element messages, warnings and QoS reports the running "
    "pipeline posted on its bus, oldest first, such as level's loudness, spectrum's "
    "bands, or a sink dropping late frames. Each names the element that posted it."
)
def latest_bus_messages(count: int = 10) -> list[dict]:
    with session.lock:
        return list(session.bus_messages)[-count:]


@server.tool(
    description="How many buffers each source pad of the running pipeline has pushed, "
    "named element.pad, with the rate while they flowed and the seconds since the last "
    "one. Data stops at the first pad whose count stopped growing. latency_ms is the "
    "element's recent average time from a buffer entering to it leaving, absent when "
    "the element changes timestamps or has no input."
)
def buffer_flow() -> list[dict]:
    running_pipeline()
    now = time.monotonic()
    return [flow.report(pad, now) for pad, flow in list(session.buffer_flows.items())]


@server.tool(
    description="The elements of the running pipeline ranked by their recent average time "
    "from a buffer entering to it leaving, slowest first, each with latency_ms. An "
    "element with no latency_ms in buffer_flow is absent, and queues are left out."
)
def element_latencies() -> list[dict]:
    slowest = {}
    for flow in buffer_flow():
        element = flow["pad"].rpartition(".")[0]
        # a queue's latency is time spent waiting for the element after it
        if "latency_ms" not in flow or element.startswith(QUEUE_NAME_PREFIX):
            continue
        slowest[element] = max(slowest.get(element, 0), flow["latency_ms"])
    ranked = sorted(slowest.items(), key=lambda item: item[1], reverse=True)
    return [
        {"element": element, "latency_ms": latency_ms} for element, latency_ms in ranked
    ]


@server.tool(
    description="Load the JSON lines a mp_metasink wrote to a file as the current records, "
    "so latest_metadata and wait_for_records read a finished run with no pipeline running."
)
def load_metadata(path: str) -> dict:
    if not os.path.isfile(path):
        raise ToolError(f"no metadata file at {path!r}")
    stop_pipeline()
    session.__init__()
    with open(path, encoding="utf-8") as lines:
        for line in lines:
            if not line.strip():
                continue
            session.records.append(json.loads(line))
            session.posted += 1
    # nothing more will arrive
    session.ended = True
    return {"records": session.posted}


def records_posted_since(posted_before, key):
    fresh = min(session.posted - posted_before, len(session.records))
    recent = list(session.records)[len(session.records) - fresh :]
    if not key:
        return recent
    return [record for record in recent if key in record]


@server.tool(
    description="Wait for mp_metasink to post new records, oldest first, and return them "
    "with the pipeline status. Give a key such as detections or alert to wait only for "
    "records carrying it. Returns whatever arrived when the pipeline ends, fails, or "
    "the timeout in seconds passes first."
)
def wait_for_records(count: int = 1, key: str = "", timeout: float = 30) -> dict:
    # a run loaded from a file has no pipeline behind it
    if not session.ended:
        running_pipeline()
    with session.lock:
        # an ended run posts nothing more
        posted_before = 0 if session.ended else session.posted
        errors_before = len(session.errors)

        def settled():
            return (
                len(records_posted_since(posted_before, key)) >= count
                or session.ended
                or len(session.errors) > errors_before
            )

        session.lock.wait_for(settled, timeout)
        matched = records_posted_since(posted_before, key)
    return {"records": matched, "status": pipeline_status()}


def newest_frame(element):
    if element:
        target = named_element(element)
    else:
        target = next(metadata_sinks(running_pipeline()), None)
        if target is None:
            raise ToolError("the pipeline has no mp_metasink to snapshot")
    name = target.get_name()
    if target.find_property("last-sample") is None:
        raise ToolError(f"{name} has no last-sample property")
    sample = target.get_property("last-sample")
    if sample is None:
        raise ToolError(f"no frame has reached {name} yet")
    return name, sample


def converted_frame(name, sample, caps):
    try:
        return GstVideo.video_convert_sample(
            sample, Gst.Caps.from_string(caps), FRAME_CONVERT_SECONDS * Gst.SECOND
        )
    except GLib.Error as error:
        raise ToolError(f"{name} is not carrying video: {error.message}") from error


@server.tool(
    description="The newest frame a mp_metasink rendered, as a JPEG image. Names a sink of "
    "the running pipeline, or the first mp_metasink when left empty."
)
def snapshot_frame(element: str = "") -> Image:
    name, sample = newest_frame(element)
    buffer = converted_frame(
        name, sample, f"image/{SNAPSHOT_IMAGE_FORMAT}"
    ).get_buffer()
    return Image(
        data=buffer.extract_dup(0, buffer.get_size()), format=SNAPSHOT_IMAGE_FORMAT
    )


def frame_image(sample):
    import numpy
    from PIL import Image as PillowImage

    caps = sample.get_caps()
    structure = caps.get_structure(0)
    width = structure.get_value("width")
    height = structure.get_value("height")
    buffer = sample.get_buffer()
    rows = numpy.frombuffer(buffer.extract_dup(0, buffer.get_size()), dtype=numpy.uint8)
    packed = width * RGB_BYTES_PER_PIXEL
    stride = GstVideo.VideoInfo.new_from_caps(caps).stride[0]
    if stride != packed:
        rows = rows.reshape(height, stride)[:, :packed]
    return PillowImage.fromarray(rows.reshape(height, width, RGB_BYTES_PER_PIXEL))


vlm_engines = {}


def vlm_engine(model_name):
    engine = vlm_engines.get(model_name)
    if engine is None:
        # torch loads only once a caption asks for it
        from engine.vlm_engine import VlmEngine

        engine = VlmEngine()
        engine.do_set_device(model_device())
        engine.do_load_model(model_name)
        vlm_engines[model_name] = engine
    return engine


@server.tool(
    description="Caption the newest frame a mp_metasink rendered with a vision-language "
    "model, answering the prompt about it. Names a sink of the running pipeline, or the "
    "first mp_metasink when left empty. On cpu a caption takes tens of seconds."
)
def describe_frame(
    prompt: str = "Describe this image.", element: str = "", max_tokens: int = 64
) -> str:
    name, sample = newest_frame(element)
    image = frame_image(converted_frame(name, sample, RGB_CAPS))
    return vlm_engine(VLM_MODEL).do_generate(
        image, prompt, None, max_tokens, VLM_TEMPERATURE
    )


def model_device():
    if MODEL_DEVICE:
        return MODEL_DEVICE
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


text_embedding_engines = {}


def text_embedding_engine(model_name):
    engine = text_embedding_engines.get(model_name)
    if engine is None:
        # torch loads only once a search asks for it
        from engine.embedding_engine import EmbeddingEngine

        engine = EmbeddingEngine()
        engine.do_set_device(model_device())
        engine.do_load_model(model_name)
        text_embedding_engines[model_name] = engine
    return engine


@server.tool(
    description="Search a video index written by mp_embeddingsink for the frames a description "
    "matches, closest first: each result has a pts in seconds, the source_id of its "
    "stream, and a cosine similarity score."
)
def search_video(query: str, index: str, count: int = 5) -> list[dict]:
    if not os.path.isfile(index):
        raise ToolError(f"no embedding index at {index!r}")
    opened = EmbeddingIndex.open(index)
    try:
        model_name = opened.model_name()
        if not model_name:
            raise ToolError(f"the index at {index!r} is empty")
        vector = text_embedding_engine(model_name).do_text_embedding(query)
        if vector is None:
            raise ToolError(f"{model_name} cannot embed text, index with a CLIP model")
        return opened.search(vector, count)
    finally:
        opened.close()


def decode_failure(pipeline, source):
    message = pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
    if message is None:
        return f"cannot decode {source!r}"
    return message.parse_error()[0].message


@server.tool(
    description="Cut the seconds of video around a pts out of a video file into a webm, so a "
    "search_video hit becomes a clip. Returns where it was written, the range it covers "
    "and how many frames it holds."
)
def clip_at(source: str, pts: float, seconds: float = 4.0, location: str = "") -> dict:
    if not os.path.isfile(source):
        raise ToolError(f"no video at {source!r}")
    start = max(pts - seconds / 2, 0)
    end = start + seconds
    path = location or str(
        Path(tempfile.gettempdir()) / f"{Path(source).stem}-{start:.2f}.webm"
    )
    decode = Gst.parse_launch(CLIP_DECODE_PIPELINE)
    # a launch line would split the path on spaces
    decode.get_by_name("source").set_property("location", source)
    frames = decode.get_by_name("frames")
    decode.set_state(Gst.State.PAUSED)
    result, _state, _pending = decode.get_state(PREROLL_SECONDS * Gst.SECOND)
    if result == Gst.StateChangeReturn.FAILURE:
        reason = decode_failure(decode, source)
        decode.set_state(Gst.State.NULL)
        raise ToolError(reason)
    seeked = decode.seek(
        1.0,
        Gst.Format.TIME,
        Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,
        Gst.SeekType.SET,
        int(start * Gst.SECOND),
        Gst.SeekType.SET,
        int(end * Gst.SECOND),
    )
    if not seeked:
        decode.set_state(Gst.State.NULL)
        raise ToolError(f"cannot seek {source!r}")
    decode.set_state(Gst.State.PLAYING)
    clip = None
    frame_count = 0
    while True:
        sample = frames.emit("pull-sample")
        if sample is None:
            break
        buffer = sample.get_buffer()
        if clip is None:
            logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
            clip = Clip(path, DEFAULT_ENCODER, sample.get_caps(), buffer.pts, logger)
        clip.push(buffer)
        frame_count += 1
    decode.set_state(Gst.State.NULL)
    if clip is None:
        raise ToolError(f"no frames in {source!r} between {start} and {end}")
    clip.finish().join(CLIP_FINISH_SECONDS)
    return {"path": path, "start": start, "end": end, "frames": frame_count}


@server.tool(
    description="Set a property on a named element of the running pipeline, the value spelled "
    "as it would be on a gst-launch line."
)
def set_property(element: str, property: str, value: str) -> dict:
    target = named_element(element)
    if target.find_property(property) is None:
        raise ToolError(f"{element} has no property {property!r}")
    Gst.util_set_object_arg(target, property, value)
    return {property: get_property(element, property)}


@server.tool(description="Read a property of a named element of the running pipeline.")
def get_property(element: str, property: str) -> str:
    target = named_element(element)
    spec = target.find_property(property)
    if spec is None:
        raise ToolError(f"{element} has no property {property!r}")
    return serialized(spec, target.get_property(property))


@server.tool(description="The ModelPipe elements this GStreamer can load.")
def list_elements() -> list[dict]:
    factories = Gst.Registry.get().get_feature_list(Gst.ElementFactory)
    return [
        {
            "name": factory.get_name(),
            "description": factory.get_metadata(Gst.ELEMENT_METADATA_DESCRIPTION),
        }
        for factory in sorted(factories, key=lambda factory: factory.get_name())
        if factory.get_name().startswith(ELEMENT_PREFIX)
    ]


def initial_value(instance, spec):
    # a fresh instance's value stands in for the default, as in gst-inspect
    if not spec.flags & GObject.ParamFlags.READABLE:
        return None
    return serialized(spec, instance.get_property(spec.name))


@server.tool(
    description="An element's description and properties, for any element gst-launch knows."
)
def inspect(element: str) -> dict:
    factory = Gst.ElementFactory.find(element)
    if factory is None:
        raise ToolError(f"no element named {element!r}")
    instance = factory.create(None)
    return {
        "name": element,
        "description": factory.get_metadata(Gst.ELEMENT_METADATA_DESCRIPTION),
        "properties": [
            {
                "name": spec.name,
                "type": spec.value_type.name,
                "blurb": spec.blurb,
                "default": initial_value(instance, spec),
            }
            for spec in instance.list_properties()
        ],
    }


@server.tool(
    description="Which engines and devices run a model name or file, with the extra to "
    "install for the missing ones."
)
def engine_options(model: str) -> dict:
    options = model_engines.engine_options(model)
    if not options["tasks"]:
        raise ToolError(f"no task or engine matches {model!r}")
    return model_engines.with_available_devices(options)


def prompt_name(heading):
    return re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")


def section_prompt(heading, descriptions, repository):
    def pipeline_section():
        opening = (
            f"These are the {heading} pipelines from ModelPipe's PIPELINES.md. "
            "start_pipeline takes each line below as written, swap a display sink such as "
            "autovideosink for mp_metasink to read the results back, and file paths are "
            f"relative to {repository}."
        )
        return "\n".join([opening, *descriptions])

    return pipeline_section


def register_pipeline_prompts(doc_path):
    sections = documented_pipelines.pipelines_by_section(doc_path)
    for heading, descriptions in sections.items():
        server.prompt(
            name=prompt_name(heading),
            description=f"The PIPELINES.md pipelines under {heading}",
        )(section_prompt(heading, descriptions, doc_path.parent))


if PIPELINES_PATH is not None:
    register_pipeline_prompts(PIPELINES_PATH)


def main():
    # video sinks put it in their window title, after the stream's title tag
    GLib.set_application_name(APPLICATION_NAME)
    server.run()


if __name__ == "__main__":
    main()
