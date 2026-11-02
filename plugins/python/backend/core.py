# MLEngineMixin
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Portable, framework-agnostic ML element logic.

`MLEngineMixin` holds everything an ML element needs that does NOT depend on
the multimedia framework: the engine/model lifecycle and the backing storage
for the common tunable properties.

A framework backend builds a concrete element by combining this mixin with its
own element base class (`GstBase.BaseTransform`, `GstBase.Aggregator`, ...) and
declaring the properties in whatever form the framework requires
(`@GObject.Property` for GStreamer, etc). The property getters/setters read and
write the `self._<name>` fields initialised here, so the shared logic stays
identical across backends.

This module imports no `gi`; keep it that way so the same mixin can load under a
non-GStreamer backend.
"""

from engine.engine_factory import EngineFactory
from engine.engine_manager import EngineManager
from log.logger_factory import LoggerFactory

#: A tunable the element stores and mirrors onto the engine under the same name:
#: ``(name, type, default, blurb)``.
ENGINE_TUNABLES = (
    ("batch_size", int, 1, "Number of items to process in a batch"),
    ("frame_stride", int, 1, "How often to process a frame"),
    (
        "stream_priority",
        int,
        0,
        "Priority of the CUDA stream the inference runs on, lower is more urgent "
        "as in CUDA, clamped to the range the device allows",
    ),
    (
        "precision",
        str,
        "fp32",
        "Inference precision, fp32 or fp16. Loading the model fails on an engine "
        "that does not run the requested precision",
    ),
)


# the input-format and post-process value that leaves the choice to the engine
ENGINE_CHOOSES = "auto"

EXPORTED_INPUT_FORMAT = "nchw"
EXPORTED_POST_PROCESS = "none"

ENGINE_NAME_BLURB = (
    f"Machine Learning Engine to use: {', '.join(EngineFactory.BUILTIN_ENGINES)}, "
    "or a custom engine name"
)


def _engine_tunable(gobject, name, value_type, default, blurb):
    field = f"_{name}"

    def get(self):
        return getattr(self, field)

    def set(self, value):
        setattr(self, field, value)
        if self.engine:
            setattr(self.engine, name, value)

    # pygobject takes the property's name off the getter
    get.__name__ = name
    return gobject.Property(get, type=value_type, default=default, blurb=blurb).setter(
        set
    )


def ml_property_namespace(gobject):
    """The tunables every ML element takes, declared with `gobject`'s `Property`.

    Returned as a namespace for an element base to unpack into its class body,
    not as a mixin to inherit: pygobject installs a property only when it sits
    in the class's own dict, so an inherited one leaves a gst element with none
    of these. Both backends build their bases from this call, so a knob cannot
    exist on one and not the other.
    """

    @gobject.Property(type=str)
    def device(self):
        "Device to run the inference on (cpu, cuda, cuda:0, cuda:1, etc.)"
        return self.mgr.device

    @device.setter
    def device(self, value):
        self.mgr.set_device(value)

    @gobject.Property(type=str)
    def model_name(self):
        "Name of the pre-trained model or local model path"
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @gobject.Property(type=str, blurb=ENGINE_NAME_BLURB)
    def engine_name(self):
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        # a task element runs pytorch through its own engine
        if value == EngineFactory.PYTORCH_ENGINE and self.task_engine_name:
            value = self.task_engine_name
        if value != self.mgr.engine_name:
            self.mgr.engine = None
        self.mgr.engine_name = value

    @gobject.Property(type=str, default="auto")
    def input_format(self):
        "Input tensor layout: auto, nhwc, or nchw"
        return self.engine.input_format if self.engine else self._input_format

    @input_format.setter
    def input_format(self, value):
        self._input_format = value
        if self.engine:
            self.engine.input_format = value

    @gobject.Property(type=str, default="auto")
    def post_process(self):
        "Post-processing format for raw engine output (auto, none, or a key from detection_decoder)"
        return self.engine.post_process if self.engine else self._post_process

    @post_process.setter
    def post_process(self, value):
        self._post_process = value
        if self.engine:
            self.engine.post_process = value

    @gobject.Property(type=bool, default=False)
    def compile(self):
        "Enable torch.compile optimization for the model"
        return self._compile

    @compile.setter
    def compile(self, value):
        self._compile = value
        if value:
            self.kwargs["compile"] = True
        else:
            self.kwargs.pop("compile", None)

    @gobject.Property(type=str, default="")
    def only_on(self):
        "Run only on frames carrying this blob, for example alert, or detections for any analytics object"
        return self._only_on

    @only_on.setter
    def only_on(self, value):
        self._only_on = value

    namespace = {
        "device": device,
        "model_name": model_name,
        "engine_name": engine_name,
        "input_format": input_format,
        "post_process": post_process,
        "compile": compile,
        "only_on": only_on,
    }
    for name, value_type, default, blurb in ENGINE_TUNABLES:
        namespace[name] = _engine_tunable(gobject, name, value_type, default, blurb)
    return namespace


#: The `only-on` value naming the analytics detections rather than a blob.
ONLY_ON_DETECTIONS = "detections"


class FrameProcessingMixin:
    """The per-frame work a video element does, shared by every backend.

    The backend driver extracts the frame and handles the framework's error
    signalling; what is left is the same on all of them, so it lives here once:
    infer, turn the result into a frame and/or a metadata blob, write both back.
    An element whose task class follows that contract supplies only its
    `META_HEADER`; one that does not overrides `process_frames`.
    """

    #: Tag on the metadata blob this element appends, e.g. ``b"GST-DEPTH:"``.
    #: `None` for an element that appends none.
    META_HEADER = None

    def process_frames(self, frames, num_sources, fmt, target):
        """Infer over the extracted frame(s) and write the result to `target`.

        Raises on a hard failure; the driver maps that to its own error return.
        """
        from backend import frameio

        result = self.forward(frames)
        if result is None:
            raise RuntimeError(f"{type(self).__name__}: inference returned None")
        # Inference sees the whole batch, but one buffer carries one frame back:
        # the overlay and the blob describe the primary source.
        frame = frames[0] if num_sources > 1 else frames
        if num_sources > 1 and isinstance(result, list):
            if not result:
                return
            result = result[0]
        output, blob = self.decode(frame, result, fmt)
        frameio.write_result(target, output, blob, self.META_HEADER)

    def carries_only_on(self, target):
        """Whether `target` carries what `only-on` names, so this frame runs.

        Both backends gate on the same two things: the detections an upstream
        element attached, or a blob under its own name.
        """
        from backend import analytics, frameio

        if self._only_on == ONLY_ON_DETECTIONS:
            meta = analytics.get_relation_meta(target)
            return meta is not None and bool(analytics.read_objects(meta))
        return self._only_on in frameio.read_blobs(target)


class PayloadProcessingMixin:
    """The per-buffer work a non-video element does, shared by every backend.

    These families read one media type and write another (text in / text out,
    audio in / text out), so a buffer is opaque bytes rather than a frame. The
    backend driver maps the input buffer and turns whatever comes back into
    output buffers; the middle, bytes to bytes, is the same everywhere and is
    what the element supplies.
    """

    def process_payload(self, payload: bytes) -> list[bytes]:
        """Turn one input buffer's bytes into the payloads to send onward.

        Zero or more, because these families are not all 1:1: an element that
        accumulates across buffers returns an empty list until it has something
        to say, and one that chunks its output returns several at once.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement process_payload"
        )

    def payload_duration_ns(self, payload_size):
        """How long a payload of this many bytes plays for.

        `None`, the usual answer, means the output covers the same stretch of
        the stream as the input did, so it keeps the input's timing. An element
        that generates media of its own length (speech from a text buffer) works
        it out from the byte count instead.
        """
        return None


class MLEngineMixin:
    """Engine/model lifecycle and the start hook, shared by every ML element."""

    def on_start(self):
        """One-time setup before the first buffer, on any backend.

        For an element with its own background work to start (an inference
        thread, a warm-up). Framework lifecycle virtuals like `do_start` only
        exist on GStreamer, so anything a hosted element also needs goes here.
        """

    def _ensure_started(self):
        """Run `on_start` once, for a backend with no start virtual of its own."""
        if not getattr(self, "_started", False):
            self._started = True
            self.on_start()

    def _ml_init(self):
        """Initialise shared ML state. Call this from the element's __init__."""
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.mgr = EngineManager(self.logger)
        self.task_engine_name = None
        self.exported_model = None
        self.kwargs = {}
        self._batch_size = 1
        self._frame_stride = 1
        self._model_name = None
        self._stream_priority = 0
        self._precision = "fp32"
        self._system_prompt = None
        self._prompt = None
        self._compile = False
        self._only_on = ""
        self._input_format = ENGINE_CHOOSES
        self._post_process = ENGINE_CHOOSES

    @property
    def engine(self):
        return self.mgr.engine

    @property
    def runs_exported_model(self):
        return (
            self.task_engine_name is not None
            and self.mgr.engine_name != self.task_engine_name
        )

    @property
    def task_engine(self):
        return self.exported_model if self.runs_exported_model else self.engine

    # Engine / model lifecycle. None of this touches the framework, so every
    # backend reuses it verbatim.
    def initialize_engine(self):
        if not self.engine and self.mgr.engine_name:
            self.mgr.initialize_engine()
            self.engine.batch_size = self._batch_size
            self.engine.frame_stride = self._frame_stride
            self.engine.stream_priority = self._stream_priority
            self.engine.precision = self._precision
            # an engine like drpai starts with a layout of its own
            if self._input_format != ENGINE_CHOOSES:
                self.engine.input_format = self._input_format
            if self._post_process != ENGINE_CHOOSES:
                self.engine.post_process = self._post_process
            if self.runs_exported_model:
                self.engine.input_format = EXPORTED_INPUT_FORMAT
                self.engine.post_process = EXPORTED_POST_PROCESS
        if not self.engine:
            self.logger.error(f"Unsupported ML engine: {self.mgr.engine_name}")

    def do_load_model(self):
        self.initialize_engine()
        if self.engine is None:
            self.logger.error(
                f"Cannot load model {self._model_name}: engine not initialized"
            )
            return
        if self._model_name is None:
            self.logger.warning("Cannot load model as model name is not set")
            return
        model_name = self._model_name
        if self.runs_exported_model:
            if self.exported_model is None:
                self.exported_model = self.export_model(model_name)
            self.exported_model.engine = self.engine
            model_name = self.exported_model.path
        self.mgr.do_load_model(model_name, **self.kwargs)

    def get_model(self):
        """Gets the model from the engine."""
        self.initialize_engine()
        if self.engine is None:
            self.logger.error(
                f"Cannot get model {self._model_name}: engine not initialized"
            )
            return None
        if self.engine:
            return self.engine.get_model()
        return None

    def set_model(self, model):
        """Sets the model in the engine."""
        self.initialize_engine()
        if self.engine is None:
            self.logger.error("Cannot load model: engine not initialized")
            return False
        self.engine.model = model
        self.logger.info("Model set successfully in the engine.")

    def get_tokenizer(self):
        self.initialize_engine()
        if self.engine is None:
            self.logger.error("Cannot get tokenizer: engine not initialized")
            return None
        return self.mgr.get_tokenizer()
