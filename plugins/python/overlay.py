# Overlay
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
import backend

from utils.analytics_utils import AnalyticsUtils

CAN_REGISTER_ELEMENT = True
try:
    from overlay_helper.overlay_utils import (
        draw_frame_objects,
        load_metadata,
    )
    from overlay_helper.overlay_utils_interface import (
        DEFAULT_RENDERER,
        GRAPHICS_TYPE_BY_RENDERER,
        TrackingDisplay,
        GraphicsType,
        OverlayGraphicsFactory,
    )
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    gi.require_version("GstGL", "1.0")
    from gi.repository import Gst, GstBase, GstVideo, GstGL  # noqa: E402

    try:
        gi.require_version("GstVulkan", "1.0")
        from gi.repository import GstVulkan
    except (ValueError, ImportError):
        # brew's gstreamer ships no vulkan typelib
        GstVulkan = None
    from backend import GObject, post_error
    from log.logger_factory import LoggerFactory
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'mp_overlay' element will not be available. Error: {e}"
    )


def is_vulkan_memory(memory):
    return GstVulkan is not None and GstVulkan.is_vulkan_image_memory(memory)


VIDEO_FORMATS = "video/x-raw, format=(string){ RGBA, ARGB, BGRA, ABGR }; video/x-raw(memory:CUDAMemory), format=(string){ RGBA, ARGB, BGRA, ABGR }; video/x-raw(memory:GLMemory), format=(string){ RGBA, ARGB, BGRA, ABGR }; video/x-raw(memory:VulkanImage), format=(string){ RGBA, BGRA }"
# Building a Gst object needs Gst.init, which only the gst backend calls.
if backend.BACKEND == "gst":
    OVERLAY_CAPS = Gst.Caps.from_string(VIDEO_FORMATS)


class Overlay(GstBase.BaseTransform):
    __gstmetadata__ = (
        "Overlay",
        "Filter/Effect/Video",
        "Overlays object detection / tracking data on video",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    if backend.BACKEND == "gst":
        src_template = Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            OVERLAY_CAPS.copy(),
        )

        sink_template = Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            OVERLAY_CAPS.copy(),
        )
        __gsttemplates__ = (src_template, sink_template)

    meta_path = GObject.Property(
        type=str,
        default=None,
        nick="Metadata File Path",
        blurb="Path to the JSON file containing frame metadata with bounding boxes and tracking data",
        flags=GObject.ParamFlags.READWRITE,
    )
    tracking = GObject.Property(
        type=bool,
        default=True,
        nick="Enable Tracking Display",
        blurb="Enable or disable tracking display",
        flags=GObject.ParamFlags.READWRITE,
    )

    @GObject.Property(
        type=str,
        default=DEFAULT_RENDERER,
        nick="Renderer",
        blurb=f"Renderer for system memory frames: {', '.join(GRAPHICS_TYPE_BY_RENDERER)}",
    )
    def renderer(self):
        return self.selected_renderer

    @renderer.setter
    def renderer(self, value):
        if value not in GRAPHICS_TYPE_BY_RENDERER:
            raise ValueError(
                f"Unknown renderer {value}, allowed: {', '.join(GRAPHICS_TYPE_BY_RENDERER)}"
            )
        self.selected_renderer = value

    def __init__(self):
        super().__init__()
        self.selected_renderer = DEFAULT_RENDERER
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.extracted_metadata = {}
        self.from_file = False
        self.frame_counter = 0
        self.tracking_display = TrackingDisplay()
        self.do_set_dims(0, 0)
        self.video_format = None
        self.overlay_graphics = None
        self.graphics_type = None
        self.gl_context = None
        self.use_opengl = False
        self.use_vulkan = False
        self.gl_display = None
        self.context_set = False
        self.created_context = False
        self.context_received = False

    def on_message(self, bus, message):
        self.logger.info(f"Received bus message: {message.type}")
        if message.type == Gst.MessageType.EOS:
            self.logger.info("Reset frame counter.")
            self.frame_counter = 0
        elif message.type == Gst.MessageType.NEED_CONTEXT:
            self.logger.info("Received NEED_CONTEXT message")
            context = message.parse_context()
            if context:
                context_type = context.get_context_type()
                self.logger.info(f"Context type: {context_type}")
                if context_type == "gst.gl.app_context":
                    self.gl_context = context.get_gl_context()
                    if self.gl_context:
                        self.logger.info(
                            "Successfully acquired OpenGL context from NEED_CONTEXT"
                        )
                        self.use_opengl = True
                        self.context_set = True
                        self.context_received = True
                    else:
                        self.logger.warning(
                            "Failed to get OpenGL context from NEED_CONTEXT: gl_context is None"
                        )
                elif context_type == "gst.gl.display":
                    self.gl_display = context.get_gl_display()
                    if self.gl_display:
                        self.logger.info(
                            "Successfully acquired GL display from NEED_CONTEXT"
                        )
                    else:
                        self.logger.warning(
                            "Failed to get GL display from NEED_CONTEXT: gl_display is None"
                        )
                else:
                    self.logger.warning(f"Unknown context type: {context_type}")
            else:
                self.logger.warning("NEED_CONTEXT message has no context")

    def do_start(self):
        self.logger.info("Starting Overlay element")
        if self.bus:
            self.logger.info("Bus is available, adding signal watch")
            self.bus.add_signal_watch()
            self.bus.connect("message", self.on_message)
            self.logger.info("Added signal watch to pipeline's bus.")
        else:
            self.logger.error("Could not get the bus from the pipeline.")

        self.logger.info("Attempting to get graphics context in do_start")
        context = self.get_context("gst.gl.app_context")
        if context:
            self.logger.info("Found existing OpenGL context in do_start")
            self.gl_context = context.get_gl_context()
            if self.gl_context:
                self.logger.info("Successfully acquired OpenGL context in do_start")
                self.use_opengl = True
                self.context_set = True
                self.context_received = True
            else:
                self.logger.warning(
                    "Failed to get OpenGL context in do_start: gl_context is None"
                )
        else:
            self.logger.warning(
                "No OpenGL context found in do_start, will wait for NEED_CONTEXT message"
            )

        return True

    def do_set_caps(self, incaps, outcaps):
        video_info = GstVideo.VideoInfo.new_from_caps(incaps)
        self.do_set_dims(video_info.width, video_info.height)
        self.video_format = video_info.finfo.name
        self.logger.info(f"Video caps set: width={self.width}, height={self.height}")

        self.use_opengl = "memory:GLMemory" in incaps.to_string()
        self.use_vulkan = "memory:VulkanImage" in incaps.to_string()
        self.logger.info(
            f"Using OpenGL: {self.use_opengl}, Using Vulkan: {self.use_vulkan}"
        )

        return True

    def do_set_dims(self, width, height):
        self.width = width
        self.height = height

    def do_transform_ip(self, buf):
        from backend.gst.cuda_memory import is_cuda_buffer

        # First try to extract metadata from frame meta
        if not self.from_file:
            analytics_utils = AnalyticsUtils()
            extracted = analytics_utils.extract_analytics_metadata(buf)
            self.logger.debug(f"Extracted buffer metadata: {extracted}")
            self.extracted_metadata = extracted
            if not extracted:
                self.logger.warning(
                    "No metadata extracted from buffer, checking file fallback"
                )

        # Fall back to file if buffer metadata is empty and meta_path is set
        if not self.extracted_metadata and self.meta_path:
            self.logger.info(f"Attempting to load metadata from file: {self.meta_path}")
            self.extracted_metadata = load_metadata(self.meta_path, self.logger)
            self.from_file = True

        # If no metadata from either source, pass through without overlay
        if not self.extracted_metadata:
            self.logger.info(
                "No metadata available from buffer or file, passing through buffer"
            )
            self.frame_counter += 1
            return Gst.FlowReturn.OK

        frame_metadata = None
        if self.from_file:
            frame_metadata = self.extracted_metadata.get(self.frame_counter, [])
            self.logger.info(
                f"Using file metadata for frame {self.frame_counter}: {frame_metadata}"
            )
        else:
            frame_metadata = self.extracted_metadata
            self.logger.debug(f"Using buffer metadata: {frame_metadata}")

        # Initialize the graphics backend if not already done
        if self.overlay_graphics is None:
            self.logger.info(
                f"Initializing graphics backend. use_opengl={self.use_opengl}, use_vulkan={self.use_vulkan}"
            )

            is_gl_buffer = False
            is_vulkan_buffer = False
            if buf.n_memory() > 0:
                memory = buf.peek_memory(0)
                is_gl_buffer = GstGL.is_gl_memory(memory)
                is_vulkan_buffer = is_vulkan_memory(memory)
            else:
                self.logger.warning(
                    "Buffer has no memory objects, falling back to Cairo"
                )
                is_gl_buffer = False
                is_vulkan_buffer = False

            # Determine the graphics backend to use
            if is_cuda_buffer(buf):
                self.graphics_type = GraphicsType.TORCH
            elif self.use_vulkan and is_vulkan_buffer:
                self.graphics_type = GraphicsType.VULKAN
            elif self.use_opengl and is_gl_buffer:
                self.graphics_type = GraphicsType.OPENGL
                # Try to get the OpenGL context if not already set
                if not self.gl_context:
                    context = self.get_context("gst.gl.app_context")
                    if context:
                        self.gl_context = context.get_gl_context()
                        if self.gl_context:
                            self.logger.info("Successfully acquired OpenGL context")
                            self.context_set = True
                        else:
                            self.logger.warning(
                                "Failed to get OpenGL context: gl_context is None"
                            )
                            self.use_opengl = False
                            self.graphics_type = GraphicsType.CAIRO
                    else:
                        self.logger.warning(
                            "No OpenGL context found, falling back to Cairo"
                        )
                        self.use_opengl = False
                        self.graphics_type = GraphicsType.CAIRO
            else:
                self.graphics_type = GRAPHICS_TYPE_BY_RENDERER[self.selected_renderer]
                self.logger.info(
                    f"Using {self.selected_renderer} rendering (no Vulkan or OpenGL buffer)"
                )

            # Create the graphics backend
            kwargs = {}
            if self.graphics_type in (
                GraphicsType.SKIA,
                GraphicsType.TORCH,
                GraphicsType.VULKAN,
            ):
                kwargs["video_format"] = self.video_format
            try:
                if self.graphics_type == GraphicsType.VULKAN:
                    kwargs["vulkan_queue"] = self.upstream_vulkan_queue()
                self.overlay_graphics = OverlayGraphicsFactory.create(
                    self.graphics_type, self.width, self.height, **kwargs
                )
            except Exception as exception:
                post_error(
                    self, f"{self.graphics_type.value} renderer setup failed", exception
                )
                return Gst.FlowReturn.ERROR
            self.logger.info(f"Drawing with the {self.graphics_type.value} renderer")

        # Handle rendering based on the graphics type
        if self.graphics_type == GraphicsType.VULKAN:
            if not is_vulkan_memory(buf.peek_memory(0)):
                self.logger.error("Buffer is not in VulkanImage memory")
                return Gst.FlowReturn.ERROR

            try:
                self.overlay_graphics.initialize(buf)
                self.do_post_process(frame_metadata)
                self.overlay_graphics.finalize()
            except Exception as exception:
                post_error(self, "vulkan rendering failed", exception)
                return Gst.FlowReturn.ERROR

        elif self.graphics_type == GraphicsType.OPENGL:
            if not GstGL.is_gl_memory(buf.peek_memory(0)):
                self.logger.error(
                    "Buffer is not in GLMemory, cannot proceed with OpenGL overlay"
                )
                return Gst.FlowReturn.ERROR

            if not self.gl_context:
                self.logger.error(
                    "No OpenGL context available, cannot proceed with OpenGL overlay"
                )
                return Gst.FlowReturn.ERROR

            try:
                self.gl_context.make_current()
                self.overlay_graphics.initialize(buf)
                self.do_post_process(frame_metadata)
                self.overlay_graphics.finalize()
            except Exception as exception:
                post_error(self, "opengl rendering failed", exception)
                return Gst.FlowReturn.ERROR
            finally:
                self.gl_context.make_current(False)

        elif self.graphics_type == GraphicsType.TORCH:
            try:
                with self.overlay_graphics.drawing_stream():
                    self.draw_in_place(buf, frame_metadata)
            except Exception as exception:
                post_error(self, "torch rendering failed", exception)
                return Gst.FlowReturn.ERROR

        else:  # Cairo or Skia rendering on system memory
            video_meta = GstVideo.buffer_get_video_meta(buf)
            if not video_meta:
                self.logger.error(
                    "No video meta available, cannot proceed with overlay"
                )
                return Gst.FlowReturn.ERROR
            try:
                self.draw_in_place(buf, frame_metadata)
            except Exception as exception:
                post_error(
                    self, f"{self.graphics_type.value} rendering failed", exception
                )
                return Gst.FlowReturn.ERROR

        self.frame_counter += 1
        return Gst.FlowReturn.OK

    def upstream_vulkan_queue(self):
        query = GstVulkan.vulkan_local_context_query(
            self, GstVulkan.VULKAN_QUEUE_CONTEXT_TYPE_STR
        )
        if query is None:
            raise RuntimeError("no upstream element shares a Vulkan queue")
        # 1.24 introspection gives context_get_vulkan_queue no out argument
        structure = query.parse_context().get_structure()
        return structure.get_value(GstVulkan.VULKAN_QUEUE_CONTEXT_TYPE_STR)

    def draw_in_place(self, buf, frame_metadata):
        try:
            frame = backend.frameio.read_frame_for_drawing(
                buf, self.width, self.height, self.video_format
            )
            self.overlay_graphics.initialize(frame)
            self.do_post_process(frame_metadata)
            self.overlay_graphics.finalize()
        finally:
            backend.frameio.release_frames(buf)

    def do_post_process(self, frame_metadata):
        draw_frame_objects(
            self.overlay_graphics,
            frame_metadata,
            self.tracking_display if self.tracking else None,
        )


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("mp_overlay", Overlay)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_overlay' element will not be registered because a module is missing."
    )
