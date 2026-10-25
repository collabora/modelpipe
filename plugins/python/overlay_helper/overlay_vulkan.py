# overlay_vulkan.py
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import ctypes
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version("GstVulkan", "1.0")
from gi.repository import Gst, GstVideo, GstVulkan  # noqa: E402

from backend.gst.cuda_memory import GST_MAP_WRITE, GstMapInfo  # noqa: E402
from backend.gst.gst_vulkan import (  # noqa: E402
    answer_context_queries,
    gobject_address,
    gstreamer_library,
    vulkan_library,
)

from .overlay_utils_interface import (  # noqa: E402
    BOX_COLOR,
    BOX_LABEL_FONT_SIZE,
    BOX_LABEL_OFFSET,
    BOX_LINE_WIDTH,
    CLASSIFICATION_BACKGROUND_COLOR,
    CLASSIFICATION_BACKGROUND_MARGIN,
    CLASSIFICATION_FONT_SIZE,
    CLASSIFICATION_LABEL_WIDTH,
    CLASSIFICATION_LINE_HEIGHT,
    CLASSIFICATION_PADDING,
    CLASSIFICATION_TEXT_COLOR,
    TRACKING_POINT_SIZE,
    OverlayGraphics,
)

SHADER_DIRECTORY = Path(__file__).parent / "shaders"
# matches GLYPH_ATLAS_SIZE in shaders/overlay.vert
GLYPH_ATLAS_SIZE = 1024
GLYPH_PADDING = 1
# solid shapes sample the middle of this opaque block in the atlas corner
SOLID_BLOCK_SIZE = 4
SOLID_TEXEL = (SOLID_BLOCK_SIZE // 2, SOLID_BLOCK_SIZE // 2)
FULL_INTENSITY = 255
# the quad draws one triangle strip with 16 bit indices
MAX_VERTICES = 1 << 16
CORNERS_PER_QUAD = 4
VERTEX_BUFFER_USAGE = 0x80
INDEX_BUFFER_USAGE = 0x40
HOST_VISIBLE_COHERENT_MEMORY = 0x2 | 0x4
UPLOAD_TIMEOUT = 5 * Gst.SECOND
FENCE_TIMEOUT_SECONDS = 5
FENCE_POLL_SECONDS = 0.0005


def set_shaders(quad, device):
    shaders = []
    try:
        for name in ("overlay.vert.spv", "overlay.frag.spv"):
            code = (SHADER_DIRECTORY / name).read_bytes()
            shader = vulkan_library.gst_vulkan_create_shader(
                gobject_address(device), code, len(code), None
            )
            if not shader:
                raise RuntimeError(f"failed to create the {name} shader")
            shaders.append(shader)
        if not vulkan_library.gst_vulkan_full_screen_quad_set_shaders(
            gobject_address(quad), *shaders
        ):
            raise RuntimeError("failed to set the overlay shaders")
    finally:
        for shader in shaders:
            gstreamer_library.gst_mini_object_unref(shader)


def host_visible_memory(device, data, usage):
    memory = vulkan_library.gst_vulkan_buffer_memory_alloc(
        gobject_address(device), data.nbytes, usage, HOST_VISIBLE_COHERENT_MEMORY
    )
    if not memory:
        raise RuntimeError("failed to allocate a Vulkan buffer")
    info = GstMapInfo()
    if not gstreamer_library.gst_memory_map(memory, ctypes.byref(info), GST_MAP_WRITE):
        gstreamer_library.gst_mini_object_unref(memory)
        raise RuntimeError("failed to map a Vulkan buffer")
    ctypes.memmove(info.data, data.ctypes.data, data.nbytes)
    gstreamer_library.gst_memory_unmap(memory, ctypes.byref(info))
    return memory


def set_geometry(quad, device, vertices, indices):
    vertex_memory = host_visible_memory(device, vertices, VERTEX_BUFFER_USAGE)
    index_memory = host_visible_memory(device, indices, INDEX_BUFFER_USAGE)
    try:
        if not vulkan_library.gst_vulkan_full_screen_quad_set_vertex_buffer(
            gobject_address(quad), vertex_memory, None
        ):
            raise RuntimeError("failed to set the overlay vertices")
        if not vulkan_library.gst_vulkan_full_screen_quad_set_index_buffer(
            gobject_address(quad), index_memory, len(indices), None
        ):
            raise RuntimeError("failed to set the overlay indices")
    finally:
        gstreamer_library.gst_mini_object_unref(vertex_memory)
        gstreamer_library.gst_mini_object_unref(index_memory)


def vulkan_context(context_type, setter, value):
    context = Gst.Context.new(context_type, True)
    setter(context, value)
    return context


# vulkanupload frees its staging buffer on stop before its copy has to finish
@contextmanager
def uploaded_image(queue, pixels):
    height, width = pixels.shape[:2]
    pipeline = Gst.parse_launch(
        "appsrc name=source "
        f"caps=video/x-raw,format=RGBA,width={width},height={height},framerate=0/1 "
        "! vulkanupload ! video/x-raw(memory:VulkanImage) ! appsink name=sink"
    )
    pipeline.set_context(
        vulkan_context(
            GstVulkan.VULKAN_INSTANCE_CONTEXT_TYPE_STR,
            GstVulkan.context_set_vulkan_instance,
            queue.device.instance,
        )
    )
    neighbour_contexts = {
        GstVulkan.VULKAN_DEVICE_CONTEXT_TYPE_STR: vulkan_context(
            GstVulkan.VULKAN_DEVICE_CONTEXT_TYPE_STR,
            GstVulkan.context_set_vulkan_device,
            queue.device,
        ),
        GstVulkan.VULKAN_QUEUE_CONTEXT_TYPE_STR: vulkan_context(
            GstVulkan.VULKAN_QUEUE_CONTEXT_TYPE_STR,
            GstVulkan.context_set_vulkan_queue,
            queue,
        ),
    }
    source = pipeline.get_by_name("source")
    # vulkanupload takes its device and queue only from a neighbouring element
    context_probe = answer_context_queries(  # noqa: F841
        source.get_static_pad("src"), neighbour_contexts.get
    )
    pipeline.set_state(Gst.State.PLAYING)
    try:
        source.emit("push-buffer", Gst.Buffer.new_wrapped(pixels.tobytes()))
        sample = pipeline.get_by_name("sink").emit("try-pull-sample", UPLOAD_TIMEOUT)
        if sample is None:
            raise RuntimeError("vulkanupload produced no image")
        yield sample.get_buffer()
    finally:
        pipeline.set_state(Gst.State.NULL)


def wait_for_fence(fence):
    deadline = time.monotonic() + FENCE_TIMEOUT_SECONDS
    while not fence.is_signaled():
        if time.monotonic() > deadline:
            raise RuntimeError("the Vulkan overlay draw did not finish")
        time.sleep(FENCE_POLL_SECONDS)


@dataclass(frozen=True)
class Glyph:
    atlas_left: int
    atlas_top: int
    width: int
    height: int
    left_bearing: int
    top_from_baseline: int
    advance: float


class GlyphAtlas:
    def __init__(self):
        self.pixels = np.full(
            (GLYPH_ATLAS_SIZE, GLYPH_ATLAS_SIZE, 4), FULL_INTENSITY, np.uint8
        )
        self.pixels[..., 3] = 0
        self.pixels[:SOLID_BLOCK_SIZE, :SOLID_BLOCK_SIZE, 3] = FULL_INTENSITY
        self.glyphs = {}
        self.fonts = {}
        self.shelf_left = SOLID_BLOCK_SIZE + GLYPH_PADDING
        self.shelf_top = 0
        self.shelf_height = SOLID_BLOCK_SIZE
        self.changed = True

    def glyph(self, character, font_size):
        key = (character, font_size)
        if key not in self.glyphs:
            self.glyphs[key] = self.rasterize(character, font_size)
        return self.glyphs[key]

    def rasterize(self, character, font_size):
        from PIL import Image, ImageDraw

        font = self.font(font_size)
        left, top, right, bottom = font.getbbox(character, anchor="ls")
        width, height = right - left, bottom - top
        atlas_left, atlas_top = 0, 0
        if width > 0 and height > 0:
            atlas_left, atlas_top = self.reserve(width, height)
            image = Image.new("L", (width, height))
            ImageDraw.Draw(image).text(
                (-left, -top), character, fill=FULL_INTENSITY, font=font, anchor="ls"
            )
            self.pixels[
                atlas_top : atlas_top + height, atlas_left : atlas_left + width, 3
            ] = np.asarray(image)
            self.changed = True
        return Glyph(
            atlas_left,
            atlas_top,
            width,
            height,
            left,
            top,
            font.getlength(character),
        )

    # a right edge on the last column would unpack as the next atlas row
    def reserve(self, width, height):
        if self.shelf_left + width >= GLYPH_ATLAS_SIZE:
            self.shelf_top += self.shelf_height + GLYPH_PADDING
            self.shelf_left = 0
            self.shelf_height = 0
        if width >= GLYPH_ATLAS_SIZE or self.shelf_top + height > GLYPH_ATLAS_SIZE:
            raise RuntimeError("the glyph atlas is full")
        position = (self.shelf_left, self.shelf_top)
        self.shelf_left += width + GLYPH_PADDING
        self.shelf_height = max(self.shelf_height, height)
        return position

    def font(self, font_size):
        from PIL import ImageFont

        if font_size not in self.fonts:
            self.fonts[font_size] = ImageFont.load_default(size=font_size)
        return self.fonts[font_size]


def packed_colour(color, alpha):
    red, green, blue, alpha = (
        round(channel * FULL_INTENSITY)
        for channel in (color.r, color.g, color.b, alpha)
    )
    return (red * 256 + green, blue * 256 + alpha)


# repeating each quad's first and last corner joins the quads into one strip
def strip_indices(quad_count):
    first_corners = np.arange(quad_count, dtype=np.uint16)[:, None] * CORNERS_PER_QUAD
    return (first_corners + np.array([0, 0, 1, 2, 3, 3], np.uint16)).reshape(-1)


class VulkanOverlayGraphics(OverlayGraphics):
    def __init__(self, width, height, video_format, vulkan_queue):
        self.width = width
        self.height = height
        self.queue = vulkan_queue
        self.device = vulkan_queue.device
        self.quad = GstVulkan.VulkanFullScreenQuad.new(vulkan_queue)
        self.quad.enable_clear(False)
        self.quad.enable_blend(True)
        set_shaders(self.quad, self.device)
        atlas_info = GstVideo.VideoInfo.new()
        atlas_info.set_format(
            GstVideo.VideoFormat.RGBA, GLYPH_ATLAS_SIZE, GLYPH_ATLAS_SIZE
        )
        frame_info = GstVideo.VideoInfo.new()
        frame_info.set_format(
            GstVideo.VideoFormat.from_string(video_format), width, height
        )
        if not self.quad.set_info(atlas_info, frame_info):
            raise RuntimeError("failed to set the overlay video info")
        self.atlas = GlyphAtlas()
        self.target = None
        self.corners = []
        self.texels = []
        self.colours = []

    def initialize(self, buffer_data):
        if buffer_data.n_memory() == 0 or not GstVulkan.is_vulkan_image_memory(
            buffer_data.peek_memory(0)
        ):
            raise ValueError("Vulkan drawing needs a buffer of VulkanImage memory")
        self.target = buffer_data

    def finalize(self):
        target, self.target = self.target, None
        if not self.corners:
            return
        vertices = self.vertices()
        self.corners, self.texels, self.colours = [], [], []
        set_geometry(
            self.quad,
            self.device,
            vertices,
            strip_indices(len(vertices) // CORNERS_PER_QUAD),
        )
        self.quad.set_output_buffer(target)
        if not self.atlas.changed:
            self.quad.draw()
            return
        # the fence covers the upload's copy and every earlier draw on this queue
        with uploaded_image(self.queue, self.atlas.pixels) as atlas_image:
            self.quad.set_input_buffer(atlas_image)
            self.atlas.changed = False
            self.quad.draw()
            wait_for_fence(self.quad.get_last_fence())

    def vertices(self):
        corners = np.array(self.corners, np.float32).reshape(-1, 2)
        if len(corners) > MAX_VERTICES:
            raise ValueError(f"the overlay has more than {MAX_VERTICES} corners")
        positions = corners / (self.width, self.height) * 2 - 1
        texels = np.array(self.texels, np.float32).reshape(-1, 2)
        packed_texels = texels[:, 0] + texels[:, 1] * GLYPH_ATLAS_SIZE
        colours = np.repeat(
            np.array(self.colours, np.float32), CORNERS_PER_QUAD, axis=0
        )
        return np.column_stack([positions, packed_texels, colours]).astype(np.float32)

    def add_quad(self, corners, texels, color, alpha):
        self.corners.append(corners)
        self.texels.append(texels)
        self.colours.append(packed_colour(color, alpha))

    def add_rectangle(self, left, top, right, bottom, color, alpha):
        self.add_quad(
            ((left, top), (right, top), (left, bottom), (right, bottom)),
            (SOLID_TEXEL,) * CORNERS_PER_QUAD,
            color,
            alpha,
        )

    def draw_metadata(self, metadata, tracking_display):
        if tracking_display:
            for point in tracking_display.history:
                self.draw_tracking_point(
                    point["center"], point["color"], point["opacity"]
                )

        classifications = [d for d in metadata if d.get("type") == "classification"]
        for index, item in enumerate(classifications):
            self.draw_classification_label(index, item)

        for data in metadata:
            if data.get("type") == "classification":
                continue
            box = data["box"]
            self.draw_bounding_box(box)
            self.draw_text(
                data.get("label", ""),
                box["x1"],
                box["y1"] - BOX_LABEL_OFFSET,
                BOX_COLOR,
                BOX_LABEL_FONT_SIZE,
            )
            track_id = data.get("track_id")
            if tracking_display and track_id is not None:
                center = {
                    "x": (box["x1"] + box["x2"]) / 2,
                    "y": (box["y1"] + box["y2"]) / 2,
                }
                tracking_display.add_tracking_point(center, track_id)

    def draw_classification_label(self, index, item):
        x = CLASSIFICATION_PADDING
        y = CLASSIFICATION_PADDING + (index + 1) * CLASSIFICATION_LINE_HEIGHT
        left = x - CLASSIFICATION_BACKGROUND_MARGIN
        top = y - CLASSIFICATION_LINE_HEIGHT + CLASSIFICATION_BACKGROUND_MARGIN
        self.add_rectangle(
            left,
            top,
            left + CLASSIFICATION_LABEL_WIDTH,
            top + CLASSIFICATION_LINE_HEIGHT,
            CLASSIFICATION_BACKGROUND_COLOR,
            CLASSIFICATION_BACKGROUND_COLOR.a,
        )
        text = f"{item['label']}: {item['confidence']:.1%}"
        self.draw_text(text, x, y, CLASSIFICATION_TEXT_COLOR, CLASSIFICATION_FONT_SIZE)

    def draw_bounding_box(self, box):
        half_width = BOX_LINE_WIDTH // 2
        left = round(box["x1"]) - half_width
        top = round(box["y1"]) - half_width
        right = round(box["x2"]) + half_width
        bottom = round(box["y2"]) + half_width
        for slab in (
            (left, top, right, top + BOX_LINE_WIDTH),
            (left, bottom - BOX_LINE_WIDTH, right, bottom),
            (left, top, left + BOX_LINE_WIDTH, bottom),
            (right - BOX_LINE_WIDTH, top, right, bottom),
        ):
            self.add_rectangle(*slab, BOX_COLOR, BOX_COLOR.a)

    def draw_text(self, label, x, y, color, font_size):
        pen = float(round(x))
        baseline = round(y)
        for character in label:
            glyph = self.atlas.glyph(character, font_size)
            left = round(pen) + glyph.left_bearing
            top = baseline + glyph.top_from_baseline
            right, bottom = left + glyph.width, top + glyph.height
            atlas_right = glyph.atlas_left + glyph.width
            atlas_bottom = glyph.atlas_top + glyph.height
            if glyph.width > 0 and glyph.height > 0:
                self.add_quad(
                    ((left, top), (right, top), (left, bottom), (right, bottom)),
                    (
                        (glyph.atlas_left, glyph.atlas_top),
                        (atlas_right, glyph.atlas_top),
                        (glyph.atlas_left, atlas_bottom),
                        (atlas_right, atlas_bottom),
                    ),
                    color,
                    color.a,
                )
            pen += glyph.advance

    def draw_tracking_point(self, center, color, opacity):
        left = round(center["x"]) - TRACKING_POINT_SIZE // 2
        top = round(center["y"]) - TRACKING_POINT_SIZE // 2
        self.add_rectangle(
            left,
            top,
            left + TRACKING_POINT_SIZE,
            top + TRACKING_POINT_SIZE,
            color,
            opacity,
        )

    def draw_line(self, start, end, color, width):
        delta_x = end["x"] - start["x"]
        delta_y = end["y"] - start["y"]
        length = math.hypot(delta_x, delta_y)
        half_width = width / 2
        if length == 0:
            self.add_rectangle(
                start["x"] - half_width,
                start["y"] - half_width,
                start["x"] + half_width,
                start["y"] + half_width,
                color,
                color.a,
            )
            return
        normal_x = -delta_y / length * half_width
        normal_y = delta_x / length * half_width
        self.add_quad(
            (
                (start["x"] + normal_x, start["y"] + normal_y),
                (end["x"] + normal_x, end["y"] + normal_y),
                (start["x"] - normal_x, start["y"] - normal_y),
                (end["x"] - normal_x, end["y"] - normal_y),
            ),
            (SOLID_TEXEL,) * CORNERS_PER_QUAD,
            color,
            color.a,
        )
