# CUDA memory
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import ctypes
import sys
from dataclasses import dataclass
from functools import cache
from types import SimpleNamespace

from backend.cuda_driver import (
    call_cuda_driver,
    current_cuda_context,
    primary_cuda_context,
)
from utils.format_converter import packed_format_layout

WINDOWS = sys.platform == "win32"
GSTREAMER_LIBRARY = "gstreamer-1.0-0.dll" if WINDOWS else "libgstreamer-1.0.so.0"
GST_CUDA_LIBRARY = "gstcuda-1.0-0.dll" if WINDOWS else "libgstcuda-1.0.so.0"

GST_MAP_READ = 1
VIDEO_MEMORY = 0
MAPPED_MEMORY_COUNT = 1
GST_MAP_WRITE = 2
# the gi map override drops this flag, hence ctypes
GST_MAP_CUDA = 1 << 17
MAP_INFO_USER_DATA_SLOTS = 4
GST_PADDING = 4

CUDA_ARRAY_INTERFACE_VERSION = 3
UNSIGNED_BYTE_TYPESTR = "|u1"
COLOUR_CHANNEL_COUNT = 3

CU_EVENT_DISABLE_TIMING = 0x2
STREAM_WAIT_EVENT_FLAGS = 0


class GstMapInfo(ctypes.Structure):
    _fields_ = [
        ("memory", ctypes.c_void_p),
        ("flags", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("maxsize", ctypes.c_size_t),
        ("user_data", ctypes.c_void_p * MAP_INFO_USER_DATA_SLOTS),
        ("padding", ctypes.c_void_p * GST_PADDING),
    ]


class GstMiniObject(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_size_t),
        ("refcount", ctypes.c_int),
        ("lockstate", ctypes.c_int),
        ("flags", ctypes.c_uint),
        ("copy", ctypes.c_void_p),
        ("dispose", ctypes.c_void_p),
        ("free", ctypes.c_void_p),
        ("priv_uint", ctypes.c_uint),
        ("priv_pointer", ctypes.c_void_p),
    ]


class GstMemory(ctypes.Structure):
    _fields_ = [
        ("mini_object", GstMiniObject),
        ("allocator", ctypes.c_void_p),
        ("parent", ctypes.c_void_p),
        ("maxsize", ctypes.c_size_t),
        ("align", ctypes.c_size_t),
        ("offset", ctypes.c_size_t),
        ("size", ctypes.c_size_t),
    ]


# truncated after context, the only field read
class GstCudaMemory(ctypes.Structure):
    _fields_ = [("mem", GstMemory), ("context", ctypes.c_void_p)]


@cache
def gstreamer_library():
    library = ctypes.CDLL(GSTREAMER_LIBRARY)
    # mapping the whole buffer would merge an appended chunk into host memory
    library.gst_buffer_map_range.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(GstMapInfo),
        ctypes.c_int,
    ]
    library.gst_buffer_map_range.restype = ctypes.c_int
    library.gst_buffer_unmap.argtypes = [ctypes.c_void_p, ctypes.POINTER(GstMapInfo)]
    library.gst_buffer_unmap.restype = None
    return library


@cache
def gst_cuda_library():
    library = ctypes.CDLL(GST_CUDA_LIBRARY)
    library.gst_cuda_memory_sync.argtypes = [ctypes.c_void_p]
    library.gst_cuda_memory_sync.restype = None
    library.gst_cuda_memory_get_stream.argtypes = [ctypes.c_void_p]
    library.gst_cuda_memory_get_stream.restype = ctypes.c_void_p
    library.gst_cuda_stream_get_handle.argtypes = [ctypes.c_void_p]
    library.gst_cuda_stream_get_handle.restype = ctypes.c_void_p
    library.gst_cuda_context_get_handle.argtypes = [ctypes.c_void_p]
    library.gst_cuda_context_get_handle.restype = ctypes.c_void_p
    return library


def wait_for_the_producer_stream(memory):
    import torch

    gst_cuda = gst_cuda_library()
    producer_stream = gst_cuda.gst_cuda_memory_get_stream(memory)
    stream_handle = (
        gst_cuda.gst_cuda_stream_get_handle(producer_stream)
        if producer_stream
        else None
    )
    if not stream_handle:
        gst_cuda.gst_cuda_memory_sync(memory)
        return
    gst_context = GstCudaMemory.from_address(memory).context
    context_handle = gst_cuda.gst_cuda_context_get_handle(gst_context)
    event = ctypes.c_void_p()
    with current_cuda_context(context_handle):
        call_cuda_driver("cuEventCreate", ctypes.byref(event), CU_EVENT_DISABLE_TIMING)
    try:
        with current_cuda_context(context_handle):
            call_cuda_driver("cuEventRecord", event, stream_handle)
        torch_stream = torch.cuda.current_stream()
        with current_cuda_context(primary_cuda_context(torch_stream.device.index)):
            call_cuda_driver(
                "cuStreamWaitEvent",
                torch_stream.cuda_stream,
                event,
                STREAM_WAIT_EVENT_FLAGS,
            )
    finally:
        call_cuda_driver("cuEventDestroy_v2", event)


@cache
def gst_cuda_module():
    import gi

    try:
        gi.require_version("GstCuda", "1.0")
    except ValueError:
        return None
    from gi.repository import GstCuda

    return GstCuda


def is_cuda_buffer(buffer):
    gst_cuda = gst_cuda_module()
    # without the typelib the gi map downloads cuda memory to the host
    if gst_cuda is None:
        return False
    # an analytics chunk or a blob after the frame is system memory by design
    if buffer.n_memory() == 0:
        return False
    return gst_cuda.is_cuda_memory(buffer.peek_memory(VIDEO_MEMORY))


@dataclass
class DeviceFrameMap:
    buffer: object
    map_info: GstMapInfo
    tensor: object

    def unmap(self):
        gstreamer_library().gst_buffer_unmap(
            self.buffer.__ptr__(), ctypes.byref(self.map_info)
        )


def gst_video_module():
    import gi

    gi.require_version("GstVideo", "1.0")
    from gi.repository import GstVideo

    return GstVideo


# cuda memory pads rows to a pitch, the video meta carries it
def row_pitch(buffer, width, channels):
    video_meta = gst_video_module().buffer_get_video_meta(buffer)
    if video_meta is None:
        return width * channels
    return video_meta.stride[0]


def map_device_frame(buffer, height, width, channels, writable=False):
    import torch

    pitch = row_pitch(buffer, width, channels)
    map_flags = GST_MAP_READ | GST_MAP_CUDA
    if writable:
        map_flags |= GST_MAP_WRITE
    map_info = GstMapInfo()
    mapped = gstreamer_library().gst_buffer_map_range(
        buffer.__ptr__(),
        VIDEO_MEMORY,
        MAPPED_MEMORY_COUNT,
        ctypes.byref(map_info),
        map_flags,
    )
    if not mapped and writable:
        raise ValueError("drawing in place needs a writable buffer, this one is shared")
    if not mapped:
        raise RuntimeError("could not map the cuda memory of the buffer")
    device_map = DeviceFrameMap(buffer, map_info, None)
    try:
        # the producer may still be writing the frame on its own cuda stream
        wait_for_the_producer_stream(map_info.memory)
        array_interface = SimpleNamespace(
            __cuda_array_interface__={
                "shape": (height, width, channels),
                "typestr": UNSIGNED_BYTE_TYPESTR,
                "data": (map_info.data, False),
                "version": CUDA_ARRAY_INTERFACE_VERSION,
                "strides": (pitch, channels, 1),
            }
        )
        device_map.tensor = torch.as_tensor(array_interface, device="cuda")
    except BaseException:
        device_map.unmap()
        raise
    return device_map


def rgb_device_frame(frame, video_format):
    channel_count, rgb_slice = packed_format_layout(video_format)
    rgb_channels = range(channel_count)[rgb_slice[-1]]
    if rgb_channels.step == 1:
        return frame[..., rgb_channels.start : rgb_channels.stop]
    # torch refuses negative strides
    return frame[..., list(rgb_channels)]


def written_channels(frame_channel_count, video_format):
    channel_count, rgb_slice = packed_format_layout(video_format)
    leaves_alpha_as_is = frame_channel_count == COLOUR_CHANNEL_COUNT < channel_count
    if not leaves_alpha_as_is:
        return slice(None)
    first_colour_channel = min(range(channel_count)[rgb_slice[-1]])
    return slice(first_colour_channel, first_colour_channel + COLOUR_CHANNEL_COUNT)


def write_device_frame(buffer, frame):
    import torch

    gst_video = gst_video_module()
    video_meta = gst_video.buffer_get_video_meta(buffer)
    if video_meta is None:
        raise ValueError("a cuda buffer without a video meta has no known layout")
    height, width, frame_channel_count = frame.shape
    if (height, width) != (video_meta.height, video_meta.width):
        raise ValueError(
            f"a {width}x{height} frame does not fit a "
            f"{video_meta.width}x{video_meta.height} buffer"
        )
    video_format = gst_video.VideoFormat.to_string(video_meta.format)
    channel_count, _ = packed_format_layout(video_format)
    channels = written_channels(frame_channel_count, video_format)
    pixels = frame if torch.is_tensor(frame) else torch.from_numpy(frame)
    device_map = map_device_frame(buffer, height, width, channel_count, writable=True)
    try:
        device_map.tensor[..., channels].copy_(pixels)
        # downstream waits on the producer's stream, never on this one
        torch.cuda.current_stream().synchronize()
    finally:
        device_map.unmap()
