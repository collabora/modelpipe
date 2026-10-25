# CudaToVulkan
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import ctypes
import time

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstBase, GstVideo  # noqa: E402

    gi.require_version("GstVulkan", "1.0")
    from gi.repository import GstVulkan  # noqa: E402

    from backend import post_error
    from backend.cuda_driver import call_cuda_driver, current_cuda_context
    from backend.gst.cuda_memory import (
        GST_MAP_CUDA,
        GST_MAP_READ,
        MAPPED_MEMORY_COUNT,
        VIDEO_MEMORY,
        GstCudaMemory,
        GstMapInfo,
        gst_cuda_library,
        gstreamer_library,
        row_pitch,
    )
    from backend.gst.cuda_vulkan_interop import (
        CHANNEL_COUNT,
        EXTERNAL_MEMORY_EXTENSIONS,
        SharedImage,
        cuda_device_uuid,
        current_cuda_device_uuid,
        vulkan_device_index_for_cuda,
    )
    from backend.gst import gst_vulkan
except (ImportError, ValueError) as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'mp_cudatovulkan' element will not be available. Error: {e}"
    )

CUDA_MEMORY = "memory:CUDAMemory"
VULKAN_IMAGE = "memory:VulkanImage"
FORMATS = "format=(string){ RGBA, BGRA }"
# the CUDA device whose GPU the Vulkan device is chosen to match
CUDA_DEVICE_ORDINAL = 0
SHARED_IMAGE_LIMIT = 16
FREE_IMAGE_TIMEOUT_SECONDS = 5
FREE_IMAGE_POLL_SECONDS = 0.0005


class CudaToVulkan(GstBase.BaseTransform):
    __gstmetadata__ = (
        "CudaToVulkan",
        "Filter/Video",
        "Copies CUDA frames into Vulkan images on the same GPU",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    if backend.BACKEND == "gst":
        __gsttemplates__ = (
            Gst.PadTemplate.new(
                "src",
                Gst.PadDirection.SRC,
                Gst.PadPresence.ALWAYS,
                Gst.Caps.from_string(f"video/x-raw({VULKAN_IMAGE}), {FORMATS}"),
            ),
            Gst.PadTemplate.new(
                "sink",
                Gst.PadDirection.SINK,
                Gst.PadPresence.ALWAYS,
                Gst.Caps.from_string(f"video/x-raw({CUDA_MEMORY}), {FORMATS}"),
            ),
        )

    def __init__(self):
        super().__init__()
        self.instance = None
        self.device = None
        self.cuda_uuid = None
        self.cuda_context = None
        self.video_info = None
        self.images = []
        # downstream Vulkan elements ask for a device before the first frame arrives
        self.context_probe = gst_vulkan.answer_context_queries(
            self.get_static_pad("src"), self.vulkan_context
        )

    def do_transform_caps(self, direction, caps, filter_caps):
        feature = VULKAN_IMAGE if direction == Gst.PadDirection.SINK else CUDA_MEMORY
        transformed = Gst.Caps.new_empty()
        for index in range(caps.get_size()):
            transformed.append_structure_full(
                caps.get_structure(index).copy(), Gst.CapsFeatures.new_single(feature)
            )
        if filter_caps is not None:
            transformed = filter_caps.intersect_full(
                transformed, Gst.CapsIntersectMode.FIRST
            )
        return transformed

    def do_set_caps(self, incaps, outcaps):
        self.video_info = GstVideo.VideoInfo.new_from_caps(incaps)
        return True

    def vulkan_context(self, context_type):
        setters = {
            GstVulkan.VULKAN_INSTANCE_CONTEXT_TYPE_STR: (
                GstVulkan.context_set_vulkan_instance,
                lambda: self.instance,
            ),
            GstVulkan.VULKAN_DEVICE_CONTEXT_TYPE_STR: (
                GstVulkan.context_set_vulkan_device,
                lambda: self.device,
            ),
        }
        if context_type not in setters:
            return None
        try:
            self.open_vulkan_device()
        except Exception as exception:
            post_error(self, "no Vulkan device shares memory with CUDA", exception)
            return None
        setter, value = setters[context_type]
        context = Gst.Context.new(context_type, True)
        setter(context, value())
        return context

    def open_vulkan_device(self):
        if self.device is not None:
            return
        self.cuda_uuid = cuda_device_uuid(CUDA_DEVICE_ORDINAL)
        instance = GstVulkan.VulkanInstance.new()
        instance.open()
        device = GstVulkan.VulkanDevice.new_with_index(
            instance, vulkan_device_index_for_cuda(instance, self.cuda_uuid)
        )
        for extension in EXTERNAL_MEMORY_EXTENSIONS:
            if not device.enable_extension(extension):
                raise RuntimeError(f"the Vulkan device lacks {extension}")
        device.open()
        self.instance = instance
        self.device = device

    def do_prepare_output_buffer(self, inbuf):
        outbuf = Gst.Buffer.new()
        GstBase.BaseTransform.do_copy_metadata(self, inbuf, outbuf)
        return Gst.FlowReturn.OK, outbuf

    def do_transform(self, inbuf, outbuf):
        try:
            image = self.copy_into_shared_image(inbuf)
        except Exception as exception:
            post_error(self, "copying a CUDA frame into Vulkan failed", exception)
            return Gst.FlowReturn.ERROR
        gst_vulkan.gstreamer_library.gst_buffer_append_memory(
            gst_vulkan.mini_object_address(outbuf), image.new_reference()
        )
        return Gst.FlowReturn.OK

    def copy_into_shared_image(self, inbuf):
        if self.device is None:
            raise RuntimeError("no downstream Vulkan element asked for a device")
        map_info = GstMapInfo()
        if not gstreamer_library().gst_buffer_map_range(
            gst_vulkan.mini_object_address(inbuf),
            VIDEO_MEMORY,
            MAPPED_MEMORY_COUNT,
            ctypes.byref(map_info),
            GST_MAP_READ | GST_MAP_CUDA,
        ):
            raise RuntimeError("could not map the CUDA memory of the frame")
        try:
            gst_cuda = gst_cuda_library()
            cuda_context = gst_cuda.gst_cuda_context_get_handle(
                GstCudaMemory.from_address(map_info.memory).context
            )
            producer_stream = gst_cuda.gst_cuda_memory_get_stream(map_info.memory)
            stream = (
                gst_cuda.gst_cuda_stream_get_handle(producer_stream)
                if producer_stream
                else None
            )
            with current_cuda_context(cuda_context):
                self.check_cuda_context(cuda_context)
                image = self.free_image()
                # the copy queues behind the producer's writes on its stream
                image.copy_from(
                    map_info.data,
                    row_pitch(inbuf, self.video_info.width, CHANNEL_COUNT),
                    stream,
                )
                call_cuda_driver("cuStreamSynchronize", stream)
            return image
        finally:
            gstreamer_library().gst_buffer_unmap(
                gst_vulkan.mini_object_address(inbuf), ctypes.byref(map_info)
            )

    def check_cuda_context(self, cuda_context):
        if self.cuda_context is None:
            if current_cuda_device_uuid() != self.cuda_uuid:
                raise RuntimeError(
                    f"the frames are not on CUDA device {CUDA_DEVICE_ORDINAL}"
                )
            self.cuda_context = cuda_context
        if cuda_context != self.cuda_context:
            raise RuntimeError("the frames moved to another CUDA context")

    def free_image(self):
        deadline = time.monotonic() + FREE_IMAGE_TIMEOUT_SECONDS
        while True:
            for image in self.images:
                if not image.is_in_use():
                    return image
            if len(self.images) < SHARED_IMAGE_LIMIT:
                self.images.append(
                    SharedImage(
                        self.device,
                        self.cuda_context,
                        self.video_info.width,
                        self.video_info.height,
                        self.video_info.finfo.name,
                    )
                )
                return self.images[-1]
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"downstream still holds all {SHARED_IMAGE_LIMIT} shared images"
                )
            time.sleep(FREE_IMAGE_POLL_SECONDS)

    def do_stop(self):
        for image in self.images:
            image.release()
        self.images = []
        self.cuda_context = None
        return True


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_cudatovulkan", CudaToVulkan
    )
