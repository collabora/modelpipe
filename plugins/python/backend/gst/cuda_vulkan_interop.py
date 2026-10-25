# CUDA and Vulkan sharing one image
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import ctypes

from backend.cuda_driver import call_cuda_driver, current_cuda_context
from backend.gst.cuda_memory import GstMiniObject
from backend.gst.gst_vulkan import (
    declare,
    device_handle,
    gobject_address,
    gstreamer_library,
    load_library,
    physical_device_handles,
    vulkan_library,
)

EXTERNAL_MEMORY_EXTENSIONS = ("VK_KHR_external_memory", "VK_KHR_external_memory_fd")
VULKAN_FORMATS = {"RGBA": 37, "BGRA": 44}
CHANNEL_COUNT = 4
UUID_SIZE = 16
VK_SUCCESS = 0
STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5
STRUCTURE_TYPE_IMAGE_CREATE_INFO = 14
STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001
STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES = 1000071004
STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO = 1000072001
STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO = 1000072002
STRUCTURE_TYPE_MEMORY_GET_FD_INFO = 1000074002
STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO = 1000127001
IMAGE_TYPE_2D = 1
IMAGE_TILING_OPTIMAL = 0
SAMPLE_COUNT_1 = 1
SHARING_MODE_EXCLUSIVE = 0
IMAGE_LAYOUT_UNDEFINED = 0
IMAGE_USAGE = 0x1 | 0x2 | 0x4 | 0x10
MEMORY_PROPERTY_DEVICE_LOCAL = 0x1
HANDLE_TYPE_OPAQUE_FD = 0x1
# sizeof (VkPhysicalDeviceProperties)
PHYSICAL_DEVICE_PROPERTIES_SIZE = 824
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1
CUDA_EXTERNAL_MEMORY_DEDICATED = 1
CU_AD_FORMAT_UNSIGNED_INT8 = 1
# CUDA has to be told the image is a colour attachment
CUDA_ARRAY3D_COLOR_ATTACHMENT = 0x20
CU_MEMORYTYPE_DEVICE = 2
CU_MEMORYTYPE_ARRAY = 3
ONLY_OWNER = 1


class VkExtent3D(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("depth", ctypes.c_uint32),
    ]


class VkImageCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("imageType", ctypes.c_int),
        ("format", ctypes.c_int),
        ("extent", VkExtent3D),
        ("mipLevels", ctypes.c_uint32),
        ("arrayLayers", ctypes.c_uint32),
        ("samples", ctypes.c_int),
        ("tiling", ctypes.c_int),
        ("usage", ctypes.c_uint32),
        ("sharingMode", ctypes.c_int),
        ("queueFamilyIndexCount", ctypes.c_uint32),
        ("pQueueFamilyIndices", ctypes.c_void_p),
        ("initialLayout", ctypes.c_int),
    ]


class VkExternalMemoryInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("handleTypes", ctypes.c_uint32),
    ]


class VkMemoryDedicatedAllocateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("image", ctypes.c_uint64),
        ("buffer", ctypes.c_uint64),
    ]


class VkMemoryAllocateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("allocationSize", ctypes.c_uint64),
        ("memoryTypeIndex", ctypes.c_uint32),
    ]


class VkMemoryRequirements(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("alignment", ctypes.c_uint64),
        ("memoryTypeBits", ctypes.c_uint32),
    ]


class VkMemoryGetFdInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("memory", ctypes.c_uint64),
        ("handleType", ctypes.c_uint32),
    ]


class VkPhysicalDeviceIDProperties(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("deviceUUID", ctypes.c_ubyte * UUID_SIZE),
        ("driverUUID", ctypes.c_ubyte * UUID_SIZE),
        ("deviceLUID", ctypes.c_ubyte * 8),
        ("deviceNodeMask", ctypes.c_uint32),
        ("deviceLUIDValid", ctypes.c_uint32),
    ]


class VkPhysicalDeviceProperties2(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("properties", ctypes.c_ubyte * PHYSICAL_DEVICE_PROPERTIES_SIZE),
    ]


class CudaExternalMemoryHandle(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("win32", ctypes.c_void_p * 2)]


class CudaExternalMemoryHandleDescription(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", CudaExternalMemoryHandle),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class CudaArray3DDescriptor(ctypes.Structure):
    _fields_ = [
        ("Width", ctypes.c_size_t),
        ("Height", ctypes.c_size_t),
        ("Depth", ctypes.c_size_t),
        ("Format", ctypes.c_int),
        ("NumChannels", ctypes.c_uint),
        ("Flags", ctypes.c_uint),
    ]


class CudaExternalMemoryMipmappedArrayDescription(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_ulonglong),
        ("arrayDesc", CudaArray3DDescriptor),
        ("numLevels", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class CudaMemcpy2D(ctypes.Structure):
    _fields_ = [
        ("srcXInBytes", ctypes.c_size_t),
        ("srcY", ctypes.c_size_t),
        ("srcMemoryType", ctypes.c_int),
        ("srcHost", ctypes.c_void_p),
        ("srcDevice", ctypes.c_uint64),
        ("srcArray", ctypes.c_void_p),
        ("srcPitch", ctypes.c_size_t),
        ("dstXInBytes", ctypes.c_size_t),
        ("dstY", ctypes.c_size_t),
        ("dstMemoryType", ctypes.c_int),
        ("dstHost", ctypes.c_void_p),
        ("dstDevice", ctypes.c_uint64),
        ("dstArray", ctypes.c_void_p),
        ("dstPitch", ctypes.c_size_t),
        ("WidthInBytes", ctypes.c_size_t),
        ("Height", ctypes.c_size_t),
    ]


vulkan_loader = load_library("vulkan")
declare(
    vulkan_loader,
    {
        "vkGetPhysicalDeviceProperties2": (
            None,
            [ctypes.c_void_p, ctypes.POINTER(VkPhysicalDeviceProperties2)],
        ),
        "vkCreateImage": (
            ctypes.c_int,
            [
                ctypes.c_void_p,
                ctypes.POINTER(VkImageCreateInfo),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
            ],
        ),
        "vkDestroyImage": (None, [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]),
        "vkGetImageMemoryRequirements": (
            None,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(VkMemoryRequirements)],
        ),
        "vkAllocateMemory": (
            ctypes.c_int,
            [
                ctypes.c_void_p,
                ctypes.POINTER(VkMemoryAllocateInfo),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
            ],
        ),
        "vkFreeMemory": (None, [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]),
        "vkBindImageMemory": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64],
        ),
    },
)
GetMemoryFd = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(VkMemoryGetFdInfo),
    ctypes.POINTER(ctypes.c_int),
)


DestroyNotify = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
images_awaiting_release = {}


@DestroyNotify
def release_freed_image(image_key):
    images_awaiting_release.pop(image_key).destroy_resources()


def check_vulkan(function_name, result):
    if result != VK_SUCCESS:
        raise RuntimeError(f"{function_name} failed with VkResult {result}")


def cuda_device_uuid(device_ordinal):
    call_cuda_driver("cuInit", 0)
    device = ctypes.c_int()
    call_cuda_driver("cuDeviceGet", ctypes.byref(device), device_ordinal)
    uuid = ctypes.create_string_buffer(UUID_SIZE)
    call_cuda_driver("cuDeviceGetUuid_v2", uuid, device)
    return uuid.raw


def current_cuda_device_uuid():
    device = ctypes.c_int()
    call_cuda_driver("cuCtxGetDevice", ctypes.byref(device))
    uuid = ctypes.create_string_buffer(UUID_SIZE)
    call_cuda_driver("cuDeviceGetUuid_v2", uuid, device)
    return uuid.raw


def vulkan_device_uuid(physical_device):
    identity = VkPhysicalDeviceIDProperties(
        sType=STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES
    )
    properties = VkPhysicalDeviceProperties2(
        sType=STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        pNext=ctypes.cast(ctypes.pointer(identity), ctypes.c_void_p),
    )
    vulkan_loader.vkGetPhysicalDeviceProperties2(
        physical_device, ctypes.byref(properties)
    )
    return bytes(identity.deviceUUID)


def vulkan_device_index_for_cuda(instance, cuda_uuid):
    for index, physical_device in enumerate(physical_device_handles(instance)):
        if vulkan_device_uuid(physical_device) == cuda_uuid:
            return index
    raise RuntimeError("no Vulkan device is the GPU of CUDA device 0")


class SharedImage:
    def __init__(self, device, cuda_context, width, height, video_format):
        self.device = device
        self.cuda_context = cuda_context
        self.width = width
        self.height = height
        self.vulkan_device = device_handle(device)
        self.image = ctypes.c_uint64()
        self.device_memory = ctypes.c_uint64()
        self.external_memory = ctypes.c_void_p()
        self.mipmapped_array = ctypes.c_void_p()
        self.array = ctypes.c_void_p()
        self.memory = None
        try:
            size = self.create_image(VULKAN_FORMATS[video_format])
            self.import_into_cuda(self.export_fd(), size)
            self.memory = vulkan_library.gst_vulkan_image_memory_wrapped(
                gobject_address(device),
                self.image.value,
                VULKAN_FORMATS[video_format],
                width,
                height,
                IMAGE_TILING_OPTIMAL,
                IMAGE_USAGE,
                id(self),
                release_freed_image,
            )
            if not self.memory:
                raise RuntimeError("failed to wrap the shared Vulkan image")
        except BaseException:
            self.destroy_resources()
            raise
        images_awaiting_release[id(self)] = self

    def create_image(self, vulkan_format):
        external = VkExternalMemoryInfo(
            sType=STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            handleTypes=HANDLE_TYPE_OPAQUE_FD,
        )
        create_info = VkImageCreateInfo(
            sType=STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            pNext=ctypes.cast(ctypes.pointer(external), ctypes.c_void_p),
            imageType=IMAGE_TYPE_2D,
            format=vulkan_format,
            extent=VkExtent3D(self.width, self.height, 1),
            mipLevels=1,
            arrayLayers=1,
            samples=SAMPLE_COUNT_1,
            tiling=IMAGE_TILING_OPTIMAL,
            usage=IMAGE_USAGE,
            sharingMode=SHARING_MODE_EXCLUSIVE,
            initialLayout=IMAGE_LAYOUT_UNDEFINED,
        )
        check_vulkan(
            "vkCreateImage",
            vulkan_loader.vkCreateImage(
                self.vulkan_device,
                ctypes.byref(create_info),
                None,
                ctypes.byref(self.image),
            ),
        )
        requirements = VkMemoryRequirements()
        vulkan_loader.vkGetImageMemoryRequirements(
            self.vulkan_device, self.image, ctypes.byref(requirements)
        )
        memory_type = ctypes.c_uint32()
        if not vulkan_library.gst_vulkan_memory_find_memory_type_index_with_requirements(
            gobject_address(self.device),
            ctypes.addressof(requirements),
            MEMORY_PROPERTY_DEVICE_LOCAL,
            ctypes.byref(memory_type),
        ):
            raise RuntimeError("the Vulkan device has no memory for a shared image")
        # CUDA imports a Vulkan image only from a dedicated allocation
        dedicated = VkMemoryDedicatedAllocateInfo(
            sType=STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
            image=self.image.value,
        )
        export = VkExternalMemoryInfo(
            sType=STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            pNext=ctypes.cast(ctypes.pointer(dedicated), ctypes.c_void_p),
            handleTypes=HANDLE_TYPE_OPAQUE_FD,
        )
        allocate_info = VkMemoryAllocateInfo(
            sType=STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext=ctypes.cast(ctypes.pointer(export), ctypes.c_void_p),
            allocationSize=requirements.size,
            memoryTypeIndex=memory_type.value,
        )
        check_vulkan(
            "vkAllocateMemory",
            vulkan_loader.vkAllocateMemory(
                self.vulkan_device,
                ctypes.byref(allocate_info),
                None,
                ctypes.byref(self.device_memory),
            ),
        )
        check_vulkan(
            "vkBindImageMemory",
            vulkan_loader.vkBindImageMemory(
                self.vulkan_device, self.image, self.device_memory, 0
            ),
        )
        return requirements.size

    def export_fd(self):
        address = vulkan_library.gst_vulkan_device_get_proc_address(
            gobject_address(self.device), b"vkGetMemoryFdKHR"
        )
        if not address:
            raise RuntimeError("the Vulkan device cannot export memory")
        fd_info = VkMemoryGetFdInfo(
            sType=STRUCTURE_TYPE_MEMORY_GET_FD_INFO,
            memory=self.device_memory.value,
            handleType=HANDLE_TYPE_OPAQUE_FD,
        )
        fd = ctypes.c_int(-1)
        check_vulkan(
            "vkGetMemoryFdKHR",
            GetMemoryFd(address)(
                self.vulkan_device, ctypes.byref(fd_info), ctypes.byref(fd)
            ),
        )
        return fd.value

    # CUDA closes the file descriptor after a successful import
    def import_into_cuda(self, fd, size):
        handle_description = CudaExternalMemoryHandleDescription(
            type=CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            size=size,
            flags=CUDA_EXTERNAL_MEMORY_DEDICATED,
        )
        handle_description.handle.fd = fd
        array_description = CudaExternalMemoryMipmappedArrayDescription(
            arrayDesc=CudaArray3DDescriptor(
                Width=self.width,
                Height=self.height,
                Format=CU_AD_FORMAT_UNSIGNED_INT8,
                NumChannels=CHANNEL_COUNT,
                Flags=CUDA_ARRAY3D_COLOR_ATTACHMENT,
            ),
            numLevels=1,
        )
        with current_cuda_context(self.cuda_context):
            call_cuda_driver(
                "cuImportExternalMemory",
                ctypes.byref(self.external_memory),
                ctypes.addressof(handle_description),
            )
            call_cuda_driver(
                "cuExternalMemoryGetMappedMipmappedArray",
                ctypes.byref(self.mipmapped_array),
                self.external_memory,
                ctypes.addressof(array_description),
            )
            call_cuda_driver(
                "cuMipmappedArrayGetLevel",
                ctypes.byref(self.array),
                self.mipmapped_array,
                0,
            )

    # a reference held downstream, by a buffer or an image view, means Vulkan may still read it
    def is_in_use(self):
        return GstMiniObject.from_address(self.memory).refcount > ONLY_OWNER

    def copy_from(self, source_pointer, source_pitch, stream):
        copy = CudaMemcpy2D(
            srcMemoryType=CU_MEMORYTYPE_DEVICE,
            srcDevice=source_pointer,
            srcPitch=source_pitch,
            dstMemoryType=CU_MEMORYTYPE_ARRAY,
            dstArray=self.array,
            WidthInBytes=self.width * CHANNEL_COUNT,
            Height=self.height,
        )
        call_cuda_driver("cuMemcpy2DAsync_v2", ctypes.addressof(copy), stream)

    def new_reference(self):
        return gstreamer_library.gst_mini_object_ref(self.memory)

    # the image is freed when the last reference to the memory is dropped
    def release(self):
        gstreamer_library.gst_mini_object_unref(self.memory)

    def destroy_resources(self):
        with current_cuda_context(self.cuda_context):
            if self.mipmapped_array:
                call_cuda_driver("cuMipmappedArrayDestroy", self.mipmapped_array)
            if self.external_memory:
                call_cuda_driver("cuDestroyExternalMemory", self.external_memory)
        if self.image:
            vulkan_loader.vkDestroyImage(self.vulkan_device, self.image, None)
        if self.device_memory:
            vulkan_loader.vkFreeMemory(self.vulkan_device, self.device_memory, None)
