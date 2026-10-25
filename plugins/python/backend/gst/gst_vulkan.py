# GstVulkan through ctypes
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import ctypes
import ctypes.util

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from backend.gst.cuda_memory import GstMapInfo, GstMiniObject  # noqa: E402


def load_library(name):
    path = ctypes.util.find_library(name)
    if path is None:
        raise ImportError(f"the {name} library is not installed")
    return ctypes.CDLL(path)


def declare(library, signatures):
    for name, (result_type, argument_types) in signatures.items():
        function = getattr(library, name)
        function.restype = result_type
        function.argtypes = argument_types


# introspection types SPIR-V as a string and Vulkan handles as opaque records
gstreamer_library = load_library("gstreamer-1.0")
declare(
    gstreamer_library,
    {
        "gst_memory_map": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(GstMapInfo), ctypes.c_int],
        ),
        "gst_memory_unmap": (None, [ctypes.c_void_p, ctypes.POINTER(GstMapInfo)]),
        "gst_mini_object_ref": (ctypes.c_void_p, [ctypes.c_void_p]),
        "gst_mini_object_unref": (None, [ctypes.c_void_p]),
        "gst_buffer_append_memory": (None, [ctypes.c_void_p, ctypes.c_void_p]),
        "gst_query_parse_context_type": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)],
        ),
        "gst_query_set_context": (None, [ctypes.c_void_p, ctypes.c_void_p]),
        "gst_pad_add_probe": (
            ctypes.c_ulong,
            [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ],
        ),
    },
)
vulkan_library = load_library("gstvulkan-1.0")
declare(
    vulkan_library,
    {
        "gst_vulkan_create_shader": (
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p],
        ),
        "gst_vulkan_buffer_memory_alloc": (
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32],
        ),
        "gst_vulkan_full_screen_quad_set_shaders": (
            ctypes.c_int,
            [ctypes.c_void_p] * 3,
        ),
        "gst_vulkan_full_screen_quad_set_vertex_buffer": (
            ctypes.c_int,
            [ctypes.c_void_p] * 3,
        ),
        "gst_vulkan_full_screen_quad_set_index_buffer": (
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p],
        ),
        "gst_vulkan_image_memory_wrapped": (
            ctypes.c_void_p,
            [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_uint32,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ],
        ),
        "gst_vulkan_memory_find_memory_type_index_with_requirements": (
            ctypes.c_int,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
            ],
        ),
        "gst_vulkan_device_get_proc_address": (
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_char_p],
        ),
    },
)
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def gobject_address(gobject):
    return ctypes.pythonapi.PyCapsule_GetPointer(gobject.__gpointer__, None)


# pygobject hashes a boxed value by its pointer
def mini_object_address(mini_object):
    return hash(mini_object)


class GstPadProbeInfoLayout(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("id", ctypes.c_ulong),
        ("data", ctypes.c_void_p),
    ]


class GstQueryLayout(ctypes.Structure):
    _fields_ = [("mini_object", GstMiniObject), ("type", ctypes.c_int)]


PadProbeCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
)


# a query wrapped for python carries a second reference, which makes it read-only
def answer_context_queries(pad, context_for_type):
    @PadProbeCallback
    def answer(_pad, info, _user_data):
        query = GstPadProbeInfoLayout.from_address(info).data
        if GstQueryLayout.from_address(query).type != Gst.QueryType.CONTEXT:
            return Gst.PadProbeReturn.OK
        context_type = ctypes.c_char_p()
        gstreamer_library.gst_query_parse_context_type(
            query, ctypes.byref(context_type)
        )
        context = context_for_type(context_type.value.decode())
        if context is None:
            return Gst.PadProbeReturn.OK
        gstreamer_library.gst_query_set_context(query, mini_object_address(context))
        return Gst.PadProbeReturn.HANDLED

    gstreamer_library.gst_pad_add_probe(
        gobject_address(pad), Gst.PadProbeType.QUERY_UPSTREAM, answer, None, None
    )
    return answer


class GstObjectLayout(ctypes.Structure):
    _fields_ = [
        ("type_instance", ctypes.c_void_p),
        ("ref_count", ctypes.c_uint),
        ("qdata", ctypes.c_void_p),
        ("lock", ctypes.c_void_p),
        ("name", ctypes.c_char_p),
        ("parent", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("control_bindings", ctypes.c_void_p),
        ("control_rate", ctypes.c_uint64),
        ("last_sync", ctypes.c_uint64),
        ("reserved", ctypes.c_void_p),
    ]


class GstVulkanInstanceLayout(ctypes.Structure):
    _fields_ = [
        ("parent", GstObjectLayout),
        ("instance", ctypes.c_void_p),
        ("physical_devices", ctypes.POINTER(ctypes.c_void_p)),
        ("n_physical_devices", ctypes.c_uint32),
    ]


class GstVulkanDeviceLayout(ctypes.Structure):
    _fields_ = [
        ("parent", GstObjectLayout),
        ("instance", ctypes.c_void_p),
        ("physical_device", ctypes.c_void_p),
        ("device", ctypes.c_void_p),
    ]


def physical_device_handles(instance):
    layout = GstVulkanInstanceLayout.from_address(gobject_address(instance))
    return [layout.physical_devices[i] for i in range(layout.n_physical_devices)]


def device_handle(device):
    return GstVulkanDeviceLayout.from_address(gobject_address(device)).device
