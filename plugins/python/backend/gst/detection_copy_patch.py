# Detection copy patch
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import ctypes
import os
import sys

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GLib  # noqa: E402

FIRST_BROKEN_VERSION = (1, 28, 0)
FIRST_FIXED_VERSION = (1, 28, 3)
PROT_READ = 1
PROT_WRITE = 2

TransformFunction = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_void_p,
)

replacement_transforms = []


def detection_copy_is_broken():
    version = tuple(Gst.version()[:3])
    return (
        sys.platform == "linux"
        and FIRST_BROKEN_VERSION <= version < FIRST_FIXED_VERSION
    )


# GStreamer 1.28.0 to 1.28.2 drop every object detection from a copied buffer's relation meta
def keep_detections_on_buffer_copy():
    if not detection_copy_is_broken():
        return
    analytics_library = ctypes.CDLL("libgstanalytics-1.0.so.0")
    analytics_library.gst_analytics_od_mtd_get_mtd_type.restype = ctypes.c_void_p
    # the object detection type is the address of its GstAnalyticsMtdImpl
    implementation = analytics_library.gst_analytics_od_mtd_get_mtd_type()
    transform_slot = ctypes.c_void_p.from_address(
        implementation + ctypes.sizeof(ctypes.c_void_p)
    )
    original_transform = TransformFunction(transform_slot.value)
    copy_quark = GLib.quark_from_string("gst-copy")

    def transform(transform_buffer, transform_mtd, buffer, transform_type, data):
        if transform_type == copy_quark:
            return True
        return original_transform(
            transform_buffer, transform_mtd, buffer, transform_type, data
        )

    replacement = TransformFunction(transform)
    replacement_transforms.append(replacement)
    make_writable(ctypes.addressof(transform_slot))
    transform_slot.value = ctypes.cast(replacement, ctypes.c_void_p).value


def make_writable(address):
    libc = ctypes.CDLL(None, use_errno=True)
    page_size = os.sysconf("SC_PAGESIZE")
    page = ctypes.c_void_p(address & ~(page_size - 1))
    if libc.mprotect(page, ctypes.c_size_t(page_size), PROT_READ | PROT_WRITE):
        error = ctypes.get_errno()
        raise OSError(error, f"mprotect failed: {os.strerror(error)}")
