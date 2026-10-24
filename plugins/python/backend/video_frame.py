# Video frame capabilities
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from dataclasses import dataclass

SYSTEM_MEMORY = "system"
CUDA_MEMORY = "cuda"
ANY_PIXEL_FORMAT = "*"
PACKED_PIXEL_FORMATS = ("RGB", "RGBA", "ARGB", "BGRA", "ABGR")
DRAWING_PIXEL_FORMATS = ("RGBA", "ARGB", "BGRA", "ABGR")
RGBA_PIXEL_FORMATS = ("RGBA",)
CALL_COMPLETION = "call-completion"


@dataclass(frozen=True)
class VideoFrameCapabilities:
    memory_types: tuple[str, ...]
    pixel_formats: tuple[str, ...]
    reads_pixels: bool
    writes_pixels: bool

    @property
    def accepts_cuda(self):
        return CUDA_MEMORY in self.memory_types

    @property
    def synchronization(self):
        return CALL_COMPLETION


SYSTEM_INFERENCE_FRAME_CAPABILITIES = VideoFrameCapabilities(
    (SYSTEM_MEMORY,), PACKED_PIXEL_FORMATS, True, False
)
CUDA_INFERENCE_FRAME_CAPABILITIES = VideoFrameCapabilities(
    (SYSTEM_MEMORY, CUDA_MEMORY), PACKED_PIXEL_FORMATS, True, False
)
SYSTEM_READ_FRAME_CAPABILITIES = VideoFrameCapabilities(
    (SYSTEM_MEMORY,), PACKED_PIXEL_FORMATS, True, False
)
SYSTEM_DRAW_FRAME_CAPABILITIES = VideoFrameCapabilities(
    (SYSTEM_MEMORY,), (ANY_PIXEL_FORMAT,), True, True
)
CUDA_DRAW_FRAME_CAPABILITIES = VideoFrameCapabilities(
    (SYSTEM_MEMORY, CUDA_MEMORY), DRAWING_PIXEL_FORMATS, True, True
)
CUDA_RGBA_DRAW_FRAME_CAPABILITIES = VideoFrameCapabilities(
    (SYSTEM_MEMORY, CUDA_MEMORY), RGBA_PIXEL_FORMATS, True, True
)
METADATA_FRAME_CAPABILITIES = VideoFrameCapabilities(
    (SYSTEM_MEMORY, CUDA_MEMORY), (ANY_PIXEL_FORMAT,), False, False
)
