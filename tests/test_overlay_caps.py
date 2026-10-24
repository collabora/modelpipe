import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)


def test_overlay_does_not_advertise_the_incomplete_vulkan_renderer():
    from overlay import VIDEO_FORMATS

    assert "VulkanMemory" not in VIDEO_FORMATS


def test_anonymizer_advertises_rgba_system_and_cuda_frames():
    from anonymize import Anonymize

    caps = Anonymize.__gsttemplates__[0].get_caps().to_string()

    assert "video/x-raw(memory:CUDAMemory), format=(string){ RGBA }" in caps
    assert "video/x-raw, format=(string){ RGBA }" in caps


def test_capability_templates_do_not_replace_extra_pads():
    from base_caption import BaseCaption

    assert [template.name_template for template in BaseCaption.__gsttemplates__] == [
        "text_src"
    ]
