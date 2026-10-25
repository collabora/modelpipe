import asyncio
import os
import sys
import types
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
PLUGIN_DIR = BASE_DIR / "plugins" / "python"

# the plugin loader reads GST_PLUGIN_PATH at Gst.init
os.environ["GST_PLUGIN_PATH"] = str(BASE_DIR / "plugins")
sys.path.insert(0, str(PLUGIN_DIR))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

from base_tts import BYTES_PER_SAMPLE  # noqa: E402

pytestmark = pytest.mark.skipif(
    Gst.ElementFactory.find("mp_whisperspeechtts") is None,
    reason="the python plugin loader did not register the modelpipe elements",
)

TRANSCRIPT = "the quick brown fox"
GENERATED_SAMPLES = 2400


# whisperspeech vocodes to a float32 tensor of shape (1, n) on the model device
@pytest.fixture(params=["batched", "flat"], ids=["batched", "flat"])
def element(request, monkeypatch):
    torch = pytest.importorskip("torch")
    batched = request.param == "batched"

    class Pipeline:
        def __init__(self, **kwargs):
            pass

        def generate(self, text, lang=None):
            samples = torch.linspace(-1.0, 1.0, GENERATED_SAMPLES, dtype=torch.float32)
            return samples.unsqueeze(0) if batched else samples

    package = types.ModuleType("whisperspeech")
    pipeline_module = types.ModuleType("whisperspeech.pipeline")
    pipeline_module.Pipeline = Pipeline
    package.pipeline = pipeline_module
    monkeypatch.setitem(sys.modules, "whisperspeech", package)
    monkeypatch.setitem(sys.modules, "whisperspeech.pipeline", pipeline_module)

    # a ci runner has no cuda
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)

    from whisperspeechtts import WhisperSpeechTTS

    element = WhisperSpeechTTS()
    element.device = "cuda"
    element.do_load_model()
    assert element.get_model() is not None
    return element


def test_load_on_cpu_names_cuda():
    from whisperspeechtts import WhisperSpeechTTS

    element = WhisperSpeechTTS()
    element.device = "cpu"
    with pytest.raises(ValueError, match="whisperspeech runs on cuda only"):
        element.do_load_model()


def test_generated_speech_is_mono_float32(element):
    audio = element.do_generate_speech(TRANSCRIPT)

    assert audio.ndim == 1, f"do_generate_speech returned shape {audio.shape}"
    assert audio.dtype.name == "float32"
    assert len(audio) == GENERATED_SAMPLES


def test_generated_speech_survives_soundfile(element):
    pytest.importorskip("soundfile")
    audio = asyncio.run(element.process_transcript(TRANSCRIPT))

    assert audio is not None, "soundfile rejected what do_generate_speech returned"
    assert len(audio.tobytes()) == GENERATED_SAMPLES * BYTES_PER_SAMPLE
    assert audio.any(), "the audio path produced silence"
