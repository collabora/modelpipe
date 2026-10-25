import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
PLUGIN_DIR = BASE_DIR / "plugins" / "python"

# the plugin loader reads GST_PLUGIN_PATH at Gst.init
os.environ["GST_PLUGIN_PATH"] = str(BASE_DIR / "plugins")
sys.path.insert(0, str(PLUGIN_DIR))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import GObject, Gst, GstBase  # noqa: E402

Gst.init(None)

from embedding_index import EmbeddingIndex  # noqa: E402
from embeddingsink import decode_embedding  # noqa: E402

pytestmark = pytest.mark.skipif(
    Gst.ElementFactory.find("mp_embeddingsink") is None,
    reason="the python plugin loader did not register the modelpipe elements",
)

EMBEDDING_HEADER = b"GST-EMBEDDING:"
MODEL_NAME = "openai/clip-vit-base-patch32"
SOURCE_ID = "camera"
DIMENSION = 3
FRAMES = 3
FRAMERATE = 30
PIPELINE_TIMEOUT = 30 * Gst.SECOND


def one_hot(position):
    vector = np.zeros(DIMENSION, dtype=np.float32)
    vector[position] = 1.0
    return vector


def embedding_payload(vector, model_name=None):
    header = {"dim": int(vector.shape[0]), "dtype": "float32"}
    if model_name:
        header["model_name"] = model_name
    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(header_bytes)) + header_bytes + vector.tobytes()


def run_to_eos(pipeline):
    pipeline.set_state(Gst.State.PLAYING)
    message = pipeline.get_bus().timed_pop_filtered(
        PIPELINE_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    pipeline.set_state(Gst.State.NULL)
    assert message is not None, "pipeline did not finish"
    if message.type == Gst.MessageType.ERROR:
        error, debug = message.parse_error()
        pytest.fail(f"{error.message}\n{debug}")


# a pad probe's buffer is not writable, an in-place transform's is
class AppendEmbedding(GstBase.BaseTransform):
    __gstmetadata__ = ("Append Embedding", "Filter", "adds one embedding", "test")
    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, Gst.Caps.new_any()
        ),
        Gst.PadTemplate.new(
            "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, Gst.Caps.new_any()
        ),
    )

    def __init__(self):
        super().__init__()
        self.set_in_place(True)
        self.frame_index = 0

    def do_transform_ip(self, buffer):
        blob = EMBEDDING_HEADER + embedding_payload(one_hot(self.frame_index))
        carrier = Gst.Buffer.new_allocate(None, len(blob), None)
        carrier.fill(0, blob)
        buffer.append_memory(carrier.get_memory(0))
        self.frame_index += 1
        return Gst.FlowReturn.OK


GObject.type_register(AppendEmbedding)
Gst.Element.register(None, "appendembedding", Gst.Rank.NONE, AppendEmbedding)


def test_search_ranks_the_closest_vector_first(tmp_path):
    index = EmbeddingIndex.open(tmp_path / "index.sqlite")
    for position in range(DIMENSION):
        index.add(SOURCE_ID, position, one_hot(position), MODEL_NAME)
    results = index.search(one_hot(1) * 5, 2)
    assert [result["pts"] for result in results] == [1.0, 0.0]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[0]["source_id"] == SOURCE_ID
    assert results[1]["score"] == pytest.approx(0.0)
    assert index.model_name() == MODEL_NAME
    index.close()


def test_an_empty_index_has_no_model_and_no_results(tmp_path):
    index = EmbeddingIndex.open(tmp_path / "empty.sqlite")
    assert index.model_name() == ""
    assert index.search(one_hot(0), 5) == []
    index.close()


def test_two_models_cannot_share_an_index(tmp_path):
    index = EmbeddingIndex.open(tmp_path / "index.sqlite")
    index.add(SOURCE_ID, 0, one_hot(0), MODEL_NAME)
    with pytest.raises(ValueError, match="facebook/dinov2-base"):
        index.add(SOURCE_ID, 1, one_hot(1), "facebook/dinov2-base")
    index.close()


def test_the_blob_header_can_name_the_model():
    header, vector = decode_embedding(embedding_payload(one_hot(2), MODEL_NAME))
    assert header["model_name"] == MODEL_NAME
    assert header["dim"] == DIMENSION
    assert list(vector) == [0.0, 0.0, 1.0]


def test_the_sink_stores_one_row_per_embedded_frame(tmp_path):
    location = tmp_path / "index.sqlite"
    pipeline = Gst.parse_launch(
        f"videotestsrc num-buffers={FRAMES} "
        f"! video/x-raw,width=64,height=48,framerate={FRAMERATE}/1,format=RGBA "
        "! appendembedding "
        f"! mp_embeddingsink location={location.as_posix()} "
        f"source-id={SOURCE_ID} model-name={MODEL_NAME}"
    )
    run_to_eos(pipeline)

    index = EmbeddingIndex.open(location)
    assert index.model_name() == MODEL_NAME
    results = index.search(one_hot(2), FRAMES)
    assert len(results) == FRAMES
    assert results[0]["pts"] == pytest.approx(2 / FRAMERATE)
    assert results[0]["score"] == pytest.approx(1.0)
    assert sorted(result["pts"] for result in results) == pytest.approx(
        [frame / FRAMERATE for frame in range(FRAMES)]
    )
    assert {result["source_id"] for result in results} == {SOURCE_ID}
    index.close()


class StubTextEngine:
    def __init__(self, vector):
        self.vector = vector
        self.queries = []

    def do_text_embedding(self, text):
        self.queries.append(text)
        return self.vector


def test_search_video_answers_from_the_index(tmp_path, monkeypatch):
    mp_mcp = pytest.importorskip("mp_mcp")
    index = EmbeddingIndex.open(tmp_path / "index.sqlite")
    for position in range(DIMENSION):
        index.add(SOURCE_ID, position, one_hot(position), MODEL_NAME)
    index.close()

    engine = StubTextEngine(one_hot(2))
    monkeypatch.setattr(mp_mcp, "text_embedding_engine", lambda name: engine)
    results = mp_mcp.search_video("a ball", str(tmp_path / "index.sqlite"), 1)

    assert engine.queries == ["a ball"]
    assert results == [
        {"pts": 2.0, "source_id": SOURCE_ID, "score": pytest.approx(1.0)}
    ]


def test_search_video_rejects_a_missing_index(tmp_path):
    mp_mcp = pytest.importorskip("mp_mcp")
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="no embedding index"):
        mp_mcp.search_video("a ball", str(tmp_path / "absent.sqlite"), 1)
