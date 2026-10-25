from pathlib import Path

import numpy as np
import pytest

SHADER_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "plugins"
    / "python"
    / "overlay_helper"
    / "shaders"
)
SPIRV_HEADER_WORDS = 5
OP_SOURCE_CONTINUED = 2
OP_SOURCE = 3
# language, version and file id come before the text
OP_SOURCE_TEXT_OPERAND = 4


def embedded_source(spirv):
    words = np.frombuffer(spirv, dtype="<u4")
    pieces = []
    position = SPIRV_HEADER_WORDS
    while position < len(words):
        word_count, opcode = words[position] >> 16, words[position] & 0xFFFF
        instruction = words[position : position + word_count]
        if opcode == OP_SOURCE and word_count > OP_SOURCE_TEXT_OPERAND:
            pieces.append(instruction[OP_SOURCE_TEXT_OPERAND:].tobytes())
        if opcode == OP_SOURCE_CONTINUED:
            pieces.append(instruction[1:].tobytes())
        position += word_count
    return b"".join(piece.rstrip(b"\0") for piece in pieces).decode()


@pytest.mark.parametrize("source_name", ["overlay.vert", "overlay.frag"])
def test_shipped_spirv_was_compiled_from_the_shader_source(source_name):
    source = (SHADER_DIRECTORY / source_name).read_text()
    spirv = (SHADER_DIRECTORY / f"{source_name}.spv").read_bytes()

    assert embedded_source(spirv).endswith(source)
