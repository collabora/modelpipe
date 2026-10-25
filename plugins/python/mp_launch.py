# mp-launch
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Run one pipeline description on whichever backend `MP_BACKEND` selects.

`gst` hands the pipeline to `gst-launch-1.0` unchanged. `g2g` rewrites it for
`g2g-launch-py`, which spells six things differently:

  * an `mp_*` element is the generic `pyelement` host, or `pysink` for a sink,
    plus the module and class to load;
  * nine elements are g2g's own rather than hosted ones: `mp_overlay` is
    `analyticsoverlay`, `mp_alert` is `analyticsalert`, `mp_digest` is
    `textdigest`, and `mp_metasink` / `mp_metareplay` / `mp_alertrecorder`
    / `mp_embeddingsink` / `mp_streammux` / `mp_streamdemux` drop the
    prefix;
  * `pyelement` and `analyticsoverlay` both work on RGBA, so a raw-video caps
    filter that leaves the format open has to pin it;
  * a hosted element carries no pad templates into g2g, so an element that
    declares `INPUT_CAPS` / `OUTPUT_CAPS` hands them to its host as properties,
    which is the only way it can take audio in and give text out;
  * a sink takes no clock properties and the overlay no `wait-text`, because
    g2g never does what those switch off;
  * an element that reads frames in CUDA memory, right after `nvdec`, is hosted
    on the decoder's NV12 frames with `cuda-frames=true`.

Everything else is named the same on both: `filesrc`, `decodebin`,
`videoconvert`, `videoscale`, `autovideosink`. Write the pipeline the way the
PIPELINES.md does, in `gst-launch` spelling, and this translates it. Either way the
line the backend runs is printed, so it can be pasted back and extended.
"""

import ast
import os
import shlex
import shutil
import site
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

#: The `plugins` tree this module sits in, in a checkout or the installed wheel.
_PLUGINS = Path(__file__).resolve().parent.parent
CHECKOUT_PLUGINS = _PLUGINS if _PLUGINS.name == "plugins" else None

USAGE = "usage: mp-launch <element> [key=value ...] ! <element> ! ..."

#: The g2g launcher, and the checkout it is built in, looked for beside this one.
G2G_LAUNCH_NAME = "g2g-launch-py"
G2G_CHECKOUT_NAME = "glass2glass"

#: The g2g elements that host a ModelPipe one: a transform, the N-in
#: batching host an aggregator needs, and a sink.
PY_ELEMENT = "pyelement"
PY_AGGREGATOR = "pyaggregator"
PY_SINK = "pysink"

#: The base class a ModelPipe element hosted on `pysink` derives from.
SINK_BASE_CLASS = "BaseSink"

#: ModelPipe elements that g2g implements natively instead of hosting.
NATIVE_EQUIVALENTS = {
    "mp_overlay": "analyticsoverlay",
    "mp_metasink": "metasink",
    "mp_metareplay": "metareplay",
    "mp_alert": "analyticsalert",
    "mp_alertrecorder": "alertrecorder",
    "mp_digest": "textdigest",
    "mp_embeddingsink": "embeddingsink",
    "mp_streammux": "streammux",
    "mp_streamdemux": "streamdemux",
}

#: What each native element calls the properties it shares with the one it
#: replaces. A property missing here has no counterpart at all: g2g runs no
#: MQTT, so `mp_alert`'s broker and topic are refused rather than dropped.
NATIVE_PROPERTIES = {
    "mp_overlay": {"tracking": "show-track"},
    "mp_metasink": {"location": "location"},
    "mp_metareplay": {"location": "location"},
    "mp_alert": {
        "rules": "rules",
        "cooldown": "cooldown",
        "draw-alert": "draw-alert",
        "webhook-url": "webhook-url",
    },
    "mp_alertrecorder": {
        "location": "location",
        "encoder": "encoder",
        "seconds-before": "seconds-before",
        "seconds-after": "seconds-after",
    },
    "mp_digest": {"window-seconds": "window-seconds"},
    "mp_embeddingsink": {
        "location": "location",
        "source-id": "source-id",
        "model-name": "model-name",
    },
}

#: What separates two elements, on both spellings.
SEPARATOR = "!"

#: The property every element on both backends takes as its handle for links.
NAME_PROPERTY = "name"


class DefaultedProperty(NamedTuple):
    g2g_value: bool
    instead: str


#: `(element name suffix, property)` a g2g pipeline does not spell, mapped to
#: the value g2g always behaves as and what to do when the other one is wanted.
DEFAULTED_PROPERTIES = {
    ("sink", "sync"): DefaultedProperty(
        False,
        "a g2g sink does not wait on the clock, put a `clocksync` element ahead "
        "of it instead",
    ),
    ("sink", "async"): DefaultedProperty(
        False,
        "a g2g sink does not wait on the clock, put a `clocksync` element ahead "
        "of it instead",
    ),
    ("textoverlay", "wait-text"): DefaultedProperty(
        False, "g2g's textoverlay never holds video back for the text pad"
    ),
    ("pipewiresrc", "do-timestamp"): DefaultedProperty(
        True, "g2g's pipewiresrc always stamps a buffer with its capture time"
    ),
}

FALSE_VALUES = ("false", "0", "no", "off")

#: `pyelement` hosts an RGBA frame by default and `analyticsoverlay` draws on
#: RGBA8 only, so a caps filter that names no format would fail to negotiate.
G2G_RAW_VIDEO_FORMAT = "RGBA"

#: The class constant an element states each of its pad caps in, and the host
#: property it becomes. Each value is one `gst-launch` caps description, which
#: g2g refuses unless it names exactly one concrete caps.
G2G_CAPS_PROPERTIES = {
    "INPUT_CAPS": "input-caps",
    "OUTPUT_CAPS": "output-caps",
}

#: The typed capability descriptions that accept CUDA frames.
CUDA_FRAME_CAPABILITIES = {
    "CUDA_INFERENCE_FRAME_CAPABILITIES",
    "CUDA_DRAW_FRAME_CAPABILITIES",
    "CUDA_RGBA_DRAW_FRAME_CAPABILITIES",
    "METADATA_FRAME_CAPABILITIES",
}
FRAME_CAPABILITIES_CONSTANT = "FRAME_CAPABILITIES"

#: The g2g decoders that leave their frames in CUDA memory, and the host
#: property and format that take those frames as they are.
CUDA_DECODERS = ("nvdec", "nvh264dec")
CUDA_FRAMES_PROPERTY = "cuda-frames=true"
CUDA_FRAME_FORMAT = "NV12"

#: How each host that takes one raw video format is told which.
FORMAT_PROPERTIES = {
    PY_ELEMENT: "format={}",
    PY_SINK: "input-caps=video/x-raw,format={}",
}

#: gst's CUDA converters, which g2g has no counterpart for.
CUDA_CONVERTERS = ("cudaconvert", "cudascale", "cudaconvertscale")
CUDA_MEMORY_FEATURE = "memory:CUDAMemory"


class ElementShell(NamedTuple):
    """What hosting one ModelPipe element on g2g takes: the module and class
    to load, and the `(property, caps)` pairs it negotiates with, if it says."""

    module: str
    cls: str
    caps: tuple = ()
    reads_device_frames: bool = False
    is_sink: bool = False


class ClassDeclaration(NamedTuple):
    bases: list
    caps: dict
    accepts_device_frames: bool | None


def plugin_dir():
    """The directory holding the element modules: the checkout this module came
    from, or the `python` subdirectory of a `GST_PLUGIN_PATH` entry."""
    if CHECKOUT_PLUGINS:
        return CHECKOUT_PLUGINS / "python"
    for entry in os.environ.get("GST_PLUGIN_PATH", "").split(os.pathsep):
        candidate = Path(entry) / "python"
        if candidate.is_dir():
            return candidate
    raise SystemExit(
        "mp-launch: no ModelPipe element modules found; point "
        "GST_PLUGIN_PATH at the checkout's plugins directory"
    )


def element_shells(directory=None):
    """Map each element name to the `ElementShell` hosting it takes.

    Read out of the sources with `ast` rather than by importing: a plugin pulls
    in torch and friends, and every one whose dependencies are missing would
    drop out of the map.
    """
    shells = {}
    registered = []
    declarations = {}
    for path in sorted((directory or plugin_dir()).glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        constants = string_constants(tree)
        declarations.update(caps_declarations(tree))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            name = (
                called.attr
                if isinstance(called, ast.Attribute)
                else getattr(called, "id", None)
            )
            if name != "register_gst_element" or len(node.args) < 2:
                continue
            element, cls = node.args[0], node.args[1]
            element = element_name(element, constants)
            if element and isinstance(cls, ast.Name):
                registered.append((element, path.stem, cls.id))
    for element, module, cls in registered:
        shells[element] = ElementShell(
            module,
            cls,
            declared_caps(cls, declarations),
            reads_device_frames(cls, declarations),
            derives_from_sink(cls, declarations),
        )
    return shells


def caps_declarations(tree):
    """Each class in the module: the classes it derives from, and the g2g caps it
    states itself."""
    declarations = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        constants = string_constants(node)
        caps = {
            property: constants[f"{node.name}.{constant}"]
            for constant, property in G2G_CAPS_PROPERTIES.items()
            if f"{node.name}.{constant}" in constants
        }
        bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
        capabilities = frame_capabilities_name(node)
        declarations[node.name] = ClassDeclaration(
            bases,
            caps,
            capabilities in CUDA_FRAME_CAPABILITIES if capabilities else None,
        )
    return declarations


def frame_capabilities_name(class_node):
    for statement in class_node.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == FRAME_CAPABILITIES_CONSTANT
            for target in statement.targets
        ):
            continue
        if isinstance(statement.value, ast.Name):
            return statement.value.id
        if isinstance(statement.value, ast.Attribute):
            return statement.value.attr
    return None


def class_lineage(cls, declarations):
    pending, seen = [cls], set()
    while pending:
        name = pending.pop(0)
        if name in seen or name not in declarations:
            continue
        seen.add(name)
        yield declarations[name]
        pending.extend(declarations[name].bases)


def reads_device_frames(cls, declarations):
    stated = (
        declaration.accepts_device_frames
        for declaration in class_lineage(cls, declarations)
        if declaration.accepts_device_frames is not None
    )
    return next(stated, False)


def derives_from_sink(cls, declarations):
    return any(
        SINK_BASE_CLASS in declaration.bases
        for declaration in class_lineage(cls, declarations)
    )


def declared_caps(cls, declarations):
    """The `(property, caps)` pairs a class negotiates with, the ones it states
    itself ahead of the ones it inherits.

    A leaf that changes one pad restates only that pad, so a class taking text in
    and giving audio out gets its input from the base and its output from itself.
    """
    caps = {}
    for declaration in class_lineage(cls, declarations):
        for property, value in declaration.caps.items():
            caps.setdefault(property, value)
    return tuple(
        (property, caps[property])
        for property in G2G_CAPS_PROPERTIES.values()
        if property in caps
    )


def string_constants(tree):
    """Every `NAME = "..."` under the node, class attributes keyed `Class.NAME`.

    Some elements register under a class constant rather than a literal, so the
    name has to be resolved before it can go in the map.
    """
    constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            prefix = f"{node.name}."
            body = node.body
        elif isinstance(node, ast.Module):
            prefix = ""
            body = node.body
        else:
            continue
        for statement in body:
            if not isinstance(statement, ast.Assign):
                continue
            if not isinstance(statement.value, ast.Constant):
                continue
            if not isinstance(statement.value.value, str):
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    constants[f"{prefix}{target.id}"] = statement.value.value
    return constants


def element_name(node, constants):
    """The element name a `register_gst_element` first argument spells, whether
    it is a literal or a constant defined in the same module."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return constants.get(f"{node.value.id}.{node.attr}")
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def rewrite_for_g2g(argv, shells):
    """Translate `gst-launch` arguments into the ones `g2g-launch-py` takes.

    Works on the argument list rather than one joined string so a property value
    the shell already unquoted (`labels="person, bicycle"`) keeps its boundary,
    and gets requoted on the way out.
    """
    rewritten = []
    raw_format = None
    upstream_on_cuda = False
    parts = list(segments(joined_assignments(argv)))
    fan_in = fan_in_names(parts)
    for segment in parts:
        # Separators and references are carried through where they were
        # written: which chains link to which is the caller's, not ours.
        if segment == [SEPARATOR]:
            rewritten.extend(segment)
            continue
        if is_reference(segment[0]):
            rewritten.extend(segment)
            upstream_on_cuda = False
            continue
        # g2g has no GPU scaler or RGBA conversion, the element takes nvdec's frame
        if upstream_on_cuda and converts_cuda_frames(segment):
            rewritten.pop()
            continue
        host = PY_AGGREGATOR if declared_name(segment) in fan_in else PY_ELEMENT
        translated = rewrite_segment(
            segment, shells, raw_format, host, upstream_on_cuda
        )
        rewritten.extend(translated)
        raw_format = raw_video_format(translated[0]) or raw_format
        # a hosted element leaves the frame where it found it
        upstream_on_cuda = (
            segment[0] in CUDA_DECODERS or CUDA_FRAMES_PROPERTY in translated
        )
    return rewritten


# gst-launch allows spaces round the `=` of a property, the g2g parser does not.
def joined_assignments(argv):
    joined = []
    for token in argv:
        previous = joined[-1] if joined else ""
        _, separator, value = previous.partition("=")
        dangling = bool(separator) and not value
        starts_value = token.startswith("=") and previous and not separator
        crosses_link = SEPARATOR in (previous, token)
        if (dangling or starts_value) and not crosses_link:
            joined[-1] = previous + token
        else:
            joined.append(token)
    return joined


# GObject reads `_` in a property name as `-`, g2g knows only the dashed name.
def dashed_name(property):
    key, separator, value = property.partition("=")
    return f"{key.replace('_', '-')}{separator}{value}"


def converts_cuda_frames(segment):
    head = segment[0]
    if head in CUDA_CONVERTERS:
        return True
    if "/" in head:
        return CUDA_MEMORY_FEATURE in "".join(segment)
    caps = next((p for p in segment[1:] if p.startswith("caps=")), "")
    return head == "capsfilter" and CUDA_MEMORY_FEATURE in caps


def declared_name(segment):
    """The handle a segment gives itself with `name=`, if any."""
    for token in segment[1:]:
        key, separator, value = token.partition("=")
        if separator and key == NAME_PROPERTY:
            return value
    return None


def fan_in_names(parts):
    """The element handles several chains feed into.

    A chain *ending* in a bare `mux.` reference links into that element, so an
    element named by more than one is taking more than its own chain's input.
    That is the one shape g2g hosts on `pyaggregator` (its N-in batching host)
    rather than the one-in `pyelement`. A reference that *starts* a chain
    (`cap.text_src ! ...`) reads the other way, out of a second source pad, and
    leaves the element a one-in transform.
    """
    feeding = [
        segment[0].partition(".")[0]
        for previous, segment in zip([None, *parts], parts)
        if previous == [SEPARATOR] and len(segment) == 1 and is_reference(segment[0])
    ]
    return {name for name in feeding if feeding.count(name) > 1}


def raw_video_format(caps):
    """The format a raw-video caps filter pins, or `None` for anything else."""
    if not caps.startswith("video/x-raw"):
        return None
    for field in caps.split(","):
        key, separator, value = field.partition("=")
        if separator and key == "format":
            return value
    return None


def segments(argv):
    """The argument list split into one token list per element, `!`, or
    reference.

    A `!` separates two elements, and a bare `name.` reference is a chain
    boundary of its own: `mux. ! sink` starts a chain at that element and
    `... ! mux.` ends one there. Both come back as their own single-token
    segment, so the links the caller wrote survive the rewrite unchanged.
    """
    segment = []
    for token in argv:
        if token == SEPARATOR or is_reference(token):
            if segment:
                yield segment
            yield [token]
            segment = []
        # a word with no `=` after an element is the next chain's first element
        elif segment and "=" not in token:
            yield segment
            segment = [token]
        else:
            segment.append(token)
    if segment:
        yield segment


def is_reference(token):
    """Whether the token is a `name.` / `name.pad` reference to an element."""
    name, separator, pad = token.partition(".")
    return bool(
        separator
        and name[:1].isalpha()
        and name.replace("_", "").replace("-", "").isalnum()
        and (not pad or pad.replace("_", "").isalnum())
    )


def rewrite_segment(
    segment, shells, raw_format=None, host=None, upstream_on_cuda=False
):
    if isinstance(segment, str):
        segment = segment.split()
    if not segment:
        return []
    head, properties = segment[0], [dashed_name(p) for p in segment[1:]]

    # A caps filter is the one segment whose first token is a media type; no
    # element name contains a slash. Caps carry no spaces, so a caps written
    # with one (`video/x-raw, width=320`) arrives as several tokens to rejoin.
    if "/" in head:
        return [with_pinned_format("".join(segment))]

    native = NATIVE_EQUIVALENTS.get(head)
    if native:
        return [native, *[quoted(p) for p in renamed_properties(head, properties)]]

    shell = shells.get(head)
    if shell:
        # Every property of a hosted element goes to the Python class, so the
        # defaulted-property table must not touch one whose name happens to end
        # like a g2g element's (`mp_kafkasink` and its `sync`).
        properties = [quoted(p) for p in properties]
        if shell.is_sink:
            host = PY_SINK
        hosted = [host or PY_ELEMENT, f"module={shell.module}", f"class={shell.cls}"]
        # `pyaggregator` negotiates from its inputs and takes no format.
        format_property = FORMAT_PROPERTIES.get(hosted[0])
        on_cuda_frames = (
            format_property is not None
            and upstream_on_cuda
            and shell.reads_device_frames
        )
        # `pyelement` takes RGBA unless told otherwise, so an upstream caps
        # filter naming another format has to reach the hosted element too.
        pin = raw_format not in (None, G2G_RAW_VIDEO_FORMAT)
        names_format = any(p.startswith("format=") for p in properties)
        if on_cuda_frames:
            hosted.append(CUDA_FRAMES_PROPERTY)
            hosted.append(format_property.format(CUDA_FRAME_FORMAT))
        elif format_property and pin and not names_format:
            hosted.append(format_property.format(raw_format))
        hosted.extend(quoted(f"{property}={value}") for property, value in shell.caps)
        return [*hosted, *properties]

    if head.startswith("mp_"):
        raise SystemExit(f"mp-launch: no ModelPipe element named {head!r}")

    return [head, *[quoted(p) for p in drop_defaulted_properties(head, properties)]]


def renamed_properties(head, properties):
    """The properties under the names g2g's own element spells them.

    A native equivalent is a different element, not a rename, so only the knobs
    that mean the same thing carry over. One that does not is an error rather
    than a guess, because dropping it would run a pipeline that quietly does
    something else.
    """
    renames = {NAME_PROPERTY: NAME_PROPERTY, **NATIVE_PROPERTIES.get(head, {})}
    renamed = []
    for property in properties:
        key, _, value = property.partition("=")
        if key not in renames:
            raise SystemExit(
                f"mp-launch: g2g's {NATIVE_EQUIVALENTS[head]} has no counterpart "
                f"for {head} {key}; of its properties only "
                f"{', '.join(sorted(renames))} carries over"
            )
        renamed.append(f"{renames[key]}={value}")
    return renamed


def drop_defaulted_properties(head, properties):
    kept = []
    for property in properties:
        key, _, value = property.partition("=")
        defaulted = next(
            (
                defaulted
                for (suffix, name), defaulted in DEFAULTED_PROPERTIES.items()
                if name == key and head.endswith(suffix)
            ),
            None,
        )
        if defaulted is None:
            kept.append(property)
            continue
        # Asking for the other value needs a pipeline change this cannot make.
        if (value.lower() not in FALSE_VALUES) != defaulted.g2g_value:
            raise SystemExit(
                f"mp-launch: {head} {property} has no equivalent on g2g; "
                f"{defaulted.instead}"
            )
    return kept


def quoted(token):
    """A token spelled so g2g's launch parser reads the value back unchanged.

    Its grammar treats a quote, a `\\`, a `!` or a `#` as syntax wherever it
    appears, and the argument list is joined with spaces before parsing, so a
    value carrying any of those has to say so.
    """
    key, separator, value = token.partition("=")
    if not separator:
        return token
    for character in ("\\", '"', "'", "!", "#"):
        value = value.replace(character, f"\\{character}")
    if any(character.isspace() for character in value):
        value = f'"{value}"'
    return f"{key}={value}"


def with_pinned_format(caps):
    if not caps.startswith("video/x-raw") or "format=" in caps:
        return caps
    return f"{caps},format={G2G_RAW_VIDEO_FORMAT}"


def _prepend_path(existing, *entries):
    paths = [str(entry) for entry in entries]
    if existing:
        paths.append(existing)
    return os.pathsep.join(paths)


def launch_environment():
    """The environment both launchers need.

    Neither runs under this interpreter: GStreamer's Python loader embeds the
    system one and g2g embeds its own, so torch and the rest are only reachable
    if the venv's site directories are on `PYTHONPATH`.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_path(
        env.get("PYTHONPATH"), plugin_dir(), *site.getsitepackages()
    )
    return env


# windows appends .exe only to a name without an extension, and ".0" counts as one
def gst_command(argv):
    gst_launch = shutil.which("gst-launch-1.0")
    if gst_launch is None:
        raise SystemExit("mp-launch: gst-launch-1.0 is not on PATH")
    env = launch_environment()
    if CHECKOUT_PLUGINS:
        env["GST_PLUGIN_PATH"] = _prepend_path(
            env.get("GST_PLUGIN_PATH"), CHECKOUT_PLUGINS
        )
    return [gst_launch, *argv], env


def g2g_binary():
    """The `g2g-launch-py` to run: an explicit `G2G_LAUNCH`, else the release
    build in the glass2glass checkout, else whatever is on `PATH`."""
    explicit = os.environ.get("G2G_LAUNCH")
    if explicit:
        return explicit
    checkout = os.environ.get("G2G_DIR")
    if not checkout and CHECKOUT_PLUGINS:
        checkout = CHECKOUT_PLUGINS.parent.parent / G2G_CHECKOUT_NAME
    if checkout:
        release = Path(checkout) / "target" / "release" / G2G_LAUNCH_NAME
        if release.is_file():
            return str(release)
    return shutil.which(G2G_LAUNCH_NAME)


def required_g2g_binary():
    binary = g2g_binary()
    if not binary:
        raise SystemExit(
            "mp-launch: no g2g-launch-py release build found; point G2G_LAUNCH "
            "at one, or G2G_DIR at a glass2glass checkout, and build it with: "
            "PYO3_PYTHON=$(which python) cargo build --release -p g2g-python "
            "--features ml --bin g2g-launch-py"
        )
    return binary


def g2g_command(pipeline):
    return [required_g2g_binary(), "--threads", *pipeline], launch_environment()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        raise SystemExit(USAGE)

    backend = os.environ.get("MP_BACKEND", "gst").lower()
    if backend == "gst":
        command, env = gst_command(argv)
    elif backend == "g2g":
        command, env = g2g_command(rewrite_for_g2g(argv, element_shells()))
    else:
        raise SystemExit(
            f"mp-launch: unknown MP_BACKEND={backend!r}; use 'gst' or 'g2g'"
        )

    # The line the backend actually runs, so it can be pasted back and extended.
    print(f"mp-launch: {shlex.join(command)}", file=sys.stderr)
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    sys.exit(main())
