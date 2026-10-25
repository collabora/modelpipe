import importlib.util
import subprocess
import os
import signal
import re
import sys
import pytest
from pathlib import Path
import shlex
import shutil
import socket
import uuid
import stat
import time
from functools import cache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "python"))

from documented_pipelines import pipelines_by_section  # noqa: E402

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "tests" / "logs"

# Seconds to let a pipeline run. Overridable because a local LLM generating a
# few hundred tokens takes minutes, not seconds.
PIPELINE_TIMEOUT = int(os.environ.get("PIPELINE_TIMEOUT", "120"))

# Only these mean the pipeline broke. Warnings are not fatal: plugins unrelated
# to the pipeline warn during setup and would fail a run that went fine.
FATAL_LOG_PATTERNS = (
    re.compile(r"^ERROR:.*", re.MULTILINE),
    re.compile(r"^WARNING: erroneous pipeline.*", re.MULTILINE),
    re.compile(r"^\S+ +\S+ +\S+ +ERROR +python .*", re.MULTILINE),
    # g2g's own failures: the launcher's parse / run errors and the rewrite's.
    re.compile(r"^(?:parse|pipeline) error:.*", re.MULTILINE),
    re.compile(r"^mp-launch: (?:no|unknown) .*", re.MULTILINE),
)

BACKEND = os.environ.get("MP_BACKEND", "gst").lower()

# The launcher the PIPELINES.md examples name. Runs from a tmp dir, so it is spelled
# absolute here.
# on windows a bare `python` finds the base interpreter beside the venv's launcher
LAUNCHER = (
    f"{shlex.quote(Path(sys.executable).as_posix())} "
    f"{shlex.quote((BASE_DIR / 'mp-launch.py').as_posix())}"
)
WINDOWS = sys.platform == "win32"

# small models, no display or microphone, and the cpu when torch sees no gpu
HEADLESS = os.environ.get("HEADLESS_PIPELINES") == "1"
CAPTURE_SOURCES = ("pulsesrc", "pipewiresrc", "autoaudiosrc", "alsasrc", "v4l2src")
# torch.compile takes minutes on a runner
TORCH_COMPILE_MARKER = "compile=True"
ENGINE_NAME = re.compile(r"\bengine-name=([^\s!]+)")
# too large for a runner, or needing a server or a model file outside the repo
HEAVY_ELEMENTS = (
    "mp_llm",
    "mp_stablediffusion",
    "mp_whisper",
    "mp_mariantranslate",
    "mp_caption_qwen",
    "mp_vlm",
    "mp_sam",
    "mp_kafkasink",
    "demo_soccer",
    "mp_demucs",
    "mp_sepformer",
    "mp_superres",
    "mp_face",
    "mp_ocr",
)
HEADLESS_SKIP_MARKERS = CAPTURE_SOURCES + (TORCH_COMPILE_MARKER,) + HEAVY_ELEMENTS
DISPLAY_SINK_PATTERN = re.compile(r"\b(?:autovideosink|glimagesink)\b")
HEADLESS_SINK = "fakevideosink"
CUDA_TO_CPU = (re.compile(r"\bdevice=cuda(?::\d+)?\b"), "device=cpu")
HEADLESS_REWRITES = (
    (re.compile(r"\byolo11m\b"), "yolo11n"),
    (DISPLAY_SINK_PATTERN, HEADLESS_SINK),
)
# an eos this many buffers before the first ml element proves frames went through it
HEADLESS_FRAME_CAP = 20
FRAME_CAP_ELEMENT = f"identity eos-after={HEADLESS_FRAME_CAP}"
FIRST_ML_ELEMENT = re.compile(r"\b(?:mp_\w+|demo_\w+)\b")
LOG_TAIL_LINES = 40


@cache
def torch_sees_a_gpu():
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


# the engines job covers the other engines, and pytorch's cuda frame examples need a gpu
def needs_another_job(pipeline):
    engine = ENGINE_NAME.search(pipeline)
    return engine is not None and (
        engine.group(1) != "pytorch" or not torch_sees_a_gpu()
    )


def runs_headless(pipeline):
    if needs_another_job(pipeline):
        return False
    return not any(marker in pipeline for marker in HEADLESS_SKIP_MARKERS)


def headless_pipeline(pipeline):
    rewrites = (
        HEADLESS_REWRITES if torch_sees_a_gpu() else (CUDA_TO_CPU, *HEADLESS_REWRITES)
    )
    for pattern, replacement in rewrites:
        pipeline = pattern.sub(replacement, pipeline)
    return capped_before_first_element(pipeline)


# only with one source and no mux or tee does an eos on that path end the whole pipeline
def capped_before_first_element(pipeline):
    if (
        pipeline.count("filesrc") != 1
        or "mp_streammux" in pipeline
        or " tee " in pipeline
    ):
        return pipeline
    first = FIRST_ML_ELEMENT.search(pipeline)
    if first is None:
        return pipeline
    return (
        f"{pipeline[: first.start()]}{FRAME_CAP_ELEMENT} ! {pipeline[first.start():]}"
    )


# a pulse source with no device named opens the default source
BARE_PULSE_SOURCE = re.compile(
    r"\b(?:pulsesrc|pipewiresrc)\b(?![^!]*\b(?:device|target-object)=)"
)
DEFAULT_SOURCE_METADATA_KEY = "key:'default.audio.source'"


# after a forced mic port pactl info still names a default but wireplumber has none
def pipewire_has_default_audio_source():
    if shutil.which("pw-metadata") is None:
        return True
    result = subprocess.run(
        ["pw-metadata", "0", "default.audio.source"], capture_output=True, text=True
    )
    return DEFAULT_SOURCE_METADATA_KEY in result.stdout


EXPORTED_MODEL_SUFFIXES = {
    ".gguf",
    ".so",
    ".vmfb",
    ".param",
    ".xml",
    ".tflite",
    ".onnx",
    ".pte",
    ".rknn",
    ".safetensors",
}
EXPORTED_MODEL_DIR_MARKERS = ("_saved_model", "_openvino_model", "snapshot")


# the engine class registers without its runtime, so check the runtime itself
ENGINE_RUNTIME_MODULES = {
    "pytorch": "torch",
    "onnx": "onnxruntime",
    "openvino": "openvino",
    "tensorflow": "tensorflow",
    "tflite": "ai_edge_litert",
    "ncnn": "ncnn",
    "executorch": "executorch",
    "iree": "iree",
    "llamacpp": "llama_cpp",
    "tvm": "tvm",
    "mlx": "mlx",
    "jax": "jax",
    "tinygrad": "tinygrad",
    "migraphx": "migraphx",
    "rknn": "rknnlite",
    "vart": "runner",
}


def engine_is_installed(engine_name):
    module = ENGINE_RUNTIME_MODULES.get(engine_name)
    return module is None or importlib.util.find_spec(module) is not None


# the engines job covers these, a local run only has the engines and exports it made
def skip_without_engine_or_exported_model(pipeline):
    engine = re.search(r"\bengine-name=([^\s!]+)", pipeline)
    if engine and not engine_is_installed(engine.group(1)):
        pytest.skip(f"engine {engine.group(1)} is not installed")
    for model in re.findall(r"\bmodel-name=([^\s!]+)", pipeline):
        candidate = Path(model.strip('"'))
        exported = candidate.suffix in EXPORTED_MODEL_SUFFIXES or any(
            marker in candidate.name for marker in EXPORTED_MODEL_DIR_MARKERS
        )
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        if exported and not candidate.exists():
            pytest.skip(f"exported model {model} is not present")


# device=rocm and device=npu need an onnxruntime built for that hardware
def skip_without_onnx_provider(pipeline):
    device = re.search(r"\bdevice=([^\s!]+)", pipeline)
    if "engine-name=onnx" not in pipeline or device is None:
        return
    import onnxruntime
    from engine.onnx_engine import (
        DEVICE_PROVIDERS,
        TENSORRT_PROVIDER,
        tensorrt_provider_load_error,
    )

    wanted = next(
        (
            providers
            for keyword, providers in DEVICE_PROVIDERS.items()
            if keyword in device.group(1)
        ),
        (),
    )
    if wanted and not set(wanted) & set(onnxruntime.get_available_providers()):
        pytest.skip(f"onnxruntime has no provider for device={device.group(1)}")
    # onnxruntime-gpu lists the tensorrt provider without its libraries
    if TENSORRT_PROVIDER in wanted:
        load_error = tensorrt_provider_load_error()
        if load_error is not None:
            pytest.skip(f"the tensorrt provider does not load: {load_error}")


def skip_without_server(pipeline):
    match = re.search(r"\b(?:broker|url)=(?:https?://)?([^\s!:/]+):(\d+)", pipeline)
    if not match:
        return
    host, port = match.group(1), int(match.group(2))
    with socket.socket() as probe:
        probe.settimeout(1)
        try:
            probe.connect((host, port))
        except OSError:
            pytest.skip(f"nothing listens on {host}:{port}")


# the first word after the launcher or a `!`, caps and pad references excluded
ELEMENT_NAME = re.compile(r"(?:^|!)\s*(?:\\\s*)*([a-z][a-z0-9_]*)(?=\s|$)")


@cache
def gstreamer_has_element(name):
    gst_inspect = shutil.which("gst-inspect-1.0")
    if gst_inspect is None:
        raise RuntimeError("gst-inspect-1.0 is not on PATH")
    completed = subprocess.run([gst_inspect, "--exists", name], capture_output=True)
    return completed.returncode == 0


# a missing modelpipe element is a failure, a missing gstreamer plugin is the platform
def skip_without_gstreamer_elements(pipeline):
    description = pipeline.removeprefix(LAUNCHER)
    for name in ELEMENT_NAME.findall(description):
        if name.startswith(("mp_", "demo_")):
            continue
        if not gstreamer_has_element(name):
            pytest.skip(f"this gstreamer has no {name} element")


def skip_without_default_audio_source(pipeline):
    if not BARE_PULSE_SOURCE.search(pipeline):
        return
    if not pipewire_has_default_audio_source():
        pytest.skip("wireplumber has no default audio source for a bare pulsesrc")


if BACKEND == "gst" and not shutil.which("gst-launch-1.0"):
    raise RuntimeError("gst-launch-1.0 not found in PATH. Please install GStreamer.")


def get_documented_pipelines():
    doc_path = BASE_DIR / "PIPELINES.md"
    if not doc_path.exists():
        pytest.fail("PIPELINES.md not found in project root")

    sections = pipelines_by_section(doc_path)
    pipelines = [
        f"{LAUNCHER} {description}"
        for descriptions in sections.values()
        for description in descriptions
    ]
    if HEADLESS:
        pipelines = [
            headless_pipeline(pipeline)
            for pipeline in pipelines
            if runs_headless(pipeline)
        ]

    modified_pipelines = []
    for pipeline in pipelines:
        parts = pipeline.split("!")

        # A `filesrc` run has no equivalent cap: these mp4s carry `moov` at the
        # end, so bounding the source by bytes leaves the decoder with no index.
        # Those pipelines run until PIPELINE_TIMEOUT instead.
        for i, part in enumerate(parts):
            part_clean = part.strip()
            if "videotestsrc" in part_clean:
                if "num-buffers=" not in part_clean:
                    # Append num-buffers=100 as part of the element, not after !
                    parts[i] = f"{part_clean} num-buffers=100"
                else:
                    parts[i] = re.sub(r"num-buffers=\d+", "num-buffers=100", part_clean)
                break

        modified_pipeline = " ! ".join(parts).strip()
        modified_pipelines.append(modified_pipeline)
    return modified_pipelines


PIPELINES = get_documented_pipelines()


def end_process_group(process):
    """Stop the pipeline and everything it started.

    `shell=True` makes the shell the direct child, so signalling the process
    alone leaves the launcher and its window behind.
    """
    if WINDOWS:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True
        )
        process.wait()
        return
    try:
        group = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    os.killpg(group, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(group, signal.SIGKILL)


def absolutize_project_inputs(pipeline):
    """Point a pipeline's relative input paths at the project directory.

    Lets the pipeline run from the test's tmp dir so its output lands there.
    Only values that already name a file are rewritten, so output paths and
    caps strings are left alone.
    """

    def rewrite(match):
        key, value = match.group(1), match.group(2)
        candidate = BASE_DIR / value.strip('"')
        return (
            f"{key}={candidate.as_posix()}" if candidate.is_file() else match.group(0)
        )

    return re.sub(r"([\w-]+)=([^\s!]+)", rewrite, pipeline)


@pytest.mark.serial
@pytest.mark.parametrize("pipeline", PIPELINES, ids=lambda p: p)
def test_pipeline(pipeline, tmp_path):
    """
    Run a PIPELINES.md pipeline and check its log for errors.

    A pipeline still running at `PIPELINE_TIMEOUT` passes: only `videotestsrc`
    takes a frame cap, so a file-backed one runs as long as its media lasts.
    """
    skip_without_default_audio_source(pipeline)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not WINDOWS:
        os.sync()
    pipeline = absolutize_project_inputs(pipeline)
    skip_without_engine_or_exported_model(pipeline)
    skip_without_onnx_provider(pipeline)
    skip_without_server(pipeline)
    skip_without_gstreamer_elements(pipeline)
    unique_id = uuid.uuid4().hex[:8]
    log_file = LOG_DIR / f"test_{unique_id}.log"

    print(f"Testing pipeline: {pipeline}")
    print(f"Log file: {log_file}")

    # Check if input file exists for filesrc
    match = re.search(r"filesrc location=([^\s!]+)", pipeline)
    if match:
        file_path = Path(match.group(1))
        if not file_path.is_absolute():
            file_path = BASE_DIR / file_path
        if not file_path.exists():
            pytest.fail(f"Input file not found: {file_path}. Full pipeline: {pipeline}")

    # Verify log directory state
    if not LOG_DIR.exists():
        pytest.fail(
            f"Log directory {LOG_DIR} does not exist after mkdir. Check permissions."
        )
    if not os.access(str(LOG_DIR), os.W_OK):
        perms = oct(stat.S_IMODE(os.stat(LOG_DIR).st_mode))
        pytest.fail(f"No write permission for {LOG_DIR}. Current perms: {perms}")

    print(f"Log dir exists: {LOG_DIR.exists()}")
    print(f"Log dir writable: {os.access(str(LOG_DIR), os.W_OK)}")
    print(f"Log dir contents: {list(LOG_DIR.iterdir())}")

    # Create the log file
    try:
        fd = os.open(str(log_file), os.O_CREAT | os.O_WRONLY, 0o666)
        os.close(fd)
        print(f"Log file {log_file} created successfully with os.open")
    except Exception as e:
        pytest.fail(f"Failed to create log file {log_file} with os.open: {e}")

    time.sleep(0.1)

    # Set up environment with latency tracer
    env = os.environ.copy()
    env["GST_TRACERS"] = "latency"
    # Colour escapes land in the log file and break matching on the level field.
    env["GST_DEBUG_NO_COLOR"] = "1"

    # Run the pipeline
    ran_to_the_cap = False
    try:
        with open(log_file, "w") as log:
            # Own process group: the shell is not the pipeline, it is the
            # launcher's parent, so killing the group is what stops the run.
            # Terminating the shell alone leaves the launcher holding a window
            # and the GPU until the machine is rebooted.
            if WINDOWS:
                # cmd.exe does not read the posix shell syntax PIPELINES.md uses
                launch = {"args": shlex.split(pipeline.replace("\\\n", " "))}
            else:
                launch = {"args": pipeline, "shell": True, "start_new_session": True}
            process = subprocess.Popen(
                **launch,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=tmp_path,
                env=env,
            )
            process.wait(timeout=PIPELINE_TIMEOUT)
            return_code = process.returncode
    except subprocess.TimeoutExpired:
        # Still running at the cap, which is what a healthy uncapped pipeline
        # does: the media outlasts any timeout worth waiting. The log below says
        # whether it was working, so the run is judged on that, not on exiting.
        end_process_group(process)
        ran_to_the_cap = True
        return_code = None
    except Exception as e:
        end_process_group(process)
        pytest.fail(
            f"Failed to execute pipeline: {e}. Full pipeline: {pipeline}. See {log_file}"
        )

    # Check logs for errors
    if not log_file.exists():
        pytest.fail(f"Log file {log_file} was not created. Full pipeline: {pipeline}")
    with open(log_file, "r", encoding="utf-8", errors="replace") as log:
        log_content = log.read()
    failures = [m.group(0) for p in FATAL_LOG_PATTERNS for m in p.finditer(log_content)]
    if failures:
        reported = "\n".join(failures)
        log_tail = "\n".join(log_content.splitlines()[-LOG_TAIL_LINES:])
        pytest.fail(
            f"Errors found in pipeline:\n{reported}\nFull pipeline: {pipeline}\n"
            f"Last {LOG_TAIL_LINES} log lines:\n{log_tail}"
        )

    # Without this a pipeline that never left PAUSED passes on an empty log.
    if "Setting pipeline to PLAYING" not in log_content:
        pytest.fail(
            f"Pipeline never reached PLAYING. Full pipeline: {pipeline}. See {log_file}"
        )

    if ran_to_the_cap and FRAME_CAP_ELEMENT in pipeline:
        pytest.fail(
            f"Pipeline did not push {HEADLESS_FRAME_CAP} buffers through its first "
            f"element within {PIPELINE_TIMEOUT}s. Full pipeline: {pipeline}. See {log_file}"
        )

    # Check exit code
    if not ran_to_the_cap and return_code != 0:
        if (
            "End-Of-Stream" not in log_content
            and "reached end of stream" not in log_content
        ):
            pytest.fail(
                f"Pipeline failed with exit code {return_code}. Full pipeline: {pipeline}. See {log_file}"
            )

    ending = f"ran the full {PIPELINE_TIMEOUT}s" if ran_to_the_cap else "ran to the end"
    print(f"Pipeline {ending} with no errors: {pipeline}")


def test_pipelines_found():
    """Ensure at least one pipeline was found in PIPELINES.md."""
    if not PIPELINES:
        pytest.fail("No mp-launch pipelines found in PIPELINES.md")
    print(f"Found {len(PIPELINES)} pipelines to test")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
