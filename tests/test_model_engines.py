import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))

from engine import benchmark, model_engines  # noqa: E402
from engine.support_matrix import (  # noqa: E402
    ENGINE_DEVICES,
    ENGINE_EXTRAS,
    ENGINE_PACKAGES,
    FILE_ENGINES,
    NCNN_BATCH_BROADCAST,
    TASK_ENGINES,
    UNTESTED_VARIANT_NOTE,
)

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
CAUSAL_LM_MODEL = "Qwen/Qwen3-0.6B"
CLIP_MODEL = "openai/clip-vit-base-patch32"
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
WHISPER_MODEL = "openai/whisper-tiny"
TORCHVISION_CLASSIFIERS = ("resnet18", "mobilenet_v3_small")
TORCHVISION_RESNETS = ("resnet18",)
BENCHMARK_FRAMES = 5
INSTALLED_ENGINES = ("pytorch", "onnx", "ncnn")
REACHABLE_DEVICES = {"pytorch": ("cpu",), "onnx": ("cpu", "cuda"), "ncnn": ("cpu",)}
CHILD_ERROR = "CUDAExecutionProvider is not available"
MEASURED_RESULTS = {
    "pytorch": {"load_seconds": 6.0, "latency_ms": 390.0, "fps": 2.5},
    "onnx": {"load_seconds": 2.0, "latency_ms": 280.0, "fps": 3.5},
}


@pytest.fixture
def hub_configs(monkeypatch):
    architectures = {
        DEPTH_MODEL: ["DepthAnythingForDepthEstimation"],
        CAUSAL_LM_MODEL: ["Qwen3ForCausalLM"],
        CLIP_MODEL: ["CLIPModel"],
        GROUNDING_MODEL: ["GroundingDinoForObjectDetection"],
        WHISPER_MODEL: ["WhisperForConditionalGeneration"],
    }
    monkeypatch.setattr(model_engines, "hub_architectures", architectures.get)


@pytest.fixture(autouse=True)
def torchvision_names(monkeypatch):
    monkeypatch.setattr(
        model_engines,
        "is_torchvision_classifier",
        lambda name: name in TORCHVISION_CLASSIFIERS,
    )
    monkeypatch.setattr(
        model_engines, "is_torchvision_resnet", lambda name: name in TORCHVISION_RESNETS
    )


def tasks_of(model):
    return {task["task"]: task for task in model_engines.engine_options(model)["tasks"]}


def engine_of(task, engine):
    return next(entry for entry in task["engines"] if entry["engine"] == engine)


def test_a_depth_model_runs_on_onnx_and_ncnn_refuses_it(hub_configs):
    depth = tasks_of(DEPTH_MODEL)["depth"]
    assert engine_of(depth, "onnx")["status"] == model_engines.STATUS_RUNS
    ncnn = engine_of(depth, "ncnn")
    assert ncnn["status"] == model_engines.STATUS_REFUSED
    assert ncnn["reason"] == NCNN_BATCH_BROADCAST


def test_clip_serves_clip_and_embedding(hub_configs):
    assert set(tasks_of(CLIP_MODEL)) == {"clip", "embedding"}


def test_a_causal_lm_runs_on_llamacpp_and_not_on_tvm(hub_configs):
    llm = tasks_of(CAUSAL_LM_MODEL)["llm"]
    assert engine_of(llm, "llamacpp")["status"] == model_engines.STATUS_RUNS
    assert engine_of(llm, "tvm")["status"] == model_engines.STATUS_NOT_PORTED


def test_a_hub_whisper_runs_on_llamacpp_and_mlx_and_not_on_tvm(hub_configs):
    whisper = tasks_of(WHISPER_MODEL)["whisper"]
    assert engine_of(whisper, "llamacpp")["status"] == model_engines.STATUS_RUNS
    assert engine_of(whisper, "mlx")["status"] == model_engines.STATUS_RUNS
    assert engine_of(whisper, "tvm")["status"] == model_engines.STATUS_NOT_PORTED


def test_grounding_dino_runs_on_pytorch_only(hub_configs):
    zero_shot = tasks_of(GROUNDING_MODEL)["zero_shot"]
    assert zero_shot["note"] == "its image encoder takes the text"
    assert [entry["engine"] for entry in zero_shot["engines"]] == ["pytorch"]
    assert zero_shot["engines"][0]["status"] == model_engines.STATUS_PYTORCH_ONLY


def test_an_ultralytics_pose_name_resolves_to_pose():
    assert [resolved.task for resolved in model_engines.resolve("yolo11n-pose")] == [
        "pose"
    ]


def test_an_untested_ultralytics_variant_says_so():
    (resolved,) = model_engines.resolve("yolo11n-cls")
    assert resolved.task == "yolo"
    assert resolved.note == UNTESTED_VARIANT_NOTE


def test_an_onnx_file_lists_its_engines_as_untested(tmp_path):
    (task,) = model_engines.engine_options(str(tmp_path / "model.onnx"))["tasks"]
    assert {entry["engine"] for entry in task["engines"]} == set(FILE_ENGINES[".onnx"])
    assert {entry["status"] for entry in task["engines"]} == {
        model_engines.STATUS_UNTESTED
    }
    assert model_engines.MISSING_FILE_NOTE in task["note"]


def test_a_torchvision_resnet_serves_classifier_and_anomaly():
    assert set(tasks_of("resnet18")) == {"classifier", "anomaly"}


def test_a_torchvision_classifier_runs_on_onnx_and_is_not_ported_to_jax():
    classifier = tasks_of("mobilenet_v3_small")["classifier"]
    assert engine_of(classifier, "onnx")["status"] == model_engines.STATUS_RUNS
    assert engine_of(classifier, "jax")["status"] == model_engines.STATUS_NOT_PORTED


def test_an_unknown_name_resolves_to_nothing():
    assert model_engines.resolve("not-a-model") == []


def test_every_engine_table_names_a_known_engine():
    for table in (ENGINE_DEVICES, ENGINE_EXTRAS):
        assert set(table) == set(ENGINE_PACKAGES)
    for engines in FILE_ENGINES.values():
        assert set(engines) <= set(ENGINE_PACKAGES)


def test_every_engine_extra_is_a_pyproject_extra():
    with open(BASE_DIR / "pyproject.toml", "rb") as pyproject:
        extras = tomllib.load(pyproject)["project"]["optional-dependencies"]
    for engine_extras in ENGINE_EXTRAS.values():
        assert set(engine_extras) <= set(extras)


@pytest.fixture
def mocked_children(monkeypatch):
    monkeypatch.setattr(
        model_engines, "is_installed", lambda engine: engine in INSTALLED_ENGINES
    )
    monkeypatch.setattr(model_engines, "available_devices", lambda: REACHABLE_DEVICES)

    def benchmark_in_child(task, model, engine, device, frames):
        if device == "cuda":
            return {"error": CHILD_ERROR}
        return MEASURED_RESULTS[engine]

    monkeypatch.setattr(benchmark, "benchmark_in_child", benchmark_in_child)


def benchmark_rows_of(model):
    options = model_engines.engine_options(model)
    rows = benchmark.benchmark_rows(options, BENCHMARK_FRAMES)
    return {(row["engine"], row["device"]): row for row in rows}


def test_a_benchmark_times_each_reachable_device_of_a_running_engine(
    hub_configs, mocked_children
):
    rows = benchmark_rows_of(DEPTH_MODEL)
    for engine in ("pytorch", "onnx"):
        row = rows[(engine, "cpu")]
        assert row["task"] == "depth"
        assert row["note"] == ""
        for name, value in MEASURED_RESULTS[engine].items():
            assert row[name] == value


def test_a_benchmark_reports_refused_and_missing_engines_without_running_them(
    hub_configs, mocked_children
):
    rows = benchmark_rows_of(DEPTH_MODEL)
    assert ("ncnn", "cpu") not in rows
    assert NCNN_BATCH_BROADCAST in rows[("ncnn", "")]["note"]
    assert rows[("openvino", "")]["note"] == benchmark.NOT_INSTALLED_NOTE
    assert rows[("openvino", "")]["fps"] is None


def test_a_benchmark_child_error_becomes_the_row_note(hub_configs, mocked_children):
    row = benchmark_rows_of(DEPTH_MODEL)[("onnx", "cuda")]
    assert row["note"] == CHILD_ERROR
    assert (row["load_seconds"], row["latency_ms"], row["fps"]) == (None, None, None)


def test_a_benchmark_of_a_causal_lm_has_no_driver(hub_configs, mocked_children):
    (row,) = benchmark_rows_of(CAUSAL_LM_MODEL).values()
    assert row["task"] == "llm"
    assert row["note"] == benchmark.NO_DRIVER_NOTE


def test_every_ported_vision_task_has_a_benchmark_driver():
    text_tasks = {model_engines.LLM_TASK, model_engines.WHISPER_TASK}
    assert set(TASK_ENGINES) - text_tasks <= set(benchmark.BENCHMARK_DRIVERS)


def completed_child(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_a_child_result_is_the_last_stdout_line():
    result = {"load_seconds": 1.0, "latency_ms": 2.0, "fps": 500.0}
    stdout = f"Ultralytics 8.3 model summary\n{json.dumps(result)}\n\n"
    assert benchmark.child_result(completed_child(0, stdout)) == result


def test_a_crashed_child_becomes_an_error_note():
    killed = completed_child(-signal.SIGTERM)
    assert benchmark.child_result(killed) == {"error": "killed by SIGTERM"}
    traceback = "Traceback (most recent call last):\nValueError: no graph\n"
    failed = completed_child(1, "noise\n", traceback)
    assert benchmark.child_result(failed) == {
        "error": "exit code 1: ValueError: no graph"
    }
    assert benchmark.child_result(completed_child(0, "noise\n")) == {
        "error": "exit code 0"
    }


def test_a_frame_count_under_one_is_rejected():
    import mp_engines

    with pytest.raises(SystemExit):
        mp_engines.main(["yolo11n", "--benchmark", "--frames", "0"])
