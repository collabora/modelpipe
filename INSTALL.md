# Install

The short path is in the [README](README.md#install). This file has the rest: Fedora, Windows, Docker, the engine extras and custom plugins.

## Host Install

### Install distribution packages

#### Ubuntu
```
sudo apt update && sudo apt -y upgrade
sudo apt install -y python3-pip  python3-venv \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gir1.2-gst-plugins-bad-1.0 python3-gst-1.0 gstreamer1.0-python3-plugin-loader \
    libcairo2 libcairo2-dev git
```

#### Fedora

Adjust the Fedora version in the rpmfusion URLs.

```
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-42.noarch.rpm https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-42.noarch.rpm
sudo dnf update -y
sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda -y
```

```
sudo dnf upgrade -y
sudo dnf install -y python3-pip \
    python3-devel cairo cairo-devel cairo-gobject-devel pkgconfig git \
    gstreamer1-plugins-base gstreamer1-plugins-base-tools \
    gstreamer1-plugins-good gstreamer1-plugins-bad-free \
    gstreamer1-plugins-bad-free-devel python3-gstreamer1
```



#### Windows

1. Install Python 3.10 or later. JetPack 6 ships 3.10; the `jax` extras need 3.11.

2. Make a venv and install GStreamer and its Python bindings from the
   GStreamer project's wheels. The venv's `python`, `gst-launch-1.0` and
   `gst-inspect-1.0` find the GStreamer libraries on their own.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install "gstreamer-meta[cli]"
```

3. For an NVIDIA GPU, install the CUDA PyTorch before the package. PyPI's
   Windows torch is CPU-only, and once it is installed pip keeps it. The driver
   is enough, the wheels bring the CUDA runtime.

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

4. Install the package with the extras you need. Cairo is not installed on
   Windows, so add `skia` for `mp_overlay` on frames in system memory:

```powershell
pip install -e ".[yolo,skia]"
```

5. Set the environment. The wheels set `GST_PLUGIN_PATH_1_0`, which hides
   `GST_PLUGIN_PATH`. The Python plugin loader embeds the base interpreter, not
   the venv, so it needs `PYTHONPATH` to find the venv packages, the bundled
   `gi` and the plugins. `cudaconvert` and `cudascale` compile their kernels
   with NVRTC, which torch ships in `torch\lib`:

```powershell
$venv = "$PWD\.venv"
$env:GST_PLUGIN_PATH_1_0 = "$PWD\plugins"
$env:PYTHONPATH = "$venv\Lib\site-packages;$venv\Lib\site-packages\gstreamer_python\Lib\site-packages;$PWD\plugins\python"
$env:PATH = "$venv\Lib\site-packages\torch\lib;$env:PATH"
```

6. Check the elements load:

```powershell
gst-inspect-1.0 mp_objectdetector
gst-inspect-1.0 cudaconvert
```

   GStreamer caches the element list in `.venv\Temp\gstreamer-1.0`. If
   `cudaconvert` is missing after `PATH` changes, delete that directory.

The Windows GStreamer packages have no Vulkan plugin, so `mp_cudatovulkan` and
the Vulkan overlay are Linux only. Show video with `autovideosink` or
`d3d12videosink`.

### Manage Python packages

#### Python version

GStreamer's Python plugin loader embeds the system interpreter, so the venv must
use the same Python. Ubuntu 24.04 is 3.12, Fedora 42 and Ubuntu 26.04 are 3.14. A
mismatch shows up at run time as `No module named 'torch'`.

#### venv on the system Python

```
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[yolo]"
```

#### With uv

Point uv at the system Python, not a downloaded one:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python /usr/bin/python3 --system-site-packages
source .venv/bin/activate
uv sync --extra yolo
```

Do not pre-install torch from the PyTorch index here. `uv sync` resolves torch from
`uv.lock`, which on Linux already pulls the CUDA wheels, and replaces whatever was
installed before.

### Feature extras

The core install covers PyTorch, torchvision, transformers and OpenCV, which is
what most elements need. Each extra adds the packages for a group of elements:

- `yolo`: `mp_yolo`, `mp_yolo_pose`, `demo_soccer`, and the `bytetrack` and `botsort` types of `mp_tracker`
- `llm`: `mp_llm`, `mp_llmstreamfilter`, `mp_caption_qwen`, `mp_caption_phi`
- `awq`: AWQ quantized models, through gptqmodel, which builds its CUDA kernels with nvcc on the first model load
- `audio`: `mp_whisperspeechtts`, `mp_whisperlive`, `mp_demucs`, `mp_sepformer`
- `diffusion`: `mp_stablediffusion`
- `face`: `mp_face`
- `superres`: `mp_superres`
- `skia`: the `renderer=skia` option of `mp_overlay`
- `kafka`: `mp_kafkasink`
- `mqtt`: `mp_alert` with `mqtt-broker=` set
- `mcp`: the MCP server
- `vad`: `mp_vad`

```
uv sync --extra yolo --extra audio
```

`all` covers `yolo`, `llm`, `audio`, `diffusion`, `face`, `superres`, `kafka`, `mqtt`
and `vad`, plus the ONNX, tinygrad, llama.cpp, OpenVINO, TensorFlow and LiteRT
engines. It leaves out `awq`, which needs nvcc. TensorFlow has no 3.14 wheel
either, so `all` resolves on 3.12 and 3.13 only.

### Engine extras

Every engine beyond PyTorch is an extra: `onnx`, `onnx-gpu`, `openvino`,
`tensorflow`, `litert`, `tvm`, `tinygrad`, `mlx`, `executorch`, `llamacpp`,
`mlx-cpu`, `jax-cpu`, `jax-gpu`, `jax-tpu`, `iree`, `ncnn`, `rknn`.

```
uv sync --extra onnx
```

- ExecuTorch has no Python 3.14 wheel.
- TensorRT is `device=tensorrt` on the ONNX engine. It needs the `onnx-gpu` extra and the TensorRT 10 libraries:
  `uv pip install tensorrt-cu13-libs==10.16.1.11`, a 4 GB install. Tested with onnxruntime-gpu 1.30.
- `tensorrt=true` on `mp_yolo` and `mp_yolo_pose` needs the `tensorrt` extra, which installs TensorRT 11.
  onnxruntime-gpu 1.30 loads `libnvinfer.so.10`, so the extra and `device=tensorrt` on the ONNX engine cannot share a venv today.
  The fp16 conversion runs the model in onnxruntime, so add the `onnx` or `onnx-gpu` extra too.
- LiteRT runs a `.tflite` file with the `litert` extra alone. Exporting one, or loading a Keras model by name, needs `tensorflow` too.
- llama.cpp builds a CPU wheel. For CUDA: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`.
  The extra also installs pywhispercpp, which runs `mp_whispertranscribe` on the CPU. For CUDA:
  `GGML_CUDA=1 pip install git+https://github.com/absadiki/pywhispercpp`.
- MLX runs on Apple Silicon. On Linux use the `mlx-cpu` extra.
- The JAX extras bring in `keras-hub`, which runs `mp_depth` and `mp_clip` on `engine-name=jax`.
  uv leaves out its `tensorflow-text` dependency. pip installs it, and with it TensorFlow, which has no 3.14 wheel,
  so on 3.14 install the JAX extras with uv.
- NCNN takes `.param` and `.bin` files. Convert an ONNX model with
  `python -m onnxsim model.onnx model_sim.onnx` (`pip install onnx-simplifier`), then
  `onnx2ncnn model_sim.onnx model.param model.bin`.
- RKNN takes a `.rknn` file built for the board with RKNN Toolkit2. The `rknn`
  extra installs RKNN Toolkit Lite2 on AArch64 Python 3.12. Its version must
  match the board's `librknnrt`. Use `engine-name=rknn device=npu`, or select
  a core with `npu:0`, `npu:1`, `npu:2`, or `npu:all`.
- VART support is preliminary and targets the AMD VEK280 reference flow in
  Vitis AI 5.1. The runtime comes from the target image rather than a uv extra.
  Source `/etc/vai.sh`, then use `engine-name=vart device=npu` and pass
  `model-name=/path/to/snapshot::network-name`. Use `device=npu-only` to reject
  CPU subgraphs. The snapshot, NPU IP and target image must match.
- The ONNX engine with `device=npu` needs the [Ryzen AI SDK](https://ryzenai.docs.amd.com/) on AMD Ryzen AI laptops.
- The ONNX engine uses the macOS ONNX Runtime CoreML provider for
  `device=coreml`.
- ZenDNN speeds up ONNX Runtime and TensorFlow on AMD EPYC CPUs with
  `ZENDNN_INT8_SUPPORT=1` and `OMP_NUM_THREADS=$(nproc)`, no separate engine.

### flash-attn

Install the prebuilt wheel matching your Python, torch and CUDA from
[flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases), for example:

```
pip install ./flash_attn-2.8.3+cu128torch2.11-cp314-cp314-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

### MiGraphX (AMD ROCm)

#### Ubuntu

Needs ROCm:
```
sudo apt install migraphx
```

Make the `migraphx` module importable:
```
export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
```

#### Fedora

Fedora ships ROCm but not MiGraphX, so build it from source against the distro packages.
Tested on Fedora 43 with ROCm 6.4 and a Rembrandt APU (gfx1035).

```
sudo dnf install -y rocminfo rocm-hip-devel rocm-cmake rocblas-devel miopen-devel \
    half-devel protobuf-devel msgpack-devel json-devel sqlite-devel \
    python3-devel python3-pybind11 cmake ninja-build
rocminfo | grep -o -m1 'gfx[0-9a-f]*'
```

Build with the gfx name printed above as `GPU_TARGETS`. The patch adds Python 3.13 and 3.14
to MiGraphX's search list and fixes the build with rocMLIR disabled (Fedora does not package it).

```
cd $HOME/src
git clone --branch rocm-6.4.4 --depth 1 https://github.com/ROCm/AMDMIGraphX.git
cd AMDMIGraphX
git apply $HOME/src/modelpipe/extern/migraphx/rocm-6.4.4-fedora.patch
CXX=/usr/lib64/rocm/llvm/bin/clang++ cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1035 \
    -DMIGRAPHX_ENABLE_MLIR=OFF -DMIGRAPHX_USE_COMPOSABLEKERNEL=OFF -DMIGRAPHX_USE_HIPBLASLT=OFF \
    -DMIGRAPHX_ENABLE_PYTHON=ON -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$PWD/install
ninja -C build install
```

`hipcc` compiles every file as HIP and the protobuf sources fail to link, hence ROCm's
`clang++`. Make the `migraphx` module importable:
```
export PYTHONPATH=$HOME/src/AMDMIGraphX/install/lib:$PYTHONPATH
```

### ONNX Runtime on ROCm

AMD publishes the wheel outside PyPI, for Python 3.12 at most. It needs the
ROCm release in its URL and, on Fedora, `hipfft` and `rccl` beside the packages
above.

```
uv pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/onnxruntime_rocm-1.21.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

The first GPU compile of a model takes a couple of minutes. The `device=cpu` reference target
runs but is very slow on detection models.

### Clone repo

```
cd $HOME/src
git clone https://github.com/collabora/modelpipe.git
```

### Plugin path

```
echo 'export GST_PLUGIN_PATH=$HOME/src/modelpipe/plugins:$GST_PLUGIN_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Docker Install

The Dockerfiles mount the checkout from `$HOME/src/modelpipe`.

### Enable Docker GPU Support on Host

Skip this on CPU.


#### Ubuntu
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Fedora

```
sudo dnf install docker
sudo usermod -aG docker $USER
# Then either log out/in completely, or:
newgrp docker
```


```
# 1. Add NVIDIA Container Toolkit repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# 2. Remove Fedora's conflicting partial package (if present)
sudo dnf remove -y golang-github-nvidia-container-toolkit 2>/dev/null || true

# 3. Install the full NVIDIA Container Toolkit
sudo dnf install -y nvidia-container-toolkit

# 4. Configure Docker to use the NVIDIA runtime as default
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "runtimes": {
    "nvidia": {
      "path": "/usr/bin/nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
EOF

# 5. Fix Fedora's broken dockerd ExecStart (required!)
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/override.conf >/dev/null <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock
EOF

# 6. Reload and restart Docker
sudo systemctl daemon-reload
sudo systemctl restart docker

# 7. Verify it works
docker info --format '{{.DefaultRuntime}}'   # → should print: nvidia
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```


### Build Container

`docker build -f ./Dockerfile_ubuntu24 -t ubuntu24:latest .`

`docker build -f ./Dockerfile_ubuntu26 -t ubuntu26:latest .`

`docker build -f ./Dockerfile_fedora42 -t fedora42:latest .`


### Run Docker Container

Drop `--gpus all` on CPU.

`docker run -v ~/src/modelpipe/:/root/modelpipe -it --rm --gpus all --name ubuntu24 ubuntu24:latest /bin/bash`

or

`docker run -v ~/src/modelpipe/:/root/modelpipe -it --rm --gpus all --name ubuntu26 ubuntu26:latest /bin/bash`

or

`docker run -v ~/src/modelpipe/:/root/modelpipe -it --rm --gpus all --name fedora42 fedora42:latest /bin/bash`

Then set up the venv in the container shell as above.


## Post Install

Run `gst-inspect-1.0 python` to list the modelpipe elements.

## Custom Plugins

Your own elements can inherit the base classes below from a separate directory.

### Directory Structure

```
my_plugins/
  python/
    my_detector.py
    my_classifier.py
```

### Example: Custom Object Detector

```python
CAN_REGISTER_ELEMENT = True
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GObject", "2.0")
    from gi.repository import GObject, Gst, GstBase
    from base_objectdetector import BaseObjectDetector
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    print(f"my_detector not available: {e}")

if CAN_REGISTER_ELEMENT:
    class MyDetector(BaseObjectDetector):
        __gstmetadata__ = (
            "My Custom Detector",
            "Video/Filter",
            "A custom object detector",
            "Your Name",
        )

    GObject.type_register(MyDetector)
    __gstelementfactory__ = ("my_detector", Gst.Rank.NONE, MyDetector)
```

The plugin loader imports every module on the plugin path at startup, whether or
not the pipeline uses it, so import numpy, torch and the like inside methods.


### Environment Setup

Set both `GST_PLUGIN_PATH` (so GStreamer discovers your `.py` files) and `PYTHONPATH`
(so Python can import your modules):

```bash
export GST_PLUGIN_PATH=$HOME/src/modelpipe/plugins:$HOME/my_plugins:$GST_PLUGIN_PATH
export PYTHONPATH=$HOME/my_plugins/python:$PYTHONPATH
```

The loader adds only the first `python/` directory it finds to `sys.path`, so list
this checkout first for the base classes and put your own directory on `PYTHONPATH`.

### Available Base Classes

| Base Class | Module | Description |
|---|---|---|
| `BaseTransform` | `base_transform` | Base for all video transform elements |
| `BaseObjectDetector` | `base_objectdetector` | Object detection with bounding boxes |
| `BaseClassifier` | `base_classifier` | Image classification |
| `BaseCaption` | `base_caption` | Video/image captioning |
| `BaseLLM` | `base_llm` | Large language models |
| `BaseTranscribe` | `base_transcribe` | Speech-to-text transcription |
| `BaseTranslate` | `base_translate` | Text translation |
| `BaseTTS` | `base_tts` | Text-to-speech synthesis |
| `BaseSeparate` | `base_separate` | Audio source separation |

### Verify

```bash
gst-inspect-1.0 my_detector
```
