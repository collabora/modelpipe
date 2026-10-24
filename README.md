# ModelPipe

[![CI](https://github.com/collabora/modelpipe/actions/workflows/ci.yml/badge.svg)](https://github.com/collabora/modelpipe/actions/workflows/ci.yml)

ModelPipe runs a wide variety of ML models in GStreamer and
[Glass2Glass](https://github.com/Glass2GlassHQ/glass2glass) multimedia
pipelines.

ModelPipe provides one Python element per task, plus base classes and inference
engines for custom models. Write the pipeline in `gst-launch` format and run it
with `mp-launch`, using GStreamer 1.24 or later or Glass2Glass as the media
backend.

## Features

- Video
  - object detection and zero-shot detection
  - tracking
  - pose estimation
  - depth estimation
  - zero-shot classification with CLIP or SigLIP
  - segmentation with SAM2
  - OCR
  - face detection and recognition
  - detection-driven blur, pixelation and blackout
  - optical flow
  - super-resolution
  - action recognition
  - anomaly detection
  - captioning and vision-language models
  - embeddings and video search
- Audio
  - voice activity detection
  - transcription and translation
  - speech separation
  - text to speech
  - audio classification with CLAP
- Text
  - local and remote LLMs
  - incident digests
  - text to image
- Around them
  - alerts over webhooks and MQTT
  - clip recording
  - metadata sink and replay
  - Kafka sink
  - [MCP server](PIPELINES.md#mcp-server) for agents
- Engines
  - PyTorch by default
  - ONNX Runtime, TensorRT through ONNX Runtime, OpenVINO, LiteRT, TensorFlow
  - Apache TVM, tinygrad, Apple MLX, ExecuTorch, llama.cpp, JAX
  - MiGraphX, IREE, NCNN, Renesas DRP-AI, Rockchip RKNN, AMD VART
  - The task elements export their model once and run it on any engine with
    `engine-name=`, and on any accelerator with `device=`
  - CI runs every engine that installs from PyPI on the CPU and checks its
    output against PyTorch, on a YOLO export and on the task elements, see
    [Engine support](#engine-support)

## Engine support

Which task runs on which engine, as the parity tests check it on the CPU. A
footnote names what the engine or its converter refuses. The table is
generated from `plugins/python/engine/support_matrix.py`, regenerate it with
`python plugins/python/engine/support_matrix.py`.

<!-- support matrix start -->
| task | pytorch | onnx | openvino | tvm | tensorflow | tflite | ncnn | executorch | iree | tinygrad | migraphx | jax | llamacpp | mlx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolo | yes | yes | yes | partial [1] | partial [2] | partial [2] | yes | yes | yes | yes | yes | no | no | no |
| pose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | no | no |
| depth | yes | yes | yes | yes | yes | yes | no [3] | yes | yes | yes | yes | yes | no | no |
| clip | yes | yes | yes | yes | partial [4] | partial [4] | partial [5] | yes | yes | yes | yes | yes | no | no |
| anomaly | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | no | no |
| classifier | yes | yes | yes | yes | partial [6] | partial [6] | partial [7, 8, 9] | yes | yes | partial [10, 11] | partial [12, 13] | no | no | no |
| embedding | yes | yes | yes | yes | yes | yes | partial [14] | yes | yes | yes | yes | no | no | no |
| action | yes | yes | yes | yes | yes | yes | no [15] | no [16] | yes | yes | yes | no | no | no |
| superres | yes | yes | yes | yes | no [17] | no [17] | yes | yes | yes | yes | yes | no | no | no |
| optical_flow | yes | yes | yes | yes | no [18] | no [18] | no [15] | yes | yes | yes | yes | no | no | no |
| sam | yes | yes | yes | yes | no [19] | no [19] | no [20] | yes | yes | yes | no [21] | no | no | no |
| zero_shot | yes | yes | yes | yes | no [22] | no [22] | no [22] | yes | yes | no [23] | no [22] | no | no | no |
| llm | yes | yes | yes | no | no | no | no | no | no | no | no | no | yes | yes |
| whisper | yes | yes | yes | no | no | no | no | no | no | no | no | no | yes | yes |

1. yolo11n-seg: tvm's vm cannot emit the subtract from a constant
2. yolo11n-seg: onnx2tf multiplies the segmentation boxes along the wrong axis
3. ncnn rejects the broadcast across the batch axis
4. google/siglip-base-patch16-224: onnx2tf splits a constant vector one off
5. google/siglip-base-patch16-224: ncnn rejects the reshape that indexes the batch
6. convnext_tiny: onnx2tf scales convnext's layers along the wrong axis
7. convnext_tiny: ncnn has no Gather layer
8. vit_b_16: ncnn rejects the reshape that indexes the batch
9. swin_t: ncnn cannot permute a 5-rank tensor
10. convnext_tiny: tinygrad's Gather rejects nested constant indices
11. swin_t: tinygrad's ScatterND writes into a tensor with other uses
12. convnext_tiny: migraphx's convnext logits are uncorrelated with onnx's
13. swin_t: migraphx's swin logits are off by up to 2
14. facebook/dinov2-small: ncnn rejects the broadcast across the batch axis
15. ncnn runs one frame at a time
16. executorch's convolution takes 3-d or 4-d input
17. onnx2tf slices a shape with a negative size
18. the onnx2tf raft flow is tens of pixels off
19. onnx2tf tiles a 4-d tensor with one multiple
20. ncnn cannot permute a 5-rank tensor
21. migraphx resizes only nearest and linear
22. converting owlv2 runs past 12 GB
23. tinygrad's Gather rejects 2-d constant indices
<!-- support matrix end -->

`mp-engines` reads the same tables for one model. It takes a Hugging Face id, a
model name such as `yolo11n-pose` or `resnet18`, or a model file, and prints each
engine's status, its devices, and the extra to install when the engine is missing.
`*` marks a device this machine reaches, `?` one it does not probe. `--json` prints
the same as JSON.

```
$ mp-engines depth-anything/Depth-Anything-V2-Small-hf
depth
  pytorch     runs                                                       cpu* cuda*
  onnx        runs                                                       cpu* cuda* tensorrt* rocm migraphx npu? coreml
  openvino    runs                                                       cpu gpu npu                                     pip install modelpipe[openvino]
```

`--benchmark` times the model on each installed engine that runs it, on each device
marked `*`, and prints load time, milliseconds per frame and frames per second. Each
engine and device runs in its own process on random frames, two warmup frames and
then 20 timed ones, set with `--frames`. An engine that fails or crashes shows its
error in the note column. For action, one frame is a 16-frame window.

## Install

Ubuntu 24.04 or later:

```
sudo apt install -y python3-pip python3-venv \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gir1.2-gst-plugins-bad-1.0 python3-gst-1.0 gstreamer1.0-python3-plugin-loader \
    libcairo2 libcairo2-dev git
```

Then a venv on the system Python, which is the one GStreamer's plugin loader
embeds. `uv sync` installs PyTorch, with CUDA on Linux, and the `yolo` extra
adds the YOLO models the quick start uses.

```
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/collabora/modelpipe.git && cd modelpipe
uv venv --python /usr/bin/python3 --system-site-packages
uv sync --extra yolo
export GST_PLUGIN_PATH=$PWD/plugins:$GST_PLUGIN_PATH
gst-inspect-1.0 python
```

The last line lists the `mp_*` elements. [INSTALL.md](INSTALL.md) covers
Fedora, Windows, Docker, the feature extras, the other engines and custom
plugins.

## Quick start

Paths are relative to the checkout directory. Change `device=cuda` to `device=cpu` on a
machine without a GPU.

Track people with YOLO:

```
python mp-launch.py filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480 ! mp_yolo model-name=yolo11m device=cuda track=True ! mp_overlay ! videoconvert ! autovideosink
```

Estimate depth with Depth Anything V2:

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_depth model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda ! videoconvert ! autovideosink sync=false
```

Transcribe Korean speech and translate it to English. Transcripts are logged at
GStreamer info level:

```
GST_DEBUG=python:4 python mp-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! mp_whispertranscribe device=cuda language=ko translate=yes ! fakesink
```

Record a clip around every person detection:

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_yolo model-name=yolo11m device=cuda ! mp_alert rules='{"class":"person","min_score":0.7}' cooldown=8 draw-alert=false ! mp_alertrecorder location=alert-%s.webm seconds-before=2 seconds-after=3 ! mp_overlay ! videoconvert ! autovideosink sync=false
```

Replay a recorded detection stream through the tracker and overlay, with no
model and no GPU:

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_metareplay location=data/people.jsonl ! mp_tracker tracker-type=sort ! mp_overlay ! videoconvert ! autovideosink sync=false
```

[PIPELINES.md](PIPELINES.md) describes every pipeline, one or two per element, how to
run them under glass2glass, and how to run the MCP server.
[FREE_THREADING.md](FREE_THREADING.md) runs the yolo detector and tracker on
glass2glass under free-threaded Python 3.14t with the GIL off.

## Demos

Each demo is a launcher script and a README under `demo/`, with a file,
display and camera mode and a `HOST=gst` switch to run the same pipeline on
`gst-launch-1.0`.

- [Football](demo/football/README.md): a detector trained on broadcast
  football, a tracker, and an overlay with team colours, a focal-player HUD,
  kick detection and clip recording.
- [Baseball](demo/baseball/README.md): the stock COCO YOLO weights, a tracker,
  and an overlay that marks the batter, fielders, bat, glove and ball with a
  ball trail. No training needed.

```
demo/baseball/run.sh display
```
