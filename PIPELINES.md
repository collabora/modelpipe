# Pipelines

## Running a pipeline

`mp-launch` runs every pipeline in this file. Run it from the checkout with
any Python: it re-runs itself under `.venv` if it finds one, so the elements see
torch and the rest. Installing the package also puts a `mp-launch` on `PATH`.

```bash
python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale \
  ! video/x-raw,width=640,height=480 \
  ! mp_yolo model-name=yolo11m device=cuda:0 track=True \
  ! mp_overlay ! videoconvert ! autovideosink
```

`mp_overlay` draws with Cairo. `mp_overlay renderer=skia` draws with Skia
instead, from the `skia` extra. On Windows Skia is the default, Cairo is not
installed there.

### Choosing the backend

The ML elements run under GStreamer or under
[glass2glass](https://github.com/Glass2GlassHQ/glass2glass), selected by
`MP_BACKEND`. It defaults to `gst`, which is why the examples leave it off.
Prefix any of them with `MP_BACKEND=g2g` to run the same line under
glass2glass instead.

Under `gst` it runs GStreamer with `GST_PLUGIN_PATH` pointing at this checkout.
Under `g2g` it runs `g2g-launch-py`, rewriting the three things g2g spells
differently: an `mp_*` element becomes `pyelement` plus the module and class to
host, an element g2g implements itself becomes that one, and a raw-video caps
filter with no format gains `format=RGBA`. An element it cannot map is an error,
not a silent pass-through. It launches with `--threads`, one OS thread per
element as in GStreamer, so a CPU-bound element does not hold up the others.

Under `gst`, detections, classifications and tracking ids travel in the
GstAnalytics relation meta when the Python bindings have GstAnalytics
(GStreamer 1.24 or newer), and otherwise in a JSON chunk appended to the buffer
as an extra memory. `MP_ANALYTICS=meta` or `MP_ANALYTICS=chunk` forces one.
An element that writes a new buffer between the detector and the element
reading it, such as a converting `videoconvert` or an encoder, drops the chunk,
while the relation meta survives it.

Seven elements are native on g2g rather than hosted: `mp_overlay` becomes
`analyticsoverlay`, `mp_alert` becomes `analyticsalert`, `mp_digest` becomes
`textdigest`, and `mp_metasink`, `mp_metareplay`, `mp_alertrecorder` and
`mp_embeddingsink` drop the prefix. Their `location`, `rules`, `cooldown`,
`draw-alert`, `webhook-url`, `encoder`, `seconds-before`, `seconds-after`,
`window-seconds`, `source-id` and `model-name` carry over under the same names,
and the file formats match, so `search_video` reads an index either backend
wrote. g2g runs no MQTT, so `mp_alert`'s `mqtt-broker` and `mqtt-topic` are
refused rather than dropped.

On g2g, `nvdec` decodes H.264, H.265 and AV1 into CUDA memory. Put it directly
before `mp_classifier`, `mp_yolo`, `mp_yolo_pose` or `mp_depth` and
mp-launch hosts that element with `cuda-frames=true format=NV12`: the element
converts the NV12 frame to RGB on the GPU and runs the same device-frame path as
on gst. The frame leaves it still CUDA NV12, so a `mp_tracker` straight after
it takes `cuda-frames=true` too. mp-launch does this only from `nvdec` on, so
a line with `decodebin` keeps the element on system memory, spell `qtdemux !
h264parse ! nvdec`. g2g has no GPU scaler or RGBA conversion, so mp-launch
drops gst's `cudaconvert`, `cudascale` and `video/x-raw(memory:CUDAMemory)` caps
filter after `nvdec`, and the element sees the decoded size. Put `videoconvert`
ahead of `mp_overlay`. An element that writes the frame, such as `mp_depth`
with `visualize=true` or `mp_superres`, needs `videoconvert` between `nvdec`
and it.
Build `g2g-launch-py` with `--features ml,g2g-plugins/nvdec,g2g-plugins/cuda`
for `nvdec`.

`analyticsoverlay` has its own properties (`show-label`, `show-track`,
`show-score`, `show-trail`, `trail-length`, `thickness`, `mask-alpha`), so only
`mp_overlay`'s `tracking` carries over, as `show-track`.

Build `g2g-launch-py` from a glass2glass checkout and put it on `PATH`, or point
`G2G_LAUNCH` at it:

```bash
PYO3_PYTHON=$(which python) cargo build --release -p g2g-python --features ml \
  --bin g2g-launch-py
```

## By task

One or two lines per element. Paths are relative to the checkout.

### Classification

```
python mp-launch.py  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_classifier model-name=resnet18 device=cuda !  videoconvert !  autovideosink
```

`mp_classifier` also takes frames in CUDA memory. On the pytorch engine with
`device=cuda`, a frame decoded by `nvh264dec` and converted by `cudaconvert`
reaches the model as a tensor on the GPU, with no copy to the host. The element
offers CUDA memory only when its engine reads tensors, which is the pytorch
engine. With an exported-model engine such as onnx it drops those caps, so a
pipeline feeding it CUDA memory fails to link when it is built rather than at
the first frame. Put `cudadownload` before the element for those engines.

`mp_yolo` and `mp_yolo_pose` take CUDA memory on the pytorch engine too. The
frame is scaled and padded to the model's input size on the GPU, and with
`track=false` the model and its non-maximum suppression run there without the
ultralytics predictor, copying only the kept boxes back, which cuts the yolo
element from about 15 ms to 9 ms a frame for yolo11n at 640x480 on an RTX 3060
Laptop. `track=true` still goes through ultralytics' tracker. Set
`tensorrt=true` to run the same path on a half precision TensorRT engine built
from the weights with the `tensorrt` extra, which cuts yolo11n to about 4.5 ms
a frame on the same GPU. The first frame builds the engine, which takes about
two minutes, and later starts load it from `~/.cache/modelpipe/tensorrt`.
The engine is built for the letterboxed shape of the frame, 480x640 for a
640x480 frame, so its boxes stay within a pixel or two of the pytorch weights,
and a new frame size builds another engine. The boxes and keypoints are
mapped back to the frame, so the analytics metadata matches the system-memory
path. Segmentation
and oriented-box models refuse these frames. On a CUDA frame `mp_yolo_pose`
with `visualize=true` draws the keypoints and skeleton in place with torch, in
ultralytics' colours but without the boxes and labels, so put `mp_overlay`
after it for those. `mp_tracker`
reads only the metadata, so it passes CUDA memory through, and `mp_overlay`
draws on CUDA frames in place with torch, so `nvh264dec ! cudaconvert !
cudascale ! mp_yolo engine-name=pytorch device=cuda ! mp_tracker !
mp_overlay` never copies a frame to the host. On CUDA frames the overlay
ignores the `renderer` property and draws boxes, labels, tracking squares and
trails without antialiasing, with Pillow's default font. `cudadownload` drops
the analytics metadata, so it goes after the last element that reads it.

`mp_overlay` also draws on `video/x-raw(memory:VulkanImage)` RGBA and BGRA
frames, with shaders on the Vulkan device. `mp_cudatovulkan` copies CUDA frames
into Vulkan images on the same GPU, device to device, so a CUDA detector feeds
the Vulkan overlay without a copy to the host. It gives its Vulkan device to
the Vulkan elements after it, the device of the GPU that CUDA device 0 is on,
and a frame on another CUDA device fails. The element exists on the gst backend
on Linux only, the Windows GStreamer packages have no Vulkan plugin. The
example downloads the frames to show them in `gtkwaylandsink`, a resizable
window. `vulkansink` opens a 320x240 window on Wayland, and shrinking the frame
that far drops the 2 pixel box edges.

```
python mp-launch.py filesrc location=data/people.mp4 ! qtdemux ! h264parse ! nvh264dec ! cudaconvert ! cudascale ! capsfilter caps="video/x-raw(memory:CUDAMemory),format=RGBA,width=1280,height=720" ! mp_yolo model-name=yolo11n engine-name=pytorch device=cuda ! mp_tracker ! mp_cudatovulkan ! mp_overlay ! vulkandownload ! videoconvert ! gtkwaylandsink
```

`mp_depth` takes CUDA memory with `engine-name=pytorch device=cuda` too: the
image processor resizes and normalises the frame on the GPU and the colormap
frame is written back into the CUDA buffer, with only the uint8 depth blob
going to the host, about 5 ms a frame less than through system memory. Add
`cudadownload` after it to save or show the colormap. `mp_superres` takes
CUDA memory on the same engine: the frame reaches Real-ESRGAN as a tensor and
comes back to the frame size with an antialiased bicubic resize on the GPU
instead of Lanczos, then is written into the CUDA buffer. Every other element
takes frames in system memory.

Each pytorch element on `device=cuda` runs its inference on a CUDA stream of its
own, so two elements on the branches of a `tee` overlap on the GPU. Set
`stream-priority` to a negative number to run one element's work first, lower is
more urgent as in CUDA and the value is clamped to the range the device allows.

```
python mp-launch.py filesrc location=data/people.mp4 ! qtdemux ! h264parse ! nvh264dec ! cudaconvert ! cudascale ! capsfilter caps="video/x-raw(memory:CUDAMemory),format=RGBA,width=640,height=480" ! mp_yolo model-name=yolo11n engine-name=pytorch device=cuda ! mp_tracker ! fakesink
```

```
python mp-launch.py filesrc location=data/people.mp4 ! qtdemux ! h264parse ! nvh264dec ! cudaconvert ! capsfilter caps="video/x-raw(memory:CUDAMemory),format=RGBA" ! mp_classifier model-name=resnet50 engine-name=pytorch device=cuda ! fakesink
```


### Torch Compile

Every PyTorch element takes `compile=True` to run the model through `torch.compile`,
which trades a slow first frame for higher steady throughput.

#### Classification with torch.compile

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale \
  ! video/x-raw,width=640,height=480 \
  ! mp_classifier model-name=resnet18 device=cuda compile=True \
  ! videoconvert ! autovideosink
```

#### Object detection with torch.compile

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale \
  ! video/x-raw,width=640,height=480 \
  ! mp_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda compile=True \
  ! videoconvert ! mp_overlay ! videoconvert ! autovideosink
```


### Object Detection

#### TorchVision

`mp_objectdetector` takes any torchvision detection model name, such as
`fasterrcnn_resnet50_fpn` or `ssdlite320_mobilenet_v3_large`.

##### fasterrcnn

`python mp-launch.py  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! videoconvert ! mp_overlay ! videoconvert ! autovideosink`

##### fasterrcnn/kafka

`mp_kafkasink` sends one JSON message per frame with `source_id`, `timestamp`
(the UTC time it was sent) and `detections`, each with `label`, `score` and
`box`. `data/mp_object_detector.json` is the JSON Schema for that message.

From the host:

```
python mp-launch.py  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! mp_kafkasink broker=localhost:29092 topic=test-kafkasink-topic
```

From a container:

```
python mp-launch.py  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! mp_kafkasink broker=kafka:9092 topic=test-kafkasink-topic
```


#### maskrcnn

```
python mp-launch.py   filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! mp_maskrcnn device=cuda batch-size=4 model-name=maskrcnn_resnet50_fpn ! videoconvert ! mp_overlay ! videoconvert ! autovideosink
```

#### yolo with tracking

```
python mp-launch.py   filesrc location=data/soccer_tracking.mp4 ! decodebin !  videoconvertscale ! video/x-raw,width=640,height=480 ! mp_yolo model-name=yolo11m device=cuda:0 track=True ! mp_overlay  ! videoconvert ! autovideosink
```

A batch from `mp_streammux` is the source frames in order, then a
`MODELPIPE` memory with each frame's source index. A detector between the
mux and `mp_streamdemux` appends its analytics chunk or a blob after that
memory, and the demux finds the source indices by their header, so batching
works under both `MP_ANALYTICS` stores.

```
python mp-launch.py   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480,format=RGB ! mp_streammux name=mux   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480,format=RGB ! mux.   mux. ! mp_yolo model-name=yolo11m device=cuda:0 track=True ! mp_streamdemux name=demux   demux. ! queue ! videoconvert ! mp_overlay ! videoconvert ! autovideosink sync=false   demux. ! queue ! videoconvert ! mp_overlay ! videoconvert !  autovideosink sync=false

```

```
python mp-launch.py filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480 ! demo_soccer model-name=yolo11m device=cuda:0 imgsz=640 hires-fallback=false ! mp_overlay ! videoconvert ! autovideosink
```

`imgsz` defaults to 1280 and `hires-fallback` runs a second 1536 pixel pass for the
ball, which together triple the time per frame. On a 640x480 input neither finds
more than `imgsz=640` does.


#### ONNX Engine

Export a YOLO11 model with ultralytics:

```
yolo export model=yolo11m.pt format=onnx
```

##### YOLO11m ONNX object detection with overlay

YOLO takes channels first, hence `input-format=nchw`. `post-process=anchor_free`
decodes the raw `[B, 4+nc, anchors]` output into boxes with NMS.

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_objectdetector engine-name=onnx model-name=yolo11m.onnx device=cpu \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! mp_overlay ! videoconvert ! autovideosink
```

##### Generic ONNX passthrough (logs raw inference output)

`mp_inference` runs any model and logs the raw output:

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_inference engine-name=onnx model-name=yolo11m.onnx device=cpu \
  ! fakesink
```

It takes every `engine-name` below.

##### Task elements on a builtin engine

`mp_depth`, `mp_clip`, `mp_anomaly`, `mp_embedding`, `mp_action`,
`mp_superres`, `mp_optical_flow`, `mp_sam`, `mp_yolo`, `mp_yolo_pose`
and `mp_zeroshotdetector` take the Hugging Face or ultralytics model name
with any of `engine-name=onnx`, `openvino`, `tvm`, `tensorflow`, `tflite`,
`ncnn`, `executorch`, `iree`, `tinygrad` or `migraphx`, and `mp_depth` and
`mp_clip` also `jax`. The first run exports the model to
`~/.cache/modelpipe/onnx/` under a name that carries the torch, torchvision
and transformers versions (ultralytics for YOLO), so a library upgrade exports
again, and an engine that does not read ONNX converts
it once more into `~/.cache/modelpipe/<engine>/`. ExecuTorch exports a
`.pte` from the torch graph instead, and jax loads a keras-hub preset.
`track=True` needs the PyTorch engine. `mp_whispertranscribe` runs on onnx and
openvino through their genai runtimes, on llamacpp through whisper.cpp and on mlx
through mlx-whisper, see [Transcription](#transcription).

Every engine runs depth, CLIP, SigLIP, YOLO and YOLO pose against the PyTorch
results in `tests/test_exported_models.py`, except: ncnn rejects depth (a
broadcast across the batch axis) and SigLIP (a reshape that indexes the
batch), so CLIP and the convolutional exports run on it (YOLO, pose, anomaly,
super resolution); the tensorflow and tflite engines get SigLIP wrong, onnx2tf
splits one of its constants off by one; TVM runs these graphs unscheduled on
the CPU, a depth frame takes over a minute. The other six task elements are
verified on the ONNX engine only, and three of them do not run on tensorflow or
tflite: onnx2tf cannot convert the super resolution export (its height and
width are dynamic) or the SAM encoder, and it converts optical flow wrong.

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! mp_depth engine-name=onnx model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda \
    ! videoconvert ! autovideosink sync=false
```

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale \
  ! video/x-raw,width=640,height=480 \
  ! mp_yolo engine-name=onnx model-name=yolo11m device=cuda \
  ! mp_overlay ! videoconvert ! autovideosink
```

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! mp_clip engine-name=onnx model-name=openai/clip-vit-base-patch32 device=cuda \
              labels="person, bicycle, car, dog, cat" top-k=3 \
    ! videoconvert ! mp_overlay ! videoconvert ! autovideosink sync=false
```

#### OpenVINO Engine


```
yolo export model=yolo11m.pt format=openvino
```


##### YOLO11m OpenVINO object detection with overlay

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_objectdetector engine-name=openvino \
              model-name=yolo11m_openvino_model/yolo11m.xml device=cpu \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! mp_overlay ! videoconvert ! autovideosink
```

`device=GPU` targets an Intel GPU. OpenVINO device names are uppercase.

#### LiteRT (TFLite) Engine


```
yolo export model=yolo11m.pt format=tflite
```


##### YOLO11m TFLite object detection with overlay

The engine reads the input layout from the model and scales the normalized boxes an
ultralytics export returns back to pixels.

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_objectdetector engine-name=tflite \
              model-name=yolo11m_saved_model/yolo11m_float32.tflite device=cpu \
              post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! mp_overlay ! videoconvert ! autovideosink
```

#### TensorFlow Engine


```
yolo export model=yolo11m.pt format=saved_model
```

##### YOLO11m TensorFlow object detection with overlay

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_objectdetector engine-name=tensorflow \
              model-name=yolo11m_saved_model device=cuda \
              post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! mp_overlay ! videoconvert ! autovideosink
```

#### tinygrad Engine

tinygrad runs `.onnx` files.

##### YOLO11m ONNX object detection with tinygrad

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_objectdetector engine-name=tinygrad model-name=yolo11m.onnx device=cuda \
              post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! mp_overlay ! videoconvert ! autovideosink
```

##### ResNet18 classification with tinygrad on GPU

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! mp_classifier model-name=resnet18 device=cuda engine-name=tinygrad \
  ! fakesink
```

##### tinygrad on CPU

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! mp_classifier model-name=resnet18 device=cpu engine-name=tinygrad \
  ! fakesink
```

#### TVM Engine

TVM takes a compiled `.so` or `.tar`, or an `.onnx` file.

##### TorchVision model compiled with TVM

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! mp_classifier model-name=resnet18 device=cuda engine-name=tvm \
  ! fakesink
```

##### Pre-compiled TVM model (.so)

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_inference engine-name=tvm model-name=compiled_model.so device=cuda \
  ! fakesink
```

#### ExecuTorch Engine

ExecuTorch runs `.pte` files, for example `yolo export model=yolo11n.pt format=executorch`.

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! mp_inference engine-name=executorch model-name=model.pte device=cpu \
  ! fakesink
```

#### llama.cpp Engine

llama.cpp runs `.gguf` files, a local path or a Hub repo with the quant that
ends the file name.

```
python mp-launch.py filesrc location=data/prompt_for_llm.txt \
  ! mp_llm engine-name=llamacpp model-name=Qwen/Qwen3-0.6B-GGUF:Q8_0 device=cpu \
  ! fakesink
```

#### JAX/Flax Engine

`mp_depth` and `mp_clip` run the keras-hub port of their Hugging Face model
on JAX. CLIP keeps its text tower on PyTorch.

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! mp_depth engine-name=jax model-name=depth-anything/Depth-Anything-V2-Small-hf device=cpu \
    ! videoconvert ! autovideosink sync=false
```

#### MiGraphX Engine

MiGraphX takes ONNX files and needs the ROCm install above.

##### YOLO11m MiGraphX object detection with overlay

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_objectdetector engine-name=migraphx model-name=yolo11m.onnx device=gpu \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! mp_overlay ! videoconvert ! autovideosink
```

##### MiGraphX on CPU (reference target)

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_inference engine-name=migraphx model-name=yolo11m.onnx device=cpu \
  ! fakesink
```

#### IREE Engine

IREE takes a compiled `.vmfb`, or an `.onnx` it compiles at load time, for `hip`,
`vulkan`, `cuda` or `cpu`.

##### IREE on AMD GPU (ROCm/HIP)

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_inference engine-name=iree model-name=yolo11m.onnx device=hip \
  ! fakesink
```

##### IREE on Vulkan (any GPU vendor)

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_inference engine-name=iree model-name=yolo11m.onnx device=vulkan \
  ! fakesink
```

##### IREE with pre-compiled module

```
# Pre-compile: iree-compile model.mlir --iree-hal-target-device=hip -o model.vmfb
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_inference engine-name=iree model-name=model.vmfb device=hip \
  ! fakesink
```

#### NCNN Engine (Vulkan)

NCNN takes a `.param` with its `.bin` alongside and runs on any Vulkan GPU or the CPU.

##### NCNN on Vulkan GPU

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_inference engine-name=ncnn model-name=yolo11m.param device=vulkan \
  ! fakesink
```

##### NCNN on CPU

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_inference engine-name=ncnn model-name=yolo11m.param device=cpu \
  ! fakesink
```

#### RKNN Engine on Rockchip NPU

RKNN takes a `.rknn` model built for the board with RKNN Toolkit2. It runs
through RKNN Toolkit Lite2 and the board's `librknnrt`. Use `npu:0`, `npu:1`,
`npu:2`, or `npu:all` to select NPU cores on supported chips.

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_inference engine-name=rknn model-name=yolo11m.rknn device=npu \
              input-format=nhwc post-process=none \
  ! fakesink
```

#### VART Engine on AMD VEK280

This preliminary path uses the Python VART API from the Vitis AI 5.1 target
image. Write the model as `snapshot-directory::network-name`. The snapshot,
NPU IP and target image must come from the same reference flow.

```
source /etc/vai.sh
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=224,height=224" \
  ! mp_inference engine-name=vart model-name=snapshot.NPU.resnet50.TF::resnet50 device=npu \
              input-format=nhwc post-process=none \
  ! fakesink
```

#### ONNX Runtime with TensorRT

`device=tensorrt` runs the model through TensorRT, and `device=tensorrt-fp16`
in half precision. The first start builds the TensorRT engine, which takes about
ten minutes for YOLO11m on a laptop GPU. It is cached under
`~/.cache/modelpipe/tensorrt`, so later starts take a second.

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_objectdetector engine-name=onnx model-name=yolo11m.onnx device=tensorrt-fp16 \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! mp_overlay ! videoconvert ! autovideosink
```

#### ONNX Runtime on AMD GPUs (ROCm)

`device=rocm` picks the ROCm execution provider, falling back to MIGraphX's,
and `device=migraphx` MIGraphX only. MIGraphX 6.4 rejects the resize attribute
ONNX Runtime adds to a graph, so it fails on YOLO and the task element exports.
On an APU ROCm does not list, such as the Rembrandt gfx1035, the ROCm provider
runs with `HSA_OVERRIDE_GFX_VERSION=10.3.0`.

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_objectdetector engine-name=onnx model-name=yolo11m.onnx device=rocm \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! mp_overlay ! videoconvert ! autovideosink
```

#### ONNX Runtime on AMD Ryzen AI NPU

`device=npu` needs the Ryzen AI SDK.

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_objectdetector engine-name=onnx model-name=yolo11m.onnx device=npu \
              input-format=nchw post-process=anchor_free \
  ! videoconvert ! "video/x-raw,format=RGBA" \
  ! mp_overlay ! videoconvert ! autovideosink
```

#### PyTorch on AMD GPUs (ROCm)

A ROCm torch answers to `device=cuda`:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda \
  ! videoconvert ! mp_overlay ! videoconvert ! autovideosink
```

With `torch.compile`, which uses Triton on AMD:

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale \
  ! "video/x-raw,format=RGB,width=640,height=640" \
  ! mp_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda compile=True \
  ! videoconvert ! mp_overlay ! videoconvert ! autovideosink
```

### Zero-Shot Object Detection

`mp_zeroshotdetector` detects the classes named in its `labels` property, so
`mp_alert`, `mp_tracker`, `mp_overlay` and `mp_metasink` work on a class
no detector was trained for. Detections carry the label text, as
`stream_0_handbag`.

Supported models:
```
google/owlv2-base-patch16-ensemble  (default)
IDEA-Research/grounding-dino-tiny   (faster, smaller)
```

#### Alert on a class no detector was trained for

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_zeroshotdetector device=cuda labels="person, handbag" confidence=0.2 \
  ! mp_alert rules='{"class":"handbag"}' \
  ! mp_overlay ! videoconvert ! autovideosink sync=false
```

### Pose Estimation

`mp_yolo_pose` takes any YOLO pose model:
```
yolo11n-pose  (fastest)
yolo11s-pose
yolo11m-pose  (best accuracy)
```

#### YOLO pose with skeleton visualization (rendered on frame)

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! mp_yolo_pose model-name=yolo11n-pose device=cuda \
    ! videoconvert ! autovideosink sync=false
```

#### YOLO pose with bounding box overlay (metadata only, no in-element rendering)

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! mp_yolo_pose model-name=yolo11n-pose device=cuda visualize=false \
    ! videoconvert ! mp_overlay ! videoconvert ! autovideosink sync=false
```

### Depth Estimation

`mp_depth` takes the Depth Anything V2 models:
```
depth-anything/Depth-Anything-V2-Small-hf  (fastest, ~100 MB)
depth-anything/Depth-Anything-V2-Base-hf
depth-anything/Depth-Anything-V2-Large-hf  (most accurate)
```

Colormaps: `inferno` (default), `jet`, `viridis`, `plasma`, `magma`.

#### DepthAnything V2 with inferno colormap

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! mp_depth model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda \
    ! videoconvert ! autovideosink sync=false
```

#### DepthAnything V2 with jet colormap

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! mp_depth model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda colormap=jet \
    ! videoconvert ! autovideosink sync=false
```

#### Depth with reduced compute via frame-stride

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! mp_depth model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda frame-stride=2 \
    ! videoconvert ! autovideosink sync=false
```

#### Depth with original video side-by-side (tee)

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! tee name=t \
    t. ! queue ! mp_depth model-name=depth-anything/Depth-Anything-V2-Small-hf device=cuda ! videoconvert ! autovideosink sync=false \
    t. ! queue ! videoconvert ! autovideosink sync=false
```

### Zero-Shot Classification (CLIP / SigLIP)

`mp_clip` classifies each frame against the text labels you pass.

Supported models:
```
openai/clip-vit-base-patch32       (default, ~600 MB)
openai/clip-vit-large-patch14      (more accurate, ~1.7 GB)
google/siglip-base-patch16-224     (SigLIP, better zero-shot accuracy)
google/siglip-large-patch16-384    (SigLIP large)
```

#### CLIP with custom labels

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! mp_clip model-name=openai/clip-vit-base-patch32 device=cuda \
              labels="person, bicycle, car, dog, cat" top-k=3 \
    ! videoconvert ! mp_overlay ! videoconvert ! autovideosink sync=false
```

#### SigLIP (better zero-shot accuracy than CLIP)

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! mp_clip model-name=google/siglip-base-patch16-224 device=cuda \
              labels="people walking, empty street, crowd, indoor scene" top-k=1 \
    ! videoconvert ! mp_overlay ! videoconvert ! autovideosink sync=false
```

#### CLIP with threshold (only report labels above 20% confidence)

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue \
    ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
    ! mp_clip model-name=openai/clip-vit-base-patch32 device=cuda \
              labels="person, bicycle, car, dog, cat" threshold=0.2 \
    ! videoconvert ! mp_overlay ! videoconvert ! autovideosink sync=false
```

### Voice Activity Detection

#### Standalone VAD with metadata (pass-through, speech probability attached to buffers)

```
python mp-launch.py pulsesrc ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! mp_vad threshold=0.7 ! fakesink
```

`mp_whispertranscribe` has its own VAD to split speech into clips, so `mp_vad` is not needed in front of it.

### Transcription

Transcripts are logged at GStreamer info level. Run with `GST_DEBUG=python:4` to see them.

`mp_whispertranscribe` runs faster-whisper on the PyTorch engine. With
`engine-name=onnx` it builds the Hugging Face checkpoint for onnxruntime-genai,
and with `engine-name=openvino` it exports it for openvino-genai, each once into
`~/.cache/modelpipe/`. `engine-name=llamacpp` downloads the whisper.cpp ggml
file of the same size, and `engine-name=mlx` the mlx-community conversion. A size
name such as `medium` means `openai/whisper-medium` on those engines.

```
python mp-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! mp_whispertranscribe engine-name=onnx device=cpu language=ko ! fakesink
```

```
python mp-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! mp_whispertranscribe engine-name=openvino device=cpu language=ko ! fakesink
```

faster-whisper's CTranslate2 wheel loads CUDA 12 cuBLAS. With a CUDA 13 torch install, add it to the venv and the loader path. On Linux:

```
pip install nvidia-cublas-cu12
export LD_LIBRARY_PATH=$(python -c "import nvidia.cublas, os; print(os.path.join(nvidia.cublas.__path__[0], 'lib'))"):$LD_LIBRARY_PATH
```

On Windows add the same package's `bin` directory to `PATH` instead.

#### live microphone (Linux)

`pulsesrc` works on PulseAudio (Ubuntu) and on PipeWire's Pulse server (Fedora). `pipewiresrc` does not work with GStreamer 1.26 (PipeWire 1.4): it stalls after a few buffers even into `fakesink`, because its buffer timestamps and the clock it provides disagree and GstBaseSrc's clock wait hangs. `do-timestamp=true` helps but is not consistent.

```
python mp-launch.py pulsesrc ! audioconvert ! audioresample ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! mp_whispertranscribe device=cuda language=en ! fakesink
```

```
python mp-launch.py pipewiresrc do-timestamp=true ! audioconvert ! audioresample ! audio/x-raw,format=S16LE,rate=16000,channels=1 ! mp_whispertranscribe device=cuda language=en ! fakesink
```

If nothing is transcribed, check which port the source captures from with `pactl list sources | grep "Active Port"`. Laptop combo jacks often report a plugged-in headset mic as not available and keep the internal mic. Force it with `pactl set-source-port <source> analog-input-mic`. On PipeWire, WirePlumber then drops that source as the default, so name it: `pulsesrc device=<source>` or `pipewiresrc target-object=<source>`. On Windows and macOS use `autoaudiosrc` and pick the input device in the OS sound settings.

#### transcription with initial prompt set

```
python mp-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! mp_whispertranscribe device=cuda language=ko initial_prompt = "Air Traffic Control은, radar systems를,  weather conditions에, flight paths를, communication은, unexpected weather conditions가, continuous training을, dedication과, professionalism" ! fakesink
```

#### translation to English

```
python mp-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! mp_whispertranscribe device=cuda language=ko translate=yes ! fakesink
```

#### demucs audio separation

```
python mp-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! audioresample ! mp_demucs device=cuda ! wavenc ! filesink location=separated_vocals.wav
```

#### sepformer audio separation

`stem` picks the output, `vocals` by default.

```
python mp-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! audioresample ! mp_sepformer device=cuda ! wavenc ! filesink location=separated_vocals.wav
```


#### whisperspeechtts

```
python mp-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! mp_whispertranscribe device=cuda language=ko translate=yes ! mp_whisperspeechtts device=cuda ! audioconvert ! wavenc ! filesink location=output_audio.wav
```

#### mariantranslate

```
python mp-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! mp_whispertranscribe device=cuda language=ko translate=yes ! mp_mariantranslate device=cuda model-name=Helsinki-NLP/opus-mt-en-fr ! fakesink
```

`model-name` is the Hugging Face model id. `Helsinki-NLP/opus-mt-en-fr` is English to French. There is no `Helsinki-NLP/opus-mt-en-ko`. English to Korean is `Helsinki-NLP/opus-mt-tc-big-en-ko`.

`opus-mt-en-fr` lists its English pieces in `vocab.json` and the element uses those ids. `opus-mt-tc-big-en-ko` lists Korean pieces there, so the element encodes the English with the source SentencePiece ids. On CUDA, "Air Traffic Control is the key element of modern aviation industry." is "Le contrôle du trafic aérien est l'élément clé de l'industrie aéronautique moderne." in French and "항공 교통 통제는 현대 항공 산업의 핵심 요소입니다." in Korean.

```
python mp-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! mp_whispertranscribe device=cuda language=ko translate=yes ! mp_mariantranslate device=cuda model-name=Helsinki-NLP/opus-mt-tc-big-en-ko ! fakesink
```


#### whisperlive

`python mp-launch.py filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! mp_whisperlive device=cuda language=ko translate=yes llm-model-name="Qwen/Qwen2.5-0.5B-Instruct" ! audioconvert ! wavenc ! filesink location=output_audio.wav`

With more than 6 GB of GPU memory, `llm-model-name="microsoft/phi-2"` also works well. On a 6 GB card it runs out of memory next to whisper and WhisperSpeech.

### LLM

Gated models need `hf auth login` first.

`python mp-launch.py filesrc location=data/prompt_for_llm.txt !  mp_llm device=cuda model-name="microsoft/phi-2" ! fakesink`

`mp_llm` takes the Hugging Face name on `engine-name=pytorch`, `onnx` and
`openvino`. On `onnx` the first run builds the model once with
onnxruntime-genai into `~/.cache/modelpipe/onnx-genai/`, int4 weights for
`device=cpu` and fp16 for `device=cuda`. On `openvino` the first run exports it
once with optimum-intel into `~/.cache/modelpipe/openvino-genai/` with int4
weights, and openvino-genai runs it. `engine-name=mlx` takes an mlx-lm model
and `engine-name=llamacpp` a `.gguf`, as a file or as a Hub repo with its
quant, `Qwen/Qwen3-0.6B-GGUF:Q8_0`.

`python mp-launch.py filesrc location=data/prompt_for_llm.txt ! mp_llm engine-name=onnx device=cpu model-name="Qwen/Qwen3-0.6B" ! fakesink`

`python mp-launch.py filesrc location=data/prompt_for_llm.txt ! mp_llm engine-name=openvino device=cpu model-name="Qwen/Qwen3-0.6B" ! fakesink`

#### Remote LLM

`mp_llm_remote` posts text to an LLM server. The OpenAI-compatible
`/v1/chat/completions` path works against Ollama, llama.cpp and vLLM. Without
`url=` it uses Ollama's native `/api/generate`, and it picks the request format
from the path.

##### Basic call

```
python mp-launch.py filesrc location=data/prompt_for_llm.txt \
  ! "text/x-raw,format=utf8" \
  ! mp_llm_remote url=http://localhost:11434/v1/chat/completions \
    model-name=llama3 \
  ! fakesink
```

##### With a system prompt and a custom model

```
python mp-launch.py filesrc location=data/prompt_for_llm.txt \
  ! "text/x-raw,format=utf8" \
  ! mp_llm_remote url=http://localhost:11434/v1/chat/completions \
    model-name=qwen3:8b \
    system-prompt="You are a helpful assistant. Answer concisely." \
    temperature=0.5 \
  ! fakesink
```

### Incident Digest

`mp_digest` collects the text buffers of a time window into one text buffer,
so `mp_llm` summarises half a minute of captions rather than a single line.
The window closes when a buffer's timestamp passes it, and the open window is
flushed at end of stream.

```
python mp-launch.py filesrc location=data/soccer_single_camera.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480 ! tee name=t t. ! queue ! textoverlay name=overlay wait-text=false ! videoconvert ! autovideosink t. ! queue leaky=2 max-size-buffers=1 ! videoconvertscale ! video/x-raw,width=240,height=180 ! mp_caption_qwen device=cuda:0 prompt="In one sentence, describe what you see?" model-name="Qwen/Qwen2.5-VL-3B-Instruct-AWQ" name=cap cap.src ! fakesink async=0 sync=0 cap.text_src ! queue ! mp_digest window-seconds=30 ! mp_llm model-name="Qwen/Qwen3-0.6B" device=cuda system-prompt="You receive every caption of the last thirty seconds. Write one paragraph describing what happened, and NEVER mention the specific times." ! queue ! overlay.text_sink
```

### Stable Diffusion

`python mp-launch.py filesrc location=data/prompt_for_stable_diffusion.txt ! mp_stablediffusion device=cuda ! pngenc ! filesink location=output_image.png`

### Caption

`mp_caption_qwen` captions frames with Qwen2.5-VL, `mp_caption_phi` with
Phi-3.5-vision. Here `coalescehistory` hands the last ten captions to an LLM for a
running summary:

```
python mp-launch.py filesrc location=data/soccer_single_camera.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480 ! tee name=t t. ! queue ! textoverlay name=overlay wait-text=false ! videoconvert ! autovideosink t. ! queue leaky=2 max-size-buffers=1 ! videoconvertscale ! video/x-raw,width=240,height=180 ! mp_caption_qwen device=cuda:0 prompt="In one sentence, describe what you see?" model-name="Qwen/Qwen2.5-VL-3B-Instruct-AWQ" name=cap cap.src ! fakesink async=0 sync=0 cap.text_src ! queue ! coalescehistory history-length=10 ! mp_llm model-name="Qwen/Qwen3-0.6B" device=cuda system-prompt="You receive the history of what happened in recent times, summarize it nicely with excitement but NEVER mention the specific times. Focus on the most recent events." ! queue ! overlay.text_sink
```

### Kafka Sink

`mp_kafkasink` needs a broker. Create a docker network and add `--network kafka-network`
to the `docker run` line in [INSTALL.md](INSTALL.md#run-docker-container) so a containerised pipeline reaches it:

```
docker network create kafka-network
```

#### Set up kafka

From the host the broker is `localhost:29092` instead of `kafka:9092`. The broker
runs in KRaft mode, the current `cp-kafka` image has no ZooKeeper support.

```
docker stop kafka
docker rm kafka
docker run -d --name kafka --network kafka-network \
  -e KAFKA_NODE_ID=1 \
  -e KAFKA_PROCESS_ROLES=broker,controller \
  -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@kafka:9093 \
  -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER \
  -e KAFKA_LISTENERS=INSIDE://0.0.0.0:9092,OUTSIDE://0.0.0.0:29092,CONTROLLER://0.0.0.0:9093 \
  -e KAFKA_ADVERTISED_LISTENERS=INSIDE://kafka:9092,OUTSIDE://localhost:29092 \
  -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=INSIDE:PLAINTEXT,OUTSIDE:PLAINTEXT,CONTROLLER:PLAINTEXT \
  -e KAFKA_INTER_BROKER_LISTENER_NAME=INSIDE \
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  -e CLUSTER_ID=MkU3OEVBNTcwNTJENDM2Qk \
  -p 9092:9092 \
  -p 29092:29092 \
  confluentinc/cp-kafka:latest
```

#### Topics

```
docker exec kafka kafka-topics --create --topic test-kafkasink-topic --bootstrap-server kafka:9092 --partitions 1 --replication-factor 1
docker exec -it kafka kafka-topics --list --bootstrap-server kafka:9092
docker exec -it kafka kafka-topics --delete --topic test-topic --bootstrap-server kafka:9092
docker exec -it kafka kafka-console-consumer --bootstrap-server kafka:9092 --topic test-kafkasink-topic --from-beginning
```

### Overlay from a metadata file

`mp_metareplay` attaches the detections a `mp_metasink` recorded, so `mp_overlay` and `mp_overlay_counter` draw them with no model in the pipeline.

`python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_metareplay location=data/people.jsonl ! mp_tracker tracker-type=sort ! mp_overlay tracking=true ! videoconvert ! autovideosink`

`python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_metareplay location=data/people.jsonl ! mp_tracker tracker-type=sort ! mp_overlay_counter tracking=true ! videoconvert ! autovideosink`


### Stream Mux and Demux

```
 python mp-launch.py   videotestsrc pattern=ball ! video/x-raw, width=320, height=240 ! queue ! mp_streammux name=mux   videotestsrc pattern=smpte ! video/x-raw, width=320, height=240 ! queue ! mux.sink_1   videotestsrc pattern=smpte ! video/x-raw, width=320, height=240 ! queue ! mux.sink_2   mux.src ! queue ! mp_streamdemux name=demux   demux.src_0 ! queue ! glimagesink  demux.src_1 ! queue ! glimagesink   demux.src_2 ! queue  ! glimagesink
```

### Segment Anything (SAM)

`mp_sam` runs SAM2 with point, box or automatic prompts.

#### Auto-mask segmentation (segment everything)

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_sam model-name=facebook/sam2-hiera-small device=cuda mode=auto \
  ! videoconvert ! autovideosink sync=false
```

#### Point-prompt segmentation (segment object at center)

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_sam model-name=facebook/sam2-hiera-small device=cuda \
            mode=points max-masks=10 \
  ! videoconvert ! autovideosink sync=false
```

### OCR

`mp_ocr` recognizes text with TrOCR and appends it as a `GST-OCR:` chunk.

#### TrOCR recognition

```
python mp-launch.py filesrc location=data/document.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_ocr model-name=microsoft/trocr-base-printed device=cuda \
  ! videoconvert ! mp_overlay ! videoconvert ! autovideosink sync=false
```

### Face Detection & Recognition

`mp_face` detects faces with RetinaFace and names them from ArcFace embeddings of a gallery.

#### Face detection only

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_face model-name=buffalo_l device=cuda \
  ! videoconvert ! mp_overlay ! videoconvert ! autovideosink sync=false
```

#### Face detection + recognition with gallery

`gallery-path` holds one image per person, named after them. Without it faces are detected but not named.

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_face model-name=buffalo_l device=cuda gallery-path=data/face_gallery/ threshold=0.6 \
  ! videoconvert ! mp_overlay ! videoconvert ! autovideosink sync=false
```

### Anonymization

`mp_anonymize` applies blur, pixelation or blackout to analytics detection
boxes. `labels` selects comma-separated class names and defaults to every
detection. `box-expansion` adds a fraction of the box size on every side and
`edge-fade` blends the effect over that many pixels at the expanded edge.

#### Blur detected faces

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvertscale ! "video/x-raw,format=RGBA,width=640,height=480" \
  ! mp_face model-name=buffalo_l device=cuda \
  ! mp_anonymize mode=blur box-expansion=0.1 edge-fade=8 \
  ! videoconvert ! autovideosink sync=false
```

The element also draws directly on `CUDAMemory`. A CUDA detector such as
`mp_yolo` can feed it without downloading the frame.

### Optical Flow

`mp_optical_flow` runs RAFT between consecutive frames.

#### RAFT optical flow with color visualization

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_optical_flow model-name=raft_small device=cuda visualize=true \
  ! videoconvert ! autovideosink sync=false
```

### Super-Resolution

`mp_superres` upscales with Real-ESRGAN.

#### 2x upscale

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=320,height=240" \
  ! mp_superres model-name=real-esrgan-x2 device=cuda \
  ! videoconvert ! autovideosink sync=false
```

#### 4x upscale

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=320,height=240" \
  ! mp_superres model-name=real-esrgan-x4 device=cuda \
  ! videoconvert ! autovideosink sync=false
```

### Action Recognition

`mp_action` classifies the action in a sliding window of frames.

#### VideoMAE action recognition

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_action model-name=MCG-NJU/videomae-base-finetuned-kinetics device=cuda num-frames=16 \
  ! videoconvert ! mp_overlay ! videoconvert ! autovideosink sync=false
```

### Anomaly Detection

`mp_anomaly` scores frames against PatchCore features of normal ones in `reference-path`.

#### PatchCore anomaly detection

```
python mp-launch.py filesrc location=data/factory.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_anomaly device=cuda reference-path=data/factory_reference.npy threshold=0.5 \
  ! videoconvert ! autovideosink sync=false
```

### Audio Classification (CLAP)

`mp_clap` classifies audio against the labels you pass with LAION CLAP.

#### CLAP audio event detection

```
python mp-launch.py filesrc location=data/audio_sample.wav ! decodebin \
  ! audioconvert ! audioresample ! audio/x-raw,format=F32LE,rate=48000,channels=1 \
  ! mp_clap device=cuda labels="gunshot,siren,baby crying,music,speech" threshold=0.3 \
  ! fakesink
```

### Vision-Language Model (VLM)

`mp_vlm` answers `prompt` about each frame with a vision-language model such as LLaVA or SmolVLM.

#### SmolVLM visual question answering

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_vlm model-name=HuggingFaceTB/SmolVLM-500M-Instruct device=cuda \
            prompt="What is happening in this scene?" \
  ! fakesink
```

`model-name=llava-hf/llava-1.5-7b-hf` also works well. It needs about 14 GB of GPU memory in fp16.

### Cascade Gating

Every video element takes `only-on`, the name of a blob a frame has to carry
before the element runs on it. `only-on=alert` runs on the frames `mp_alert`
flagged, `only-on=detections` on the frames that carry at least one analytics
object, and any other value names the blob an element attached under its own
name (`depth`, `vlm`, ...). Frames without it pass through untouched, so a cheap
detector decides which frames an expensive model sees.

#### VLM only on alerted frames

The alert decides which frames the vision-language model sees. The same property
works on `mp_caption_qwen`, `mp_depth` or any other video element.

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_yolo model-name=yolo11m device=cuda ! mp_alert rules='{"class":"person","min_score":0.8}' cooldown=5 draw-alert=false ! mp_vlm model-name=HuggingFaceTB/SmolVLM-500M-Instruct device=cuda only-on=alert max-tokens=40 prompt="What is the person in the centre doing?" ! mp_metasink location=people.jsonl
```

The g2g host hands a hosted element the frame's upstream detections and
blobs, so `only-on` gates the same way on both backends.

### Embedding Extractor

`mp_embedding` attaches a CLIP or DINOv2 embedding to each frame.

#### CLIP embedding extraction

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_embedding model-name=openai/clip-vit-base-patch32 device=cuda \
            normalize=true \
  ! fakesink
```

#### DINOv2 embeddings saved to file

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_embedding model-name=facebook/dinov2-base device=cuda \
            frame-stride=5 \
  ! fakesink
```

### Video Memory

`mp_embeddingsink` stores the vectors `mp_embedding` produces in an sqlite
index, one row per embedded frame: its `pts` in seconds, the `source-id` you give
the stream, and the embedding. Search it by text through the MCP server's
`search_video`. One index holds one model, and the embedding blob does not carry
the model name, so set `model-name` on the sink to the extractor's model.

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_embedding model-name=openai/clip-vit-base-patch32 device=cuda frame-stride=30 ! mp_embeddingsink location=people.sqlite source-id=people model-name=openai/clip-vit-base-patch32
```

### Multi-Object Tracker

`mp_tracker` tracks the detections of any upstream detector. A detection
matches a track by box overlap (`iou-threshold`) or by centre distance within
`distance-gate` box sizes, so small boxes that move their own width a frame
keep their id. Before matching, a detection covered by more than
`merge-overlap` of its area by a higher-scoring one is dropped, and a detection
overlapping a track that matched this frame by more than
`new-track-max-overlap` starts no track, since both are second boxes on one
object that NMS kept. `camera-motion` re-matches leftovers after a pan.

Use it instead of `track=true` on the detector, not as well as it. `track=true`
runs ultralytics' BOTSORT inside the detector, about 8 ms a frame on the CPU
with its optical-flow motion compensation, while `mp_tracker` costs about
2 ms, so a pipeline with both tracks twice and pays for the slow one.

#### YOLO + standalone SORT tracker

```
python mp-launch.py filesrc location=data/soccer_tracking.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda \
  ! mp_tracker tracker-type=sort max-age=30 min-hits=3 iou-threshold=0.3 \
  ! mp_overlay ! videoconvert ! autovideosink sync=false
```

### ML Alert

`mp_alert` fires on upstream detections that match `rules`.

#### Webhook alert on person detection

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda \
  ! mp_alert rules='{"class":"person","min_score":0.8}' \
              webhook-url=http://localhost:8080/alert cooldown=10 \
  ! mp_overlay ! videoconvert ! autovideosink sync=false
```

To authenticate the POST, set `webhook-token-environment-variable` to the name of
an environment variable holding the token. The element sends its value as
`Authorization: Bearer`, and fails at the first alert if the variable is unset.
The token never appears in the pipeline string this way. Native `analyticsalert`
has no counterpart, so the launcher refuses the property under g2g.

#### MQTT alert with zone filtering

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin name=d \
  d. ! queue ! videoconvert ! videoscale ! "video/x-raw,width=640,height=480" \
  ! mp_yolo model-name=yolo11m device=cuda \
  ! mp_alert rules='{"class":"person","min_score":0.7,"zone":[0,0,320,240]}' \
              mqtt-broker=localhost:1883 mqtt-topic=alerts/zone1 cooldown=5 \
  ! mp_overlay ! videoconvert ! autovideosink sync=false
```

### Alert Recorder

`mp_alertrecorder` writes a clip around each alert that an upstream `mp_alert`
attached to the buffer. It keeps `seconds-before` of video in memory and records
until `seconds-after` have passed since the last alert. `%s` in `location` is the
time the alert fired. The default encoder is `vp8enc deadline=1 ! webmmux`. Set
`encoder` and the extension together, for example
`encoder="x264enc tune=zerolatency ! mp4mux" location=alert-%s.mp4`.

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_yolo model-name=yolo11m device=cuda ! mp_alert rules='{"class":"person","min_score":0.7}' cooldown=8 draw-alert=false ! mp_alertrecorder location=alert-%s.webm seconds-before=2 seconds-after=3 ! mp_overlay ! videoconvert ! autovideosink sync=false
```

### Metadata Sink

`mp_metasink` writes one JSON line per buffer with whatever the pipeline knows
about it: `pts` in seconds, `detections` from the analytics metadata, any JSON
blob an element attached under its name (`alert`, `depth`, `vlm`, ...) and, for a
text stream, `text`. Lines go to `location`, or to stdout when it is unset. Use a
`tee` to keep a display alongside it.

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_yolo model-name=yolo11m device=cuda ! mp_alert rules='{"class":"person","min_score":0.7}' draw-alert=false ! mp_metasink location=people.jsonl
```

```
{"pts": 0.08, "detections": [{"label": "stream_0_person", "x": 469, "y": 312, "w": 37, "h": 82, "score": 0.85}], "alert": [{"timestamp": 1789611763.6, "rule": {"class": "person", "min_score": 0.7}, "detection": {"label": "stream_0_person", "x": 469, "y": 312, "w": 37, "h": 82, "score": 0.85}}]}
```

### Metadata Replay

`mp_metareplay` reads a JSON lines file a `mp_metasink` wrote and reattaches
each record to the frame at the same timestamp: `detections` become analytics
metadata again, every other key becomes the blob it came from. A record matches a
frame whose timestamp is within half a frame duration of it. Downstream
`mp_tracker`, `mp_alert`, `mp_overlay` and `mp_alertrecorder` then run
with no model loaded. Record once, then rerun the rules as often as you like.

Record once:

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_yolo model-name=yolo11m device=cuda confidence=0.5 ! mp_metasink location=people.jsonl
```

Then replay. `data/people.jsonl` is that recording, shipped so this line runs with
no GPU at all:

```
python mp-launch.py filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! mp_metareplay location=data/people.jsonl ! mp_tracker tracker-type=sort ! mp_overlay ! videoconvert ! autovideosink sync=false
```

## MCP Server

`mp-mcp` is an MCP server over stdio that runs the gst backend in-process, so an
agent can start a pipeline, read its results and change a property while it runs.
End every pipeline with a `mp_metasink`, that is where records come from.

- `start_pipeline`, `pipeline_status`, `stop_pipeline`, `set_property`, `get_property`, `list_elements`, `inspect`.
- One pipeline runs at a time, `start_pipeline` stops the previous one. `loop=true` replays a file source when it ends.
- `set_property` and `get_property` find an element by its name, so give the ones an agent should tune a `name=`, such as `mp_yolo name=detector`.
- When the pipeline has an element named `record`, such as `filesink name=record`, `stop_pipeline` sends end of stream first so the mp4 it writes is playable.
- `latest_metadata` returns the newest records, at most the last 1000. `wait_for_records` blocks until the sink posts new ones, optionally only those carrying a key such as `detections`.
- `load_metadata` reads a JSON lines file a `mp_metasink` wrote, so a finished run needs no pipeline.
- `engine_options` lists the engines and devices that run a model name or file, as `mp-engines --json` prints them.
- `latest_bus_messages` returns the newest element messages, warnings and QoS reports any element posted, such as `level` loudness or a sink dropping late frames, at most the last 1000. Debug logs never reach the bus. Errors in `pipeline_status` and warnings carry GStreamer's debug detail, such as `not-negotiated`.
- `buffer_flow` counts the buffers each source pad pushed, with the rate while they flowed and the seconds since the last one. Data stops at the first pad whose count stopped growing. `latency_ms` is the element's recent average time from a buffer entering to it leaving, matched by timestamp, so an element that changes timestamps, such as an encoder, has none, and a `queue` shows how long buffers wait in it. It covers the pipeline's own elements, not the ones inside a bin such as `autovideosink`.
- `snapshot_frame` returns the newest rendered frame as a JPEG. `describe_frame` captions it with the model `MP_MCP_VLM_MODEL` names, `HuggingFaceTB/SmolVLM-500M-Instruct` by default.
- `search_video` reads an index `mp_embeddingsink` wrote and returns the frames closest to a description, embedded with the index's own model. That model has to embed text too, so build the index with CLIP, not DINOv2. `clip_at` cuts a webm around a pts, which turns a hit into a clip.
- Every pipeline section of this file is a prompt named after it, such as `object_detection`.

Models run on `MP_MCP_DEVICE`, `cuda` when torch sees one, else `cpu`, where a
caption takes tens of seconds.

```
uv sync --extra mcp
claude mcp add modelpipe -- /path/to/modelpipe/.venv/bin/mp-mcp
```

With `MP_BACKEND=g2g`, `mp-mcp` serves glass2glass's own MCP server from
`g2g-launch-py --mcp`, on `PATH` or at `G2G_LAUNCH`. It takes pipelines in
the `gst-launch` spelling this file uses and rewrites them as `mp-launch` does.
An element passed to `insert_transform` is rewritten on its own, so name its
`format=` or `cuda-frames=true` yourself. The tools and prompts are g2g's. It has
`latest_metadata`, `wait_for_records`, `load_metadata`, `snapshot_frame`,
`clip_at`, `set_property` and `get_property` under these names, but no
`describe_frame`, `search_video` or prompts from this file.

```
claude mcp add modelpipe-g2g -e MP_BACKEND=g2g -- /path/to/modelpipe/.venv/bin/mp-mcp
```
