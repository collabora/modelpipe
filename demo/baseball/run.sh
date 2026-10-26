#!/usr/bin/env bash
# Baseball overlay demo.
#
# Pipeline:
#   mp_yolo (stock coco weights)  ->  mp_tracker (ByteTrack)  ->  mp_yolo_pose  ->  mp_baseball_overlay
#
# Usage:
#   demo/baseball/run.sh [INPUT.mp4] [OUTPUT.mp4] [WxH]      # file -> annotated mp4
#   demo/baseball/run.sh display [INPUT.mp4] [WxH]           # file -> live on-screen
#   demo/baseball/run.sh camera [/dev/videoN] [WxH]          # live camera -> on-screen
#   HOST=gst demo/baseball/run.sh ...                         # same, on gst-launch-1.0
#   CUDA=true demo/baseball/run.sh ...                        # decode, detect, track, draw and sink on cuda memory
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
source .venv/bin/activate
export GST_PLUGIN_PATH="$REPO/plugins:${GST_PLUGIN_PATH:-}"
# gst-launch's python loader runs the system interpreter, so hand it the venv.
VENV_SITE="$(python -c 'import site; print(":".join(site.getsitepackages()))')"
export PYTHONPATH="$REPO/plugins/python:$VENV_SITE:${PYTHONPATH:-}"

HOST="${HOST:-g2g}"        # g2g = g2g-launch-py hosting the python elements, gst = gst-launch-1.0
CUDA="${CUDA:-false}"      # decode, detect, track and draw on cuda memory
MODEL="${MODEL:-yolo11m}"  # stock ultralytics weights, downloaded into models/baseball on first use
INTERVAL="${INTERVAL:-1}"  # run detection every Nth frame, the ball trail needs every frame
CONF="${CONF:-0.25}"       # detector confidence threshold
IMGSZ="${IMGSZ:-1280}"     # network input size, 640 misses the ball on most frames
NEWTRACK="${NEWTRACK:-0.4}" # min confidence to start a new track
BALL_TRAIL="${BALL_TRAIL:-true}"
LABELS="${LABELS:-true}"   # write batter / fielder / bat / glove next to each marker
EQUIPMENT="${EQUIPMENT:-true}" # outline the bat and the glove
POSE="${POSE:-true}"       # draw the batter's skeleton from $MODEL-pose
CLIP="data/baseball_batter.mp4"

if [[ "$HOST" == "g2g" ]]; then
  DETECTOR_ELEMENT="pyelement module=yolo class=YOLOTransform"
  TRACKER_ELEMENT="pyelement module=tracker class=TrackerTransform"
  POSE_ELEMENT="pyelement module=pose class=YOLOPoseTransform"
  OVERLAY_ELEMENT="pyelement module=baseball_overlay class=BaseballOverlay"
  CONVERT="videoconvertscale"
  DECODE="qtdemux ! ffmpegdec ! $CONVERT"
  DISPLAY_SINK="videoconvertscale ! waylandsink"
  CAMERA_SINK="videoconvertscale ! waylandsink"
  ENCODE="videoconvertscale ! x264enc ! mp4mux"
  LAUNCH="g2g-launch-py --threads"
  if [[ "$CUDA" == "true" ]]; then
    CUDA_FRAMES="cuda-frames=true format=NV12"
    DECODE="qtdemux ! h264parse ! nvdec"
    DETECTOR_ELEMENT="$DETECTOR_ELEMENT $CUDA_FRAMES"
    TRACKER_ELEMENT="$TRACKER_ELEMENT $CUDA_FRAMES"
    POSE_ELEMENT="$POSE_ELEMENT $CUDA_FRAMES"
    OVERLAY_ELEMENT="$OVERLAY_ELEMENT $CUDA_FRAMES"
    DISPLAY_SINK="cudaglsink"
    ENCODE="nvenc ! mp4mux"
  fi
else
  DETECTOR_ELEMENT="mp_yolo name=detector"
  TRACKER_ELEMENT="mp_tracker"
  POSE_ELEMENT="mp_yolo_pose name=pose"
  OVERLAY_ELEMENT="mp_baseball_overlay name=overlay"
  CONVERT="videoconvert ! videoscale"
  DOWNLOAD=""
  if [[ "$CUDA" == "true" ]]; then
    CONVERT="cudaupload ! cudaconvertscale"
    DOWNLOAD="cudadownload ! "
  fi
  DECODE="decodebin ! $CONVERT"
  LAUNCH="gst-launch-1.0 -e"
fi
# nvdec decodes at the clip's own size and format, nothing to ask for
input_caps() {
  [[ "$HOST" == "g2g" && "$CUDA" == "true" ]] && return
  local caps="video/x-raw"
  [[ "$CUDA" == "true" ]] && caps="video/x-raw(memory:CUDAMemory)"
  echo "! $caps,width=$1,height=$2,format=RGBA"
}
DETECT="$DETECTOR_ELEMENT model-name=models/baseball/$MODEL device=cuda:0 interval=$INTERVAL imgsz=$IMGSZ confidence=$CONF"
TRACK="$TRACKER_ELEMENT tracker-type=bytetrack new-track-confidence=$NEWTRACK"
POSE_ESTIMATE="$POSE_ELEMENT model-name=models/baseball/$MODEL-pose device=cuda:0 visualize=false"
OVERLAY="$OVERLAY_ELEMENT ball-trail=$BALL_TRAIL show-labels=$LABELS show-equipment=$EQUIPMENT"

if [[ "$HOST" == "g2g" ]]; then
  # g2g has no queue element and needs none
  CHAIN="$DETECT ! $TRACK ! $OVERLAY"
  [[ "$POSE" == "true" ]] && CHAIN="$DETECT ! $TRACK ! $POSE_ESTIMATE ! $OVERLAY"
else
  # a queue at each stage boundary lets inference, tracking and the sink overlap
  Q="queue max-size-buffers=8 max-size-time=0 max-size-bytes=0"
  PREROLL="queue max-size-buffers=100 max-size-time=0 max-size-bytes=0"
  CHAIN="$Q ! $DETECT ! $Q ! $TRACK ! $Q ! $OVERLAY"
  [[ "$POSE" == "true" ]] && CHAIN="$Q ! $DETECT ! $Q ! $TRACK ! $Q ! $POSE_ESTIMATE ! $Q ! $OVERLAY"
  DISPLAY_SINK="$PREROLL ! ${DOWNLOAD}videoconvert ! autovideosink sync=true"
  CAMERA_SINK="$Q ! ${DOWNLOAD}videoconvert ! autovideosink sync=false"
  ENCODE="$Q ! ${DOWNLOAD}videoconvert ! openh264enc bitrate=8000000 ! h264parse ! mp4mux"
fi

MODE="${1:-file}"
if [[ "$MODE" == "camera" ]]; then
  [[ "$HOST" == "g2g" && "$CUDA" == "true" ]] && { echo "CUDA=true on g2g decodes a file with nvdec, camera mode runs without it" >&2; exit 1; }
  DEV="${2:-/dev/video0}"; SIZE="${3:-1280x720}"
  W="${SIZE%x*}"; H="${SIZE#*x}"
  echo "[$HOST] live camera $DEV @ ${W}x${H} -> display"
  exec $LAUNCH \
    v4l2src device="$DEV" ! $CONVERT \
    $(input_caps "$W" "$H") \
    ! $CHAIN \
    ! $CAMERA_SINK
elif [[ "$MODE" == "display" ]]; then
  IN="${2:-$CLIP}"
  SIZE="${3:-1280x720}"
  W="${SIZE%x*}"; H="${SIZE#*x}"
  [[ -f "$IN" ]] || { echo "input not found: $IN" >&2; exit 1; }
  echo "[$HOST] '$IN' @ ${W}x${H} -> live display (real-time)"
  exec $LAUNCH \
    filesrc location="$IN" ! $DECODE \
    $(input_caps "$W" "$H") \
    ! $CHAIN \
    ! $DISPLAY_SINK
else
  IN="${1:-$CLIP}"
  OUT="${2:-demo/baseball/out.mp4}"
  SIZE="${3:-1280x720}"
  W="${SIZE%x*}"; H="${SIZE#*x}"
  [[ -f "$IN" ]] || { echo "input not found: $IN" >&2; exit 1; }
  echo "[$HOST] '$IN' @ ${W}x${H} -> '$OUT'"
  $LAUNCH \
    filesrc location="$IN" ! $DECODE \
    $(input_caps "$W" "$H") \
    ! $CHAIN \
    ! $ENCODE ! filesink location="$OUT"
  echo "Done: $OUT"
fi
