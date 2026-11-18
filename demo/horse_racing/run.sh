#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
source .venv/bin/activate

export MP_BACKEND="${HOST:-g2g}"
if [[ -z "${MODELS:-}" ]]; then
  python demo/horse_racing/fetch_models.py >&2
  MODELS="$REPO/models/horse_racing"
fi
DETECTOR="${DETECTOR:-$HOME/.roboflow/models/rf-detr-medium.pth}"
CONF="${CONF:-0.5}"
WINDOW="${WINDOW:-120}"
IN="${1:?usage: demo/horse_racing/run.sh INPUT.mp4 [OUTPUT.mp4|OUTPUT.jsonl]}"
OUT="${2:-}"
[[ -f "$IN" ]] || { echo "input not found: $IN" >&2; exit 1; }

START="mp_race_start state-model=$MODELS/race_state.pth window-model=$MODELS/race_start_window.pth window-closes-seconds=$WINDOW"
DETECT="mp_rfdetr model-name=$DETECTOR classes=horse confidence=$CONF device=cuda:0 only-on=race-window"
NUMBERS="mp_racer_number_classifier model=$MODELS/numbers.pth"
TRACK="mp_racer_number_tracker"
# a queue at each stage boundary lets decoding, inference and output overlap
Q="queue max-size-buffers=8 max-size-time=0 max-size-bytes=0 !"
if [[ "$OUT" == *.jsonl ]]; then
  # mp_metasink appends to its file
  rm -f "$OUT"
  SINK="mp_metasink location=$OUT"
elif [[ -n "$OUT" ]]; then
  SINK="mp_racer_overlay number-cloths=horse ! videoconvert ! x264enc ! mp4mux ! filesink location=$OUT"
else
  SINK="mp_racer_overlay number-cloths=horse ! videoconvert ! autovideosink"
fi

echo "[$MP_BACKEND] '$IN' -> ${OUT:-screen}"
mp-launch \
  filesrc location="$IN" ! qtdemux ! h264parse ! nvh264dec ! videoconvert \
  ! video/x-raw,format=RGBA ! $Q $START ! $Q $DETECT ! $Q $NUMBERS ! $TRACK ! $Q $SINK
