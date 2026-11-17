#!/usr/bin/env bash
set -euo pipefail

USAGE="usage: demo/horse_racing/fetch_races.sh DATA_DIR"
DEMO="$(cd "$(dirname "$0")" && pwd)"
DATA="$(realpath "${1:?$USAGE}")"
mkdir -p "$DATA/broadcasts" "$DATA/races" "$DATA/clips"
API="https://commons.wikimedia.org/w/api.php"
for NUMBER in 01 02 03 04 05 06 07 08 09; do
  NAME="reunion_$NUMBER"
  [[ -f "$DATA/races/$NAME.csv" ]] && continue
  URL=$(curl -sS -A "modelpipe-demo/1.0" -G "$API" \
    --data-urlencode "action=query" --data-urlencode "format=json" \
    --data-urlencode "prop=imageinfo" --data-urlencode "iiprop=url" \
    --data-urlencode "generator=categorymembers" --data-urlencode "gcmtype=file" --data-urlencode "gcmlimit=100" \
    --data-urlencode "gcmtitle=Category:Hipódromo de La Rinconada" |
    python3 -c "import json, sys; print(next(p['imageinfo'][0]['url'] for p in json.load(sys.stdin)['query']['pages'].values() if p['title'].startswith('File:REUNIÓN $NUMBER ')))")
  BROADCAST="$DATA/broadcasts/$NAME.webm"
  curl -sSL -A "modelpipe-demo/1.0" -C - -o "$BROADCAST" "$URL"
  python "$DEMO/find_races.py" "$BROADCAST" "$DATA/races/$NAME.csv.partial"
  python "$DEMO/cut_races.py" "$BROADCAST" "$DATA/races/$NAME.csv.partial" "$DATA/clips"
  python "$DEMO/sample_other_frames.py" "$BROADCAST" "$DATA/other_frames"
  mv "$DATA/races/$NAME.csv.partial" "$DATA/races/$NAME.csv"
  rm "$BROADCAST"
done
