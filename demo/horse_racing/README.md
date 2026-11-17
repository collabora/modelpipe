# Horse racing demo

Race start, horse detection and number cloth tracking on TV broadcasts from
the La Rinconada racecourse in Caracas:
`mp_race_start` -> `mp_rfdetr classes=horse` -> `mp_racer_number_classifier` ->
`mp_racer_number_tracker`.

The detector is the stock COCO RF-DETR Medium. The race start and number cloth
models are trained here, on the nine "REUNIÓN 0N - HIPÓDROMO LA RINCONADA"
broadcasts in the Wikimedia Commons category
[Hipódromo de La Rinconada](https://commons.wikimedia.org/wiki/Category:Hip%C3%B3dromo_de_La_Rinconada),
marked PD-VenezuelaGov. The test race, "2026 Segula C Classic.webm", is in the
same category and stays out of training.

`DATA` below is one directory outside the repo. Run everything from the repo
root with the venv active.

## Races

```bash
demo/horse_racing/fetch_races.sh DATA
python demo/horse_racing/graphic_starts.py DATA/clips DATA/labels/graphic_starts.csv
```

`fetch_races.sh` takes one broadcast at a time (4 to 5 GB each), cuts its live
races to `DATA/clips`, samples frames away from the races to
`DATA/other_frames` and deletes the broadcast. It skips a broadcast that
already has `DATA/races/<broadcast>.csv`.

- Races are found by the INH timing graphic, `inh_logo.png`. Replays carry the
  same graphic and are dropped by thumbnail distance to the live race: replays
  measured at most 12.7 grey levels on REUNIÓN 01, other races at least 20.8.
- REUNIÓN 01 is AV1 at 59.94 fps, REUNIÓN 02 is VP9 at 30 fps. ffmpeg's VP9
  decoder ignores `-skip_frame nokey`, so a VP9 broadcast is decoded in full,
  about 20 minutes per pass.

## Start models

The doors open between 0.27 s and 1.40 s before the timing graphic and the gap
changes from race to race, so each race's door frame is labelled by hand. One
Claude subagent per broadcast follows `door_labels_spec.md` and writes
`DATA/labels/doors/<broadcast>.csv`.

```bash
PYTHONPATH=plugins/python python demo/horse_racing/extract_start_frames.py DATA/clips DATA/labels/doors DATA/start_frames
PYTHONPATH=plugins/python python demo/horse_racing/train_start_models.py window \
  DATA/start_frames race_start_window.pth --test-races reunion_01_race_8 reunion_02_race_10
```

The window model runs out of memory on a 6 GB GPU at 512 px. On an RTX 3090 it
put 147 of 154 held out frames in the right class. All 7 misses were frames
after the start called before it.

TODO: add `DATA/other_frames` as the `other` class of the state model.

## Number cloth model

```bash
python demo/horse_racing/extract_horse_crops.py DATA/clips DATA/labels/graphic_starts.csv DATA/crops
python demo/horse_racing/number_sheets.py DATA/crops DATA/number_sheets
```

Six Claude subagents of nine sheets each follow `number_labels_spec.md` and
write `DATA/number_labels/batch_<n>.csv`. A seventh labels every tenth sheet
into `DATA/number_labels/second.csv`.

```bash
PYTHONPATH=plugins/python python demo/horse_racing/merge_number_labels.py DATA/number_sheets DATA/number_labels
PYTHONPATH=plugins/python python demo/horse_racing/train_numbers.py \
  DATA/crops DATA/number_labels/labels.csv numbers.pth --test-races reunion_01_race_8
```

On the REUNIÓN 01 crops the two labellers agreed on 157 of 180 crops, and every
disagreement was a number against `none`. The classifier put 137 of 200 crops
of the held out race in the labellers' class, most misses going to `none`.
