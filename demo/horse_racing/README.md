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

## Run

```bash
demo/horse_racing/run.sh INPUT.mp4 OUTPUT.mp4
```

`run.sh` downloads the three models from
[boxerab/modelpipe-horse-racing](https://huggingface.co/boxerab/modelpipe-horse-racing)
into `models/horse_racing`. Set `MODELS` to a directory with your own
`race_state.pth`, `race_start_window.pth` and `numbers.pth` to skip the
download. The input has to be H.264 in MP4, so convert the Commons `.webm`
files with ffmpeg first. An output ending in `.jsonl` gets the metadata
instead of the video.

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

The doors open anywhere from 0.03 s to 2.5 s before the timing graphic, and on
some races the graphic comes up seconds late, so each race's door frame is
labelled by hand. One Claude subagent per broadcast follows
`door_labels_spec.md` and writes `DATA/labels/doors/<broadcast>.csv`.
`fetch_races.sh` cuts some races twice, the second copy after a "Válida"
bumper card. Leave the duplicate's `door_frame` empty.

The `other` class of the state model comes from the frames away from races,
which also hold warm-ups, replays and gate shots:

```bash
python demo/horse_racing/frame_sheets.py DATA/other_frames 1200 DATA/other_sheets
```

Four Claude subagents of ten sheets each follow `other_labels_spec.md` and
write `DATA/other_labels/batch_<n>.csv`. A fifth labels every tenth sheet into
`DATA/other_labels/second.csv`. `horses_running` frames are left out of
training.

```bash
PYTHONPATH=plugins/python python demo/horse_racing/merge_sheet_labels.py other DATA/other_sheets DATA/other_labels
PYTHONPATH=plugins/python python demo/horse_racing/extract_start_frames.py \
  DATA/clips DATA/labels/doors DATA/other_frames DATA/other_labels/labels.csv DATA/start_frames
PYTHONPATH=plugins/python python demo/horse_racing/train_start_models.py state \
  DATA/start_frames race_state.pth --test-broadcasts reunion_08
PYTHONPATH=plugins/python python demo/horse_racing/train_start_models.py window \
  DATA/start_frames race_start_window.pth --test-broadcasts reunion_08
```

REUNIÓN 08 holds the Segula C race (`reunion_08_race_3`), so it is the test
broadcast. The window model runs out of memory on a 6 GB GPU at 512 px, both
were trained on an RTX 3090. On REUNIÓN 08 the state model put 2463 of 2478
frames in the right class and the window model 993 of 1020, all 27 misses
being frames after the start called before it. The scene labellers agreed on
115 of 120 frames.

## Number cloth model

`extract_horse_crops.py` keeps the horse boxes at least 30 px tall on every
tenth frame of a race. The labelled crops come in three sets: REUNIÓN 01 at 200
crops per race, cut when the minimum was 60 px, REUNIÓN 02 to 07 and 09 at 60
per race for training, and REUNIÓN 08 at 60 per race for testing.
`graphic_starts_more.csv` and `graphic_starts_08.csv` hold the rows of
`graphic_starts.csv` for those broadcasts, without the duplicate races left
blank in `labels/doors`.

```bash
python demo/horse_racing/extract_horse_crops.py DATA/clips DATA/labels/graphic_starts_more.csv DATA/crops_more
python demo/horse_racing/number_sheets.py DATA/crops_more 60 DATA/number_sheets_more
```

REUNIÓN 08 goes the same way into `DATA/crops_08` and `DATA/number_sheets_08`.
Claude subagents of eight to ten sheets each follow `number_labels_spec.md` and
write `batch_<n>.csv` into `DATA/number_labels_more` or `DATA/number_labels_08`.
One more labels every tenth sheet of both sets into `second.csv`.

```bash
PYTHONPATH=plugins/python python demo/horse_racing/merge_sheet_labels.py numbers DATA/number_sheets_more DATA/number_labels_more
PYTHONPATH=plugins/python python demo/horse_racing/merge_sheet_labels.py numbers DATA/number_sheets_08 DATA/number_labels_08
```

Copy the labelled crops of the three sets into `DATA/crops_labelled` with their
rows of each `crops.csv`, join the three `labels.csv` files and train with
REUNIÓN 08 held out:

```bash
PYTHONPATH=plugins/python python demo/horse_racing/train_numbers.py \
  DATA/crops_labelled DATA/labels.csv numbers.pth --test-races reunion_08_race_{1..9} reunion_08_race_11
```

The two labellers agreed on 157 of 180 crops of REUNIÓN 01, 372 of 420 of the
other training broadcasts and 56 of 60 of REUNIÓN 08. Nearly every
disagreement was a number against `none`. On the 600 REUNIÓN 08 crops the
classifier trained on all three sets was right on 468, against 437 for one
trained on REUNIÓN 01 alone. It read 303 of the 365 numbered crops, against
248, and gave a crop the wrong number 13 times, against 43.
