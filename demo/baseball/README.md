# Baseball demo

Detection, tracking and a baseball overlay:
`mp_yolo` -> `mp_tracker` -> `mp_yolo_pose` -> `mp_baseball_overlay`.

The detector is the stock `yolo11m` COCO model, which already knows `person`,
`baseball bat`, `baseball glove` and `sports ball`. No training is needed. The
overlay draws a foot ellipse per person coloured by role, an outline around the
bat and the glove, a ring on the ball and a trail of where the ball has been.
A person whose box overlaps a bat is the batter (red), one overlapping a glove
is a fielder (blue), anyone else is white. The role is voted per track id, so
it stays put once the bat or glove is hidden for a few frames. The batter's
skeleton comes from `yolo11m-pose`, matched to the batter's box.

## Models and clip

`run.sh` points `mp_yolo` at `models/baseball/yolo11m` and `mp_yolo_pose` at
`models/baseball/yolo11m-pose`. Ultralytics downloads the weights there on
first use. `MODEL=yolo11n` and `MODEL=yolo11x` work the
same way.

The default clip `data/baseball_batter.mp4` is Mixkit's free
["Batter hits a baseball"](https://mixkit.co/free-stock-video/batter-hits-a-baseball-853/)
(17 s, 1280x720, 30 fps, filmed from behind the batter), under the
[Mixkit Stock Video Free License](https://mixkit.co/license/#videoFree).

## Run

```bash
cd modelpipe
demo/baseball/run.sh                       # data/baseball_batter.mp4 -> demo/baseball/out.mp4
demo/baseball/run.sh display               # same clip, on screen
demo/baseball/run.sh display my_game.mp4 1920x1080
demo/baseball/run.sh camera /dev/video0
HOST=gst demo/baseball/run.sh display      # same pipeline on gst-launch-1.0
CUDA=true demo/baseball/run.sh display     # frames stay on the card from decoder to sink
```

`run.sh` expects the repo venv at `.venv` and `g2g-launch-py` on `PATH`, as the
football demo does.

## Environment knobs

| Var        | Default   | Meaning |
|------------|-----------|---------|
| `HOST`     | `g2g`     | `g2g` runs the python elements under `g2g-launch-py`, `gst` runs the same chain on `gst-launch-1.0`. |
| `CUDA`     | `false`   | `true` decodes, detects, tracks and draws on CUDA memory. |
| `MODEL`    | `yolo11m` | Stock ultralytics weights name. |
| `INTERVAL` | `1`       | Run detection every Nth frame. The ball trail needs 1. |
| `IMGSZ`    | `1280`    | Network input size. The ball is a few pixels wide, 640 misses most of its frames. |
| `CONF`     | `0.25`    | Detector confidence threshold. |
| `NEWTRACK` | `0.4`     | Minimum confidence to start a new track. |
| `BALL_TRAIL` | `true`  | Draw the ball trail. |
| `LABELS`   | `true`    | Write the role next to each person and the class next to bat and glove. |
| `EQUIPMENT` | `true`   | Outline the bat and the glove. |
| `POSE`     | `true`    | Draw the batter's skeleton with `$MODEL-pose`. |

## Overlay properties

`mp_baseball_overlay` takes `show-labels`, `show-ids`, `show-equipment`,
`ball-trail`, `trail-length` and `min-confidence`. The trail resets when the
ball is unseen for 15 frames or jumps more than a tenth of the frame width per
frame, which drops the detector's stray hits on other round objects.
