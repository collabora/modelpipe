# Doors-open frame per race

Each clip in `DATA/clips/<clip>.webm` is one live race from a Venezuelan TV broadcast, cut about 30 s before the race timing graphic appears. `DATA/labels/graphic_starts.csv` gives `graphic_frame`, the first frame with that graphic. The stall doors open somewhere between about 2.5 s and 0.3 s before it. Clips run at 59.94 fps (`reunion_01`) or 30 fps (check each clip with ffprobe).

For each race, find `door_frame`: the first frame in which the front doors of the starting stalls have visibly moved open. When the doors themselves can't be seen (a side view, a flag or post in the way), use the first frame in which the front of any horse has moved past the front of the stalls, and set `cue` to `horses` instead of `doors`.

Frame numbers are 0-based decoded frame indices, as `ffmpeg -i CLIP -vf "select=eq(n\,N)" -fps_mode passthrough -frames:v 1 out.png` counts them. Work coarse to fine: first a strip of frames every 6 frames from `graphic_frame - 3 s` to `graphic_frame`, then every frame around the change, and crop or enlarge the stall region to see the doors. Look at the frames one before and one after your answer to confirm it.

If the start is not shown live (the camera is elsewhere at the moment the doors open), leave `door_frame` empty and say what is shown in `note`.

Write `DATA/labels/doors/<broadcast>.csv`, one file per broadcast such as `reunion_01.csv`, with the header `clip,door_frame,cue,note` and one row per clip of that broadcast. Write scratch images only under `/tmp/doors_<broadcast>/`. Do not change any other file.
