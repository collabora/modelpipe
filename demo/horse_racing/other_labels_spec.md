# Scene labels for frames away from live races

Each sheet `DATA/other_sheets/NNNN.png` has 30 numbered frames from Venezuelan TV broadcasts of the La Rinconada racecourse, taken away from the live races. `NNNN.csv` maps each cell number to a frame name. Give each cell one class:

- `gate_scene`: the starting gate is the main subject, with horses in the stalls, being loaded or standing around it. A close-up of horses and handlers at the stalls counts.
- `horses_running`: horses cantering or galloping on the track, in any shot: warm-ups before a race, replays, slow motion, aerial shots of a race, horses pulling up after the finish.
- `other`: everything else. Studio, presenters, interviews, crowd, paddock and parade ring, horses walking, empty track, the gate with no horses, graphics and results tables, black or blank frames.

When a frame is a split screen or picture in picture, use the class of the largest part. When a frame fits two classes, `horses_running` wins over `gate_scene`, and `gate_scene` wins over `other`.

Write `DATA/other_labels/<batch>.csv` with the header `sheet,number,class`, one row per filled cell, `sheet` as the four digit name without extension.
