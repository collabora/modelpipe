import argparse
import csv
import pathlib
import subprocess

import cv2
import numpy as np
from rfdetr import RFDETRMedium

from find_races import BROADCAST_SIZE

COCO_HORSE = 19
CONFIDENCE = 0.5
MINIMUM_HEIGHT = 30
FRAME_STEP = 10
# the running order bar covers the bottom of the frame during a race
BAR_TOP = 575


def race_frames(clip, first_frame):
    width, height = BROADCAST_SIZE
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(clip),
            "-vf",
            f"select='gte(n\\,{first_frame})*not(mod(n-{first_frame}\\,{FRAME_STEP}))'",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    frame = first_frame
    while chunk := process.stdout.read(width * height * 3):
        yield frame, np.frombuffer(chunk, np.uint8).reshape(height, width, 3)
        frame += FRAME_STEP
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {clip}")


def main():
    parser = argparse.ArgumentParser(
        description="Save a crop of each horse the stock COCO RF-DETR finds in the race clips"
    )
    parser.add_argument("clips", type=pathlib.Path)
    parser.add_argument(
        "graphic_starts", type=pathlib.Path, help="CSV written by graphic_starts.py"
    )
    parser.add_argument(
        "output", type=pathlib.Path, help="directory for the crops and crops.csv"
    )
    arguments = parser.parse_args()
    model = RFDETRMedium()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with open(arguments.output / "crops.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["crop", "clip", "frame", "x", "y", "w", "h", "score"])
        for row in csv.DictReader(open(arguments.graphic_starts)):
            clip = arguments.clips / f"{row['clip']}.webm"
            for frame, rgb in race_frames(clip, int(row["graphic_frame"])):
                detections = model.predict(rgb, threshold=CONFIDENCE)
                for box, class_id, score in zip(
                    detections.xyxy, detections.class_id, detections.confidence
                ):
                    x1, y1, x2, y2 = (int(round(value)) for value in box)
                    if (
                        class_id != COCO_HORSE
                        or y2 - y1 < MINIMUM_HEIGHT
                        or y2 > BAR_TOP
                    ):
                        continue
                    name = f"{row['clip']}_{frame:05d}_{x1:04d}_{y1:03d}"
                    cv2.imwrite(
                        str(arguments.output / f"{name}.jpg"),
                        cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR),
                    )
                    writer.writerow(
                        [
                            name,
                            row["clip"],
                            frame,
                            x1,
                            y1,
                            x2 - x1,
                            y2 - y1,
                            f"{score:.3f}",
                        ]
                    )
            print(row["clip"])


if __name__ == "__main__":
    main()
