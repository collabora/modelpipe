import argparse
import csv
import pathlib
import subprocess

import cv2
import numpy as np

from find_races import BROADCAST_SIZE
from race_start import STATE_FRAME_SIZE, STATE_INTERVAL_FRAMES, WINDOW_FRAME_SIZE

FIRST_SECONDS, LAST_SECONDS = -6.0, 15.0
WINDOW_SECONDS = 1.0
# horses running away from the live races are replays and warm-ups
STATE_CLASSES_AWAY_FROM_RACES = {"other", "gate_scene"}


def clip_frames(clip, first_frame, last_frame):
    width, height = BROADCAST_SIZE
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(clip),
            "-vf",
            f"select='between(n\\,{first_frame}\\,{last_frame})'",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    frame = first_frame
    while chunk := process.stdout.read(width * height * 3):
        yield frame, np.frombuffer(chunk, np.uint8).reshape(height, width, 3)
        frame += 1
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {clip}")
    if frame != last_frame + 1:
        raise RuntimeError(f"{clip} ends at frame {frame}, before {last_frame}")


def main():
    parser = argparse.ArgumentParser(
        description="Save frames around each start, squashed to the start "
        "classifier sizes, with their seconds from the doors opening, and the "
        "labelled frames away from races as state frames of their class"
    )
    parser.add_argument("clips", type=pathlib.Path)
    parser.add_argument(
        "doors",
        type=pathlib.Path,
        help="directory of CSV files with the door_frame of each clip",
    )
    parser.add_argument(
        "other_frames",
        type=pathlib.Path,
        help="directory of <broadcast>_<seconds>.jpg written by sample_other_frames.py",
    )
    parser.add_argument(
        "other_labels",
        type=pathlib.Path,
        help="labels.csv of merge_sheet_labels.py other",
    )
    parser.add_argument(
        "output", type=pathlib.Path, help="directory for state/, window/ and frames.csv"
    )
    arguments = parser.parse_args()
    for kind in ("state", "window"):
        (arguments.output / kind).mkdir(parents=True, exist_ok=True)
    with open(arguments.output / "frames.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["kind", "clip", "frame", "seconds_from_start", "class", "path"]
        )
        rows = [
            row
            for path in sorted(arguments.doors.glob("*.csv"))
            for row in csv.DictReader(open(path))
            if row["door_frame"]
        ]
        for row in rows:
            clip = arguments.clips / f"{row['clip']}.webm"
            numerator, denominator = (
                subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=r_frame_rate",
                        "-of",
                        "csv=p=0",
                        str(clip),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                .stdout.strip()
                .split("/")
            )
            frames_per_second = int(numerator) / int(denominator)
            start = int(row["door_frame"])
            first = start + round(FIRST_SECONDS * frames_per_second)
            last = start + round(LAST_SECONDS * frames_per_second)
            for frame, image in clip_frames(clip, max(first, 0), last):
                seconds = (frame - start) / frames_per_second
                saved = []
                if frame % STATE_INTERVAL_FRAMES == 0:
                    saved.append(("state", STATE_FRAME_SIZE))
                if abs(seconds) <= WINDOW_SECONDS:
                    saved.append(("window", WINDOW_FRAME_SIZE))
                for kind, size in saved:
                    path = f"{kind}/{row['clip']}_{frame:05d}.jpg"
                    cv2.imwrite(
                        str(arguments.output / path),
                        cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA),
                    )
                    writer.writerow(
                        [kind, row["clip"], frame, f"{seconds:.4f}", "", path]
                    )
            print(row["clip"], flush=True)
        for label in csv.DictReader(open(arguments.other_labels)):
            if label["class"] not in STATE_CLASSES_AWAY_FROM_RACES:
                continue
            broadcast = label["image"].rsplit("_", 1)[0]
            path = f"state/{label['image']}.jpg"
            cv2.imwrite(
                str(arguments.output / path),
                cv2.resize(
                    cv2.imread(str(arguments.other_frames / f"{label['image']}.jpg")),
                    (STATE_FRAME_SIZE, STATE_FRAME_SIZE),
                    interpolation=cv2.INTER_AREA,
                ),
            )
            writer.writerow(["state", broadcast, "", "", label["class"], path])


if __name__ == "__main__":
    main()
