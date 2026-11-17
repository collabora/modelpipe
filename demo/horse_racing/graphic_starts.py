import argparse
import csv
import pathlib
import subprocess

import cv2
import numpy as np

from find_races import (
    BROADCAST_SIZE,
    LOGO_LEFT,
    LOGO_TOP,
    MATCH_THRESHOLD,
    SEARCH_MARGIN,
)

SCANNED_SECONDS = 60
# the graphic stays up once a race is running
STEADY_FRAMES = 30


def logo_scores(clip, logo):
    width, height = BROADCAST_SIZE
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(clip),
            "-t",
            str(SCANNED_SECONDS),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width)
    logo_height, logo_width = logo.shape
    windows = frames[
        :,
        LOGO_TOP - SEARCH_MARGIN : LOGO_TOP + logo_height + SEARCH_MARGIN,
        LOGO_LEFT - SEARCH_MARGIN : LOGO_LEFT + logo_width + SEARCH_MARGIN,
    ]
    return np.array(
        [
            cv2.matchTemplate(window, logo, cv2.TM_CCOEFF_NORMED).max()
            for window in windows
        ]
    )


def graphic_frame(scores):
    shown = scores >= MATCH_THRESHOLD
    for frame in range(len(shown) - STEADY_FRAMES):
        if shown[frame : frame + STEADY_FRAMES].all():
            return frame
    raise ValueError("the race timing graphic never stays up")


def main():
    parser = argparse.ArgumentParser(
        description="Find the first frame of each race clip with the race timing graphic"
    )
    parser.add_argument(
        "clips", type=pathlib.Path, help="directory of clips written by cut_races.py"
    )
    parser.add_argument(
        "output", type=pathlib.Path, help="CSV of clip and graphic frame"
    )
    arguments = parser.parse_args()
    logo = cv2.imread(
        str(pathlib.Path(__file__).with_name("inh_logo.png")), cv2.IMREAD_GRAYSCALE
    )
    with open(arguments.output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["clip", "graphic_frame"])
        for clip in sorted(arguments.clips.glob("*.webm")):
            frame = graphic_frame(logo_scores(clip, logo))
            writer.writerow([clip.stem, frame])
            print(clip.stem, frame)


if __name__ == "__main__":
    main()
