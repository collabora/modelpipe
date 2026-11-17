import argparse
import pathlib
import subprocess
import tempfile

import cv2

from find_races import (
    KEYFRAMES_ONLY,
    keyframe_logo_scores,
    keyframe_times,
    race_segments,
)

SAMPLE_SPACING_SECONDS = 20.0
BEFORE_RACE_SECONDS = 60.0
AFTER_RACE_SECONDS = 30.0
SAMPLE_SIZE = "640:360"


def main():
    parser = argparse.ArgumentParser(
        description="Save broadcast frames away from every race and replay, as other frames for the start classifier"
    )
    parser.add_argument("broadcast", type=pathlib.Path)
    parser.add_argument(
        "output", type=pathlib.Path, help="directory for <broadcast>_<seconds>.jpg"
    )
    arguments = parser.parse_args()
    logo = cv2.imread(
        str(pathlib.Path(__file__).with_name("inh_logo.png")), cv2.IMREAD_GRAYSCALE
    )
    times = keyframe_times(arguments.broadcast)
    segments = race_segments(times, keyframe_logo_scores(arguments.broadcast, logo))
    arguments.output.mkdir(parents=True, exist_ok=True)
    # path.rename fails across filesystems
    with tempfile.TemporaryDirectory(dir=arguments.output) as keyframes:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-skip_frame",
                "nokey",
                "-i",
                str(arguments.broadcast),
                "-fps_mode",
                "passthrough",
                "-vf",
                f"{KEYFRAMES_ONLY},scale={SAMPLE_SIZE}",
                "-q:v",
                "3",
                f"{keyframes}/%06d.jpg",
            ],
            check=True,
        )
        decoded = sorted(pathlib.Path(keyframes).glob("*.jpg"))
        if len(decoded) != len(times):
            raise RuntimeError(
                f"{len(times)} keyframe times but {len(decoded)} keyframes decoded"
            )
        last_sample = -SAMPLE_SPACING_SECONDS
        for time, path in zip(times, decoded):
            near_race = any(
                start - BEFORE_RACE_SECONDS <= time <= end + AFTER_RACE_SECONDS
                for start, end in segments
            )
            if near_race or time - last_sample < SAMPLE_SPACING_SECONDS:
                continue
            path.rename(
                arguments.output / f"{arguments.broadcast.stem}_{time:08.1f}.jpg"
            )
            last_sample = time


if __name__ == "__main__":
    main()
