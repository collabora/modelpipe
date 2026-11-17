import argparse
import csv
import pathlib
import subprocess

# the gate opens before the race timing graphic appears
BEFORE_GRAPHIC_SECONDS = 30.0
AFTER_GRAPHIC_SECONDS = 10.0


def main():
    parser = argparse.ArgumentParser(
        description="Cut each race that find_races.py found into its own clip"
    )
    parser.add_argument("broadcast", type=pathlib.Path)
    parser.add_argument("races", type=pathlib.Path, help="CSV written by find_races.py")
    parser.add_argument(
        "output", type=pathlib.Path, help="directory for <broadcast>_race_<n>.webm"
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(csv.DictReader(open(arguments.races)), 1):
        start = float(row["start_seconds"]) - BEFORE_GRAPHIC_SECONDS
        end = float(row["end_seconds"]) + AFTER_GRAPHIC_SECONDS
        clip = arguments.output / f"{arguments.broadcast.stem}_race_{number}.webm"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                f"{start:.2f}",
                "-to",
                f"{end:.2f}",
                "-i",
                str(arguments.broadcast),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                str(clip),
            ],
            check=True,
        )
        print(clip)


if __name__ == "__main__":
    main()
