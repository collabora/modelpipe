import argparse
import csv
import json
import pathlib
import subprocess

import cv2
import numpy as np

BROADCAST_SIZE = (1280, 720)
LOGO_LEFT, LOGO_TOP = 32, 50
SEARCH_MARGIN = 8
MATCH_THRESHOLD = 0.7
MERGE_GAP_SECONDS = 15.0
MINIMUM_RACE_SECONDS = 20.0
THUMBNAIL_SIZE = (64, 36)
THUMBNAIL_SECONDS = 60
MAXIMUM_REPLAY_SHIFT_SECONDS = 8
MINIMUM_COMPARED_SECONDS = 20
# replays on the first broadcast differ from their race by at most 12.7 grey levels, other races by at least 20.8
REPLAY_DISTANCE = 17.0


# ffmpeg's vp9 decoder ignores -skip_frame nokey and outputs every frame
KEYFRAMES_ONLY = "select='eq(key\\,1)'"


def keyframe_times(path):
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,flags",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [
        float(packet["pts_time"])
        for packet in json.loads(output)["packets"]
        if "K" in packet["flags"]
    ]


def keyframe_logo_scores(path, logo):
    width, height = BROADCAST_SIZE
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-skip_frame",
            "nokey",
            "-i",
            str(path),
            "-vf",
            KEYFRAMES_ONLY,
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    logo_height, logo_width = logo.shape
    top, left = LOGO_TOP - SEARCH_MARGIN, LOGO_LEFT - SEARCH_MARGIN
    scores = []
    while chunk := process.stdout.read(width * height):
        frame = np.frombuffer(chunk, np.uint8).reshape(height, width)
        window = frame[
            top : LOGO_TOP + logo_height + SEARCH_MARGIN,
            left : LOGO_LEFT + logo_width + SEARCH_MARGIN,
        ]
        scores.append(
            float(cv2.matchTemplate(window, logo, cv2.TM_CCOEFF_NORMED).max())
        )
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {path}")
    return scores


def race_segments(times, scores):
    segments = []
    for time, score in zip(times, scores):
        if score < MATCH_THRESHOLD:
            continue
        if segments and time - segments[-1][1] <= MERGE_GAP_SECONDS:
            segments[-1][1] = time
        else:
            segments.append([time, time])
    return [
        (start, end) for start, end in segments if end - start >= MINIMUM_RACE_SECONDS
    ]


def segment_thumbnails(path, start):
    width, height = THUMBNAIL_SIZE
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            str(start),
            "-i",
            str(path),
            "-t",
            str(THUMBNAIL_SECONDS),
            "-vf",
            f"fps=1,scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width).astype(np.float32)


def shifted_distance(first, second):
    distances = []
    for shift in range(-MAXIMUM_REPLAY_SHIFT_SECONDS, MAXIMUM_REPLAY_SHIFT_SECONDS + 1):
        a, b = (first[shift:], second) if shift >= 0 else (first, second[-shift:])
        count = min(len(a), len(b))
        if count >= MINIMUM_COMPARED_SECONDS:
            distances.append(float(np.abs(a[:count] - b[:count]).mean()))
    return min(distances)


def live_races(path, segments):
    kept = []
    for start, end in segments:
        thumbnails = segment_thumbnails(path, start)
        if all(
            shifted_distance(earlier, thumbnails) > REPLAY_DISTANCE
            for _, _, earlier in kept
        ):
            kept.append((start, end, thumbnails))
    return [(start, end) for start, end, _ in kept]


def main():
    parser = argparse.ArgumentParser(
        description="Find the live races in a TVes La Rinconada broadcast by its race timing graphic, leaving out replays"
    )
    parser.add_argument("broadcast", type=pathlib.Path)
    parser.add_argument(
        "output", type=pathlib.Path, help="CSV of race start and end seconds"
    )
    arguments = parser.parse_args()
    logo = cv2.imread(
        str(pathlib.Path(__file__).with_name("inh_logo.png")), cv2.IMREAD_GRAYSCALE
    )
    times = keyframe_times(arguments.broadcast)
    scores = keyframe_logo_scores(arguments.broadcast, logo)
    if len(times) != len(scores):
        raise RuntimeError(
            f"{len(times)} keyframe times but {len(scores)} keyframes decoded"
        )
    with open(arguments.output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["broadcast", "start_seconds", "end_seconds"])
        for start, end in live_races(arguments.broadcast, race_segments(times, scores)):
            writer.writerow([arguments.broadcast.stem, f"{start:.2f}", f"{end:.2f}"])
            print(f"{start:8.1f} {end:8.1f} {end - start:6.1f} s")


if __name__ == "__main__":
    main()
