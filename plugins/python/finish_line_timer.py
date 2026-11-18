# FinishLineTimer
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json

import backend
from backend import analytics, frameio, GObject, InPlaceVideoTransform
from backend.video_frame import (
    RGBA_PIXEL_FORMATS,
    SYSTEM_MEMORY,
    VideoFrameCapabilities,
)
from utils.blob_headers import (
    FINISH_LINE_META_HEADER,
    RACE_WINDOW_BLOB_NAME,
    RACER_NUMBERS_BLOB_NAME,
)
from utils.finish_crossing import crossing
from utils.scene_match import (
    SIFT_FEATURES,
    matching_features,
    similarity_between,
    transform_points,
)

# sift takes 5 ms at scale 0.25 and 33 ms at scale 0.5
MATCH_SCALE = 0.25
FRAME_TO_FRAME_INLIERS = 20
LINE_SEARCH_FROM_SECONDS, LINE_SEARCH_TO_SECONDS = 15.0, 40.0
LINE_INTERVAL_FRAMES = 5
MINIMUM_LINE_CONFIDENCE = 0.5
LINE_AGREEMENT_PIXELS = 30.0
RACERS_BEFORE_LINE_SECONDS, RACERS_AFTER_LINE_SECONDS = 3.0, 2.0
# crossing() takes 7 ms over 3 s of one racer's boxes
RACER_TIMING_INTERVAL_FRAMES = 10
RACER_TIMING_WINDOW_SECONDS = 1.0


LINE_DEVICE = "cuda:0"


class FinishLineTimer(InPlaceVideoTransform):
    __gstmetadata__ = (
        "Finish Line Timer",
        "Filter/Effect/Video",
        "On frames carrying a race-window blob, finds the finish line with a "
        "keypoint model, follows it through the camera pan and writes the time "
        "from the race start to each numbered racer crossing it",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    FRAME_CAPABILITIES = VideoFrameCapabilities(
        (SYSTEM_MEMORY,), RGBA_PIXEL_FORMATS, True, False
    )

    line_model = GObject.Property(
        type=str,
        default="",
        nick="Line Model",
        blurb="Heatmap checkpoint for the far and near ends of the finish line",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self._frame = 0
        self._line_detector = None
        self._start_race(None)

    def _start_race(self, race_start_seconds):
        import numpy as np

        self._race_start_seconds = race_start_seconds
        self._history = []
        self._previous_features = None
        self._segment = 0
        self._cumulative = np.eye(3)
        self._last_sample = None
        self._line = None
        self._finished = False
        self._crossing_seconds_by_number = {}
        self.race_time_seconds = None

    def _load_models(self):
        import cv2
        from utils.finish_line_model import load_finish_line_model

        if not self.line_model:
            raise ValueError("set line-model")
        self._line_detector = load_finish_line_model(self.line_model, LINE_DEVICE)
        self._sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    def _tracked_racers(self, buf):
        meta = analytics.get_relation_meta(buf)
        if not meta:
            return []
        # mp_racer_number_tracker leaves boxes with no number untracked
        return [
            obj for obj in analytics.read_objects(meta) if obj["track_id"] is not None
        ]

    def _follow_camera(self, gray, racer_boxes):
        import numpy as np

        features = matching_features(self._sift, gray, racer_boxes, MATCH_SCALE)
        step = None
        if self._previous_features is not None:
            step = similarity_between(
                self._matcher,
                self._previous_features,
                features,
                FRAME_TO_FRAME_INLIERS,
                match_scale=MATCH_SCALE,
            )
        if step is None:
            self._segment += 1
            self._cumulative = np.eye(3)
        else:
            self._cumulative = step @ self._cumulative
        self._previous_features = features

    def _search_line(self, rgb):
        import numpy as np

        line = self._line_detector.predict(rgb)
        sample = {
            "frame": self._frame,
            "confidence": line.confidence,
            "points": line.ends,
            "segment": self._segment,
            "cumulative": self._cumulative,
        }
        previous = self._last_sample
        self._last_sample = sample
        if previous is None or previous["segment"] != self._segment:
            return
        if min(sample["confidence"], previous["confidence"]) < MINIMUM_LINE_CONFIDENCE:
            return
        # near ends are at least 100 px below far ends on all 40 labelled tracks
        near_end_above_far_end = any(
            near[1] <= far[1] for far, near in (sample["points"], previous["points"])
        )
        if near_end_above_far_end:
            return
        moved = transform_points(
            self._cumulative @ np.linalg.inv(previous["cumulative"]), previous["points"]
        )
        distance = float(np.linalg.norm(moved - sample["points"], axis=1).mean())
        if distance <= LINE_AGREEMENT_PIXELS:
            self._line = sample
            self.logger.info(
                f"finish line found at {self._frame} with confidence "
                f"{sample['confidence']:.2f}"
            )

    def _line_transform(self, cumulative):
        import numpy as np

        return cumulative @ np.linalg.inv(self._line["cumulative"])

    def _time_crossing(self, frames_per_second):
        line_segment = [
            e for e in self._history if e["segment"] == self._line["segment"]
        ]
        transforms = {
            e["frame"]: self._line_transform(e["cumulative"]) for e in line_segment
        }
        boxes = {e["frame"]: e["racer_boxes"] for e in self._history}
        far, near = self._line["points"]
        result = crossing(transforms, far, near, boxes, 0.0, frames_per_second)
        if "seconds" in result:
            self.race_time_seconds = result["seconds"] - self._race_start_seconds
            self._finished = True
            self.logger.info(
                f"first racer finishes in {self.race_time_seconds:.3f} s ({result['status']})"
            )
        elif (
            self._frame
            > self._line["frame"] + RACERS_AFTER_LINE_SECONDS * frames_per_second
        ):
            self._finished = True
            self.logger.info(f"no finish time: {result['status']}")

    def _time_racers(self, frames_per_second, window_seconds):
        first_frame = self._frame - window_seconds * frames_per_second
        line_segment = [
            e
            for e in self._history
            if e["segment"] == self._line["segment"] and e["frame"] >= first_frame
        ]
        transforms = {
            e["frame"]: self._line_transform(e["cumulative"]) for e in line_segment
        }
        far, near = self._line["points"]
        untimed = {
            number for e in line_segment for number in e["numbered_boxes"]
        } - self._crossing_seconds_by_number.keys()
        for number in sorted(untimed):
            boxes = {
                e["frame"]: [e["numbered_boxes"][number]]
                for e in line_segment
                if number in e["numbered_boxes"]
            }
            result = crossing(transforms, far, near, boxes, 0.0, frames_per_second)
            if "seconds" in result:
                self._crossing_seconds_by_number[number] = result["seconds"]
                self.logger.info(
                    f"racer {number} finishes in "
                    f"{result['seconds'] - self._race_start_seconds:.3f} s "
                    f"({result['status']})"
                )

    def process_in_place(self, frame, fmt, buf):
        import cv2

        if self._line_detector is None:
            self._load_models()
        if self.framerate is None:
            raise ValueError("the stream has no fixed framerate to time the finish")
        numerator, denominator = self.framerate
        frames_per_second = numerator / denominator
        blobs = frameio.read_blobs(buf)
        window_blob = blobs.get(RACE_WINDOW_BLOB_NAME)
        race_start_seconds = (
            None if window_blob is None else json.loads(window_blob)["start_seconds"]
        )
        if race_start_seconds is not None and (
            race_start_seconds != self._race_start_seconds
        ):
            self._start_race(race_start_seconds)
        finish = {}
        if race_start_seconds is not None:
            numbers_by_track_id = json.loads(blobs[RACER_NUMBERS_BLOB_NAME])[
                "numbers_by_track_id"
            ]
            racers = self._tracked_racers(buf)
            racer_boxes = [
                (d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]) for d in racers
            ]
            racer_numbers = [
                numbers_by_track_id.get(str(d["track_id"])) for d in racers
            ]
            self._follow_camera(cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY), racer_boxes)
            self._history.append(
                {
                    "frame": self._frame,
                    "racer_boxes": racer_boxes,
                    "numbered_boxes": {
                        number: box
                        for number, box in zip(racer_numbers, racer_boxes)
                        if number is not None
                    },
                    "segment": self._segment,
                    "cumulative": self._cumulative,
                }
            )
            oldest = (
                self._line["frame"] if self._line else self._frame
            ) - RACERS_BEFORE_LINE_SECONDS * frames_per_second
            self._history = [e for e in self._history if e["frame"] >= oldest]
            since_start = self._frame / frames_per_second - race_start_seconds
            if (
                self._line is None
                and LINE_SEARCH_FROM_SECONDS <= since_start < LINE_SEARCH_TO_SECONDS
                and self._frame % LINE_INTERVAL_FRAMES == 0
            ):
                self._search_line(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB))
            if self._line is not None and not self._finished:
                self._time_crossing(frames_per_second)
            if self._line is not None:
                frames_since_line = self._frame - self._line["frame"]
                if frames_since_line == 0:
                    self._time_racers(frames_per_second, RACERS_BEFORE_LINE_SECONDS)
                elif frames_since_line % RACER_TIMING_INTERVAL_FRAMES == 0:
                    self._time_racers(frames_per_second, RACER_TIMING_WINDOW_SECONDS)
            if self._line is not None and self._line["segment"] == self._segment:
                far, near = transform_points(
                    self._line_transform(self._cumulative), self._line["points"]
                )
                finish["line"] = [far.tolist(), near.tolist()]
        if self.race_time_seconds is not None:
            finish["first_crossing_seconds"] = (
                self._race_start_seconds + self.race_time_seconds
            )
        if self._crossing_seconds_by_number:
            finish["crossing_seconds_by_racer_number"] = {
                str(number): seconds
                for number, seconds in self._crossing_seconds_by_number.items()
            }
        if finish:
            frameio.append_blob(
                buf, FINISH_LINE_META_HEADER, json.dumps(finish).encode()
            )
        self._frame += 1


if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_finish_line_timer", FinishLineTimer
    )
