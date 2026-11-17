# RaceStart
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import concurrent.futures
import json

import backend
from backend import frameio, GObject, InPlaceVideoTransform
from backend.video_frame import (
    RGBA_PIXEL_FORMATS,
    SYSTEM_MEMORY,
    VideoFrameCapabilities,
)
from utils.blob_headers import RACE_WINDOW_META_HEADER
from utils.image_classifier import load_image_classifier
from utils.step_fit import step_fit
from utils.scene_match import SIFT_FEATURES, matching_features, similarity_between

# the start classifier was trained on every fifth frame squashed to 384 x 384
STATE_INTERVAL_FRAMES = 5
STATE_FRAME_SIZE = 384
# the window classifier was trained on frames squashed to 512 x 512
WINDOW_FRAME_SIZE = 512
WINDOW_BEFORE_SECONDS, WINDOW_AFTER_SECONDS = 1.5, 1.0
RISE = 0.3
RISE_SAMPLES = 5
RISE_SEPARATION_SAMPLES = 10
MINIMUM_CONTRAST = 0.5
CUT_BEFORE_FRAMES, CUT_AFTER_FRAMES = 25, 10
CUT_INLIERS = 20
CUT_THREADS = 8
RACE_AFTER_SECONDS = 10.0
MINIMUM_RACE_AFTER = 0.5
# races in the start model training clips last at least 18 s
STARTS_APART_SECONDS = 20.0
WINDOW_CLOSES_SECONDS = 45.0
# candidates are found about 1.3 s after their center and look back 1.5 s
WINDOW_FRAMES_KEPT_SECONDS = 5.0
STATES_KEPT_SECONDS = 15.0


def frame_features(gray):
    import cv2

    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    return matching_features(sift, gray, [], match_scale=1.0)


def frames_match(first_features, second_features):
    import cv2

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    return (
        similarity_between(
            matcher,
            first_features.result(),
            second_features.result(),
            CUT_INLIERS,
            match_scale=1.0,
        )
        is not None
    )


CLASSIFIER_DEVICE = "cuda:0"


class RaceStart(InPlaceVideoTransform):
    __gstmetadata__ = (
        "Race Start",
        "Filter/Analyzer/Video",
        "Finds race starts with frame classifiers and tags frames "
        "from about 1.3 s to 45 s after each start with a race-window blob "
        "carrying the start time, dropping the tags of a start that the next "
        "10 s of frames do not confirm",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    FRAME_CAPABILITIES = VideoFrameCapabilities(
        (SYSTEM_MEMORY,), RGBA_PIXEL_FORMATS, True, False
    )

    state_model = GObject.Property(
        type=str,
        default="",
        nick="State Model",
        blurb="torchvision classifier of frames with a race_running class",
        flags=GObject.ParamFlags.READWRITE,
    )

    window_model = GObject.Property(
        type=str,
        default="",
        nick="Window Model",
        blurb="torchvision classifier of frames before and after the start, "
        "with an after class",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self._frame = 0
        self._state_classifier = None
        self._window_classifier = None
        self._states = []
        self._best_rise = None
        self._after = {}
        self._grays = {}
        self._pending = []
        self._cut_checks = {}
        self._starts = []
        # opencv releases the gil during sift, so the threads run in parallel
        self._feature_pool = concurrent.futures.ThreadPoolExecutor(CUT_THREADS)
        self._match_pool = concurrent.futures.ThreadPoolExecutor(CUT_THREADS)

    def _load_models(self):
        if not self.state_model or not self.window_model:
            raise ValueError("set state-model and window-model")
        self._state_classifier = load_image_classifier(
            self.state_model, CLASSIFIER_DEVICE
        )
        self._window_classifier = load_image_classifier(
            self.window_model, CLASSIFIER_DEVICE
        )
        self._race_running_index = self._state_classifier.classes.index("race_running")
        self._after_index = self._window_classifier.classes.index("after")

    def _classify_state(self, rgb):
        import cv2

        small = cv2.resize(rgb, (STATE_FRAME_SIZE, STATE_FRAME_SIZE))
        probabilities = self._state_classifier.probabilities([small])[0]
        race_running = probabilities[self._race_running_index].item()
        is_race = int(probabilities.argmax()) == self._race_running_index
        self._states.append((self._frame, race_running, is_race))

    def _find_rise(self):
        if len(self._states) <= RISE_SAMPLES:
            return
        change = self._states[-1][1] - self._states[-1 - RISE_SAMPLES][1]
        if change >= RISE and (self._best_rise is None or change > self._best_rise[1]):
            center = self._states[-1 - RISE_SAMPLES // 2][0]
            self._best_rise = (self._frame, change, center)
        if self._best_rise is not None and (
            self._frame - self._best_rise[0]
            > RISE_SEPARATION_SAMPLES * STATE_INTERVAL_FRAMES
        ):
            self._pending.append({"center": self._best_rise[2]})
            self._best_rise = None

    def _refine(self, candidate, frames_per_second):
        first = candidate["center"] - round(WINDOW_BEFORE_SECONDS * frames_per_second)
        last = candidate["center"] + round(WINDOW_AFTER_SECONDS * frames_per_second)
        frames = [f for f in range(first, last) if f in self._after]
        split, contrast = step_fit([self._after[f] for f in frames])
        if contrast < MINIMUM_CONTRAST:
            return False
        candidate["start"] = frames[split]
        candidate["contrast"] = contrast
        cut_frames = frames[
            max(split - CUT_BEFORE_FRAMES, 0) : split + CUT_AFTER_FRAMES
        ]
        features = [
            self._feature_pool.submit(frame_features, self._grays[f])
            for f in cut_frames
        ]
        candidate["matches"] = [
            self._match_pool.submit(frames_match, first, second)
            for first, second in zip(features, features[1:])
        ]
        return True

    def _race_fraction_after(self, start, frames_per_second):
        last = start + RACE_AFTER_SECONDS * frames_per_second
        after = [is_race for frame, _, is_race in self._states if start <= frame < last]
        return sum(after) / len(after)

    def _could_start(self, candidate, frames_per_second):
        if not all(match.result() for match in candidate["matches"]):
            return False
        if self._starts and candidate["start"] - self._starts[-1] < (
            STARTS_APART_SECONDS * frames_per_second
        ):
            return False
        return not any(
            other is not candidate
            and "contrast" in other
            and abs(other["start"] - candidate["start"]) <= frames_per_second
            and other["contrast"] > candidate["contrast"]
            for other in self._pending
        )

    def _provisional_start(self, frames_per_second):
        return max(
            (
                c["start"]
                for c in self._pending
                if "start" in c
                and all(match.done() for match in c["matches"])
                and self._could_start(c, frames_per_second)
            ),
            default=None,
        )

    def _confirm(self, frames_per_second):
        still_pending = []
        for candidate in self._pending:
            window_end = candidate["center"] + WINDOW_AFTER_SECONDS * frames_per_second
            if "start" not in candidate:
                if self._frame < window_end:
                    still_pending.append(candidate)
                elif self._refine(candidate, frames_per_second):
                    still_pending.append(candidate)
                continue
            start = candidate["start"]
            if self._frame < start + RACE_AFTER_SECONDS * frames_per_second:
                still_pending.append(candidate)
                continue
            if not all(match.done() for match in candidate["matches"]):
                still_pending.append(candidate)
                continue
            if not self._could_start(candidate, frames_per_second):
                continue
            if self._race_fraction_after(start, frames_per_second) < MINIMUM_RACE_AFTER:
                continue
            self._starts.append(start)
            self.logger.info(f"race started at {start / frames_per_second:.2f} s")
        self._pending = still_pending

    def _forget_old(self, frames_per_second):
        oldest_window_frame = (
            self._frame - WINDOW_FRAMES_KEPT_SECONDS * frames_per_second
        )
        self._after = {f: a for f, a in self._after.items() if f >= oldest_window_frame}
        self._grays = {f: g for f, g in self._grays.items() if f >= oldest_window_frame}
        oldest_state = self._frame - STATES_KEPT_SECONDS * frames_per_second
        self._states = [s for s in self._states if s[0] >= oldest_state]

    def _tag(self, buf, start, frames_per_second):
        payload = {"start_seconds": start / frames_per_second}
        frameio.append_blob(buf, RACE_WINDOW_META_HEADER, json.dumps(payload).encode())

    def process_in_place(self, frame, fmt, buf):
        import cv2

        if self._state_classifier is None:
            self._load_models()
        if self.framerate is None:
            raise ValueError("the stream has no fixed framerate to time the race")
        numerator, denominator = self.framerate
        frames_per_second = numerator / denominator
        if (
            self._starts
            and self._frame
            < self._starts[-1] + WINDOW_CLOSES_SECONDS * frames_per_second
        ):
            self._tag(buf, self._starts[-1], frames_per_second)
            self._states, self._pending, self._best_rise = [], [], None
            self._frame += 1
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        window_frame = cv2.resize(rgb, (WINDOW_FRAME_SIZE, WINDOW_FRAME_SIZE))
        self._after[self._frame] = self._window_classifier.probabilities(
            [window_frame]
        )[0][self._after_index].item()
        self._grays[self._frame] = cv2.cvtColor(window_frame, cv2.COLOR_RGB2GRAY)
        if self._frame % STATE_INTERVAL_FRAMES == 0:
            self._classify_state(rgb)
            self._find_rise()
        self._confirm(frames_per_second)
        tagged_start = (
            self._starts[-1]
            if self._starts
            and self._frame
            < self._starts[-1] + WINDOW_CLOSES_SECONDS * frames_per_second
            else self._provisional_start(frames_per_second)
        )
        if tagged_start is not None:
            self._tag(buf, tagged_start, frames_per_second)
        self._forget_old(frames_per_second)
        self._frame += 1


if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("mp_race_start", RaceStart)
