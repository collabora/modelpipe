# RacerNumberTracker
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import collections
import dataclasses
import json

import backend
from backend import analytics, frameio, InPlaceVideoTransform
from backend.video_frame import (
    RGBA_PIXEL_FORMATS,
    SYSTEM_MEMORY,
    VideoFrameCapabilities,
)
from tracker import iou_batch, merge_covered_detections
from utils.blob_headers import (
    RACE_WINDOW_BLOB_NAME,
    RACER_NUMBER_PROBABILITIES_BLOB_NAME,
    RACER_NUMBERS_META_HEADER,
)


@dataclasses.dataclass(frozen=True)
class RacerNumberSlotSettings:
    # scores are log likelihoods, values picked on the six eval clips
    same_racer_covered_fraction: float = 0.5
    position_spread_box_sizes: float = 0.3
    position_spread_growth_per_frame: float = 0.02
    unknown_position_score: float = -5.0
    minimum_score: float = -7.0
    unread_box_margin: float = 2.0
    slot_memory_frames: int = 100
    number_evidence_decay: float = 0.5
    same_box_iou: float = 0.3
    minimum_number_weight: float = 0.3
    field_entry_frames: int = 25
    # on Segula C misread numbers reached 7.5% of the most read number's frames, runners 18%
    field_entry_share: float = 0.1


class RacerNumberSlots:
    def __init__(self, settings=RacerNumberSlotSettings()):
        self._settings = settings
        self._race_start_seconds = None
        self._start_race()

    def _start_race(self):
        self._last_seen_by_panel = collections.defaultdict(dict)
        self._read_frames_by_number = collections.Counter()
        self._previous_frame = None
        self._previous_boxes = []
        self._previous_evidence = []

    def _position_scores(self, frame, panel, box, number_count):
        import numpy as np

        settings = self._settings
        scores = np.full(number_count, settings.unknown_position_score)
        centre = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])
        for number, (last_box, last_frame) in self._last_seen_by_panel[panel].items():
            frames_since = frame - last_frame
            if frames_since > settings.slot_memory_frames:
                continue
            last_centre = np.array(
                [last_box[0] + last_box[2] / 2, last_box[1] + last_box[3] / 2]
            )
            distance = np.linalg.norm(centre - last_centre) / max(last_box[2:])
            spread = (
                settings.position_spread_box_sizes
                + settings.position_spread_growth_per_frame * frames_since
            )
            scores[number - 1] = max(
                -0.5 * (distance / spread) ** 2, settings.unknown_position_score
            )
        return scores

    def _number_evidence(self, frame, boxes, readings, frame_width, number_count):
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        evidence = []
        for box, reading in zip(boxes, readings):
            cloth_probabilities = np.array(reading[:number_count])
            given_a_number = cloth_probabilities / max(cloth_probabilities.sum(), 1e-9)
            # number cloths on racers cut off by the frame edge read as other numbers
            cut_off = box[0] <= 0 or box[0] + box[2] >= frame_width
            weight = 0.0 if cut_off else 1.0 - reading[number_count]
            evidence.append((weight, np.log(np.maximum(given_a_number, 1e-3))))
        if self._previous_frame != frame - 1:
            return evidence
        overlaps = iou_batch(boxes, self._previous_boxes)
        rows, columns = linear_sum_assignment(overlaps, maximize=True)
        for row, column in zip(rows, columns):
            if overlaps[row, column] < self._settings.same_box_iou:
                continue
            weight, log_probabilities = evidence[row]
            previous_weight, previous_log_probabilities = self._previous_evidence[
                column
            ]
            carried = self._settings.number_evidence_decay * previous_weight
            if carried == 0:
                continue
            total = carried + weight
            evidence[row] = (
                total,
                (carried * previous_log_probabilities + weight * log_probabilities)
                / total,
            )
        return evidence

    def follow_race(self, race_start_seconds):
        if race_start_seconds != self._race_start_seconds:
            self._race_start_seconds = race_start_seconds
            self._start_race()

    def field(self):
        if not self._read_frames_by_number:
            return []
        settings = self._settings
        most_frames = max(self._read_frames_by_number.values())
        return sorted(
            number
            for number, frames in self._read_frames_by_number.items()
            if frames >= settings.field_entry_frames
            and frames >= settings.field_entry_share * most_frames
        )

    def assign(self, frame, objects, number_probabilities, frame_width, divider_row):
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        racers = [
            row[5]
            for row in merge_covered_detections(
                [[o["x"], o["y"], o["w"], o["h"], o["score"], o] for o in objects],
                self._settings.same_racer_covered_fraction,
            )
        ]
        boxes = [[racer["x"], racer["y"], racer["w"], racer["h"]] for racer in racers]
        readings = [number_probabilities[str(racer["ref"])] for racer in racers]
        # readings end with the probability of no number
        number_count = len(readings[0]) - 1
        evidence = self._number_evidence(
            frame, boxes, readings, frame_width, number_count
        )
        self._previous_frame = frame
        self._previous_boxes = boxes
        self._previous_evidence = evidence
        settings = self._settings
        is_read = [weight >= settings.minimum_number_weight for weight, _ in evidence]
        for read, (_, log_probabilities) in zip(is_read, evidence):
            if read:
                self._read_frames_by_number[int(np.argmax(log_probabilities)) + 1] += 1
        field = self.field()
        if not field:
            return []
        panels = [
            0 if divider_row is None or box[1] + box[3] / 2 < divider_row else 1
            for box in boxes
        ]
        field_columns = [number - 1 for number in field]
        numbered = []
        for panel in set(panels):
            rows = [row for row, box_panel in enumerate(panels) if box_panel == panel]
            scores = np.zeros((len(rows), len(field)))
            for index, row in enumerate(rows):
                number_scores = (
                    evidence[row][1]
                    if is_read[row]
                    else np.full(number_count, np.log(1 / number_count))
                )
                scores[index] = (
                    number_scores
                    + self._position_scores(frame, panel, boxes[row], number_count)
                )[field_columns]
            indices, columns = linear_sum_assignment(scores, maximize=True)
            for index, column in zip(indices, columns):
                row = rows[index]
                score = scores[index, column]
                if score < settings.minimum_score:
                    continue
                other_scores = np.delete(scores[index], column)
                unread_without_margin = (
                    not is_read[row]
                    and other_scores.size > 0
                    and score - other_scores.max() < settings.unread_box_margin
                )
                if unread_without_margin:
                    continue
                number = field[column]
                self._last_seen_by_panel[panel][number] = (boxes[row], frame)
                numbered.append((racers[row], number))
        return numbered


# la rinconada split screen banners have a line of 3 to 8 grey levels across the whole width, 4 rows tall at 720p
DIVIDER_DARK_LEVEL = 40
DIVIDER_DARK_SHARE = 0.98
DIVIDER_CLEARANCE_ROWS_PER_720 = 10


def split_screen_divider_row(frame):
    import numpy as np

    height = frame.shape[0]
    clearance = round(height * DIVIDER_CLEARANCE_ROWS_PER_720 / 720)
    first_row, last_row = int(height * 0.3), int(height * 0.7)
    band = frame[first_row - clearance : last_row + clearance, :, :3]
    dark_share = (band.max(axis=2) < DIVIDER_DARK_LEVEL).mean(axis=1)
    row_count = last_row - first_row
    is_divider = (
        (dark_share[clearance : clearance + row_count] >= DIVIDER_DARK_SHARE)
        & (dark_share[:row_count] < 0.5)
        & (dark_share[2 * clearance : 2 * clearance + row_count] < 0.5)
    )
    divider_rows = np.flatnonzero(is_divider)
    return int(first_row + divider_rows[0]) if divider_rows.size else None


class RacerNumberTracker(InPlaceVideoTransform):
    __gstmetadata__ = (
        "Racer Number Tracker",
        "Filter/Analyzer/Video",
        "On frames carrying a race-window blob, gives each racer box one of the "
        "racer numbers from its number cloth readings on this and earlier "
        "frames and its distance from where each number was last seen, tags the "
        "box with the number as its track id and writes a racer-numbers blob "
        "with the numbers and the field, the numbers read on enough frames of "
        "the race. On split screen frames, where a dark line crosses the middle "
        "of the frame, the boxes above and below the line are numbered "
        "separately, so a number can be on one box in each view",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    FRAME_CAPABILITIES = VideoFrameCapabilities(
        (SYSTEM_MEMORY,), RGBA_PIXEL_FORMATS, True, False
    )

    def __init__(self):
        super().__init__()
        self._frame = -1
        self._slots = RacerNumberSlots()

    def process_in_place(self, frame, fmt, buf):
        self._frame += 1
        blobs = frameio.read_blobs(buf)
        window_blob = blobs.get(RACE_WINDOW_BLOB_NAME)
        if window_blob is None:
            return
        self._slots.follow_race(json.loads(window_blob)["start_seconds"])
        meta = analytics.get_relation_meta(buf)
        objects = analytics.read_objects(meta) if meta else []
        numbered = []
        if objects:
            numbered = self._slots.assign(
                self._frame,
                objects,
                json.loads(blobs[RACER_NUMBER_PROBABILITIES_BLOB_NAME]),
                self.width,
                split_screen_divider_row(frame),
            )
        for racer, number in numbered:
            analytics.track_object(meta, racer, number)
        blob = {
            "numbers_by_track_id": {str(number): number for _, number in numbered},
            "field": self._slots.field(),
        }
        frameio.append_blob(buf, RACER_NUMBERS_META_HEADER, json.dumps(blob).encode())


if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_racer_number_tracker", RacerNumberTracker
    )
