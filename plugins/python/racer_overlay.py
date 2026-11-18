# RacerOverlay
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import dataclasses
import json

import backend
from backend import (
    analytics,
    frameio,
    GObject,
    InPlaceVideoTransform,
    CUDA_DRAW_FRAME_CAPABILITIES,
)
from sport_overlay_drawing import drawing_for
from utils.blob_headers import (
    FINISH_LINE_BLOB_NAME,
    RACE_WINDOW_BLOB_NAME,
    RACER_NUMBERS_BLOB_NAME,
)


@dataclasses.dataclass(frozen=True)
class NumberCloth:
    cloth_rgba: tuple
    number_rgba: tuple
    on_black_rgba: tuple


WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
YELLOW = (255, 220, 0, 255)
# black and dark grey cloths vanish against the black header
LIGHT_GREY = (200, 200, 200, 255)


def number_cloth(cloth_rgba, number_rgba):
    return NumberCloth(cloth_rgba, number_rgba, cloth_rgba)


NUMBER_CLOTHS = {
    "eight-rugs": {
        1: number_cloth((220, 40, 40, 255), WHITE),
        2: NumberCloth((90, 90, 90, 255), WHITE, LIGHT_GREY),
        3: number_cloth(WHITE, BLACK),
        4: number_cloth((30, 120, 220, 255), WHITE),
        5: number_cloth(YELLOW, BLACK),
        6: number_cloth((40, 170, 60, 255), WHITE),
        7: NumberCloth(BLACK, WHITE, LIGHT_GREY),
        8: number_cloth((255, 120, 200, 255), BLACK),
    },
    "horse": {
        1: number_cloth((220, 40, 40, 255), WHITE),
        2: number_cloth(WHITE, BLACK),
        3: number_cloth((30, 120, 220, 255), WHITE),
        4: number_cloth(YELLOW, BLACK),
        5: number_cloth((40, 170, 60, 255), WHITE),
        6: NumberCloth(BLACK, YELLOW, LIGHT_GREY),
        7: number_cloth((255, 140, 0, 255), BLACK),
        8: number_cloth((255, 120, 200, 255), BLACK),
        9: number_cloth((60, 210, 210, 255), (220, 40, 40, 255)),
        10: number_cloth((130, 50, 180, 255), WHITE),
        11: number_cloth((150, 150, 150, 255), (220, 40, 40, 255)),
        12: number_cloth((170, 230, 40, 255), BLACK),
        13: number_cloth((140, 80, 40, 255), WHITE),
        14: number_cloth((150, 20, 50, 255), YELLOW),
    },
}

LINE_RGBA = (255, 40, 40, 255)
UNNUMBERED_RACER_RGBA = (255, 230, 0, 255)
HEADER_RGBA = BLACK
TEXT_RGBA = WHITE
TAG_OUTLINE_RGBA = (128, 128, 128, 255)
LINE_THICKNESS = 2
RACER_THICKNESS = 2
NUMBER_TAG_HEIGHT = 26
ONE_DIGIT_TAG_WIDTH, TWO_DIGIT_TAG_WIDTH = 24, 40
NUMBER_TEXT_SCALE = 0.8
HEADER_HEIGHT = 92
CLOCK_LABEL_ORIGIN, CLOCK_ORIGIN = (16, 34), (16, 76)
CLOCK_LABEL_SCALE, CLOCK_SCALE = 0.8, 1.2
NUMBER_COLUMN_LEFT, WIDEST_NUMBER_COLUMN_STEP = 330, 120
NUMBER_COLUMN_TAG_TOP = 12
NUMBER_COLUMN_TIME_Y = 76
NUMBER_COLUMN_TIME_SCALE = 0.65
FINISH_ROW_HEIGHT = 30
FINISH_LIST_WIDTH = 230
FINISH_LIST_TEXT_SCALE = 0.8
TEXT_THICKNESS = 2


def number_tag_width(number):
    return ONE_DIGIT_TAG_WIDTH if number < 10 else TWO_DIGIT_TAG_WIDTH


def number_column_lefts(numbers, frame_width):
    if len(numbers) == 1:
        return {numbers[0]: NUMBER_COLUMN_LEFT}
    last_tag_room = frame_width - NUMBER_COLUMN_LEFT - number_tag_width(max(numbers))
    step = min(WIDEST_NUMBER_COLUMN_STEP, last_tag_room // (len(numbers) - 1))
    return {
        number: NUMBER_COLUMN_LEFT + index * step
        for index, number in enumerate(numbers)
    }


class RacerOverlay(InPlaceVideoTransform):
    __gstmetadata__ = (
        "Racer Overlay",
        "Filter/Effect/Video",
        "On frames carrying a race-window blob, draws each numbered racer box in "
        "its number cloth colour with a line up to its number in a header bar "
        "holding the race clock and the numbers of the field, and draws the "
        "finish line and finish times of a finish-line blob",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    FRAME_CAPABILITIES = CUDA_DRAW_FRAME_CAPABILITIES

    number_cloths = GObject.Property(
        type=str,
        default="",
        nick="Number Cloths",
        blurb="Number cloth colours of the racers, eight-rugs or horse",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self._frame = 0
        self._race_start_seconds = None
        self._field = []
        self._drawings = {}

    def _draw_number_tag(self, drawing, cloths, number, left, bottom):
        cloth = cloths[number]
        top_left = (left, bottom - NUMBER_TAG_HEIGHT)
        bottom_right = (left + number_tag_width(number), bottom)
        drawing.fill_rectangle(top_left, bottom_right, cloth.cloth_rgba)
        if cloth.cloth_rgba == HEADER_RGBA:
            drawing.draw_rectangle(top_left, bottom_right, TAG_OUTLINE_RGBA, 1)
        drawing.draw_text(
            str(number),
            (left + 5, bottom - 6),
            NUMBER_TEXT_SCALE,
            cloth.number_rgba,
            TEXT_THICKNESS,
        )

    def _draw_header(
        self, drawing, cloths, column_lefts, width, clock_seconds, finished
    ):
        drawing.fill_rectangle((0, 0), (width, HEADER_HEIGHT), HEADER_RGBA)
        drawing.draw_text(
            "Time", CLOCK_LABEL_ORIGIN, CLOCK_LABEL_SCALE, TEXT_RGBA, TEXT_THICKNESS
        )
        drawing.draw_text(
            f"{clock_seconds:5.2f}",
            CLOCK_ORIGIN,
            CLOCK_SCALE,
            TEXT_RGBA,
            TEXT_THICKNESS,
        )
        for number, left in column_lefts.items():
            self._draw_number_tag(
                drawing,
                cloths,
                number,
                left,
                NUMBER_COLUMN_TAG_TOP + NUMBER_TAG_HEIGHT,
            )
        for place, (number, race_seconds) in enumerate(finished, 1):
            drawing.draw_text(
                f"{race_seconds:.2f}({place})",
                (column_lefts[number] - 30, NUMBER_COLUMN_TIME_Y),
                NUMBER_COLUMN_TIME_SCALE,
                cloths[number].on_black_rgba,
                TEXT_THICKNESS,
            )

    def _draw_finish_list(self, drawing, cloths, height, finished):
        top = height - FINISH_ROW_HEIGHT * len(finished) - 20
        drawing.fill_rectangle((0, top), (FINISH_LIST_WIDTH, height), HEADER_RGBA)
        for index, (number, race_seconds) in enumerate(finished):
            y = top + FINISH_ROW_HEIGHT * (index + 1)
            drawing.draw_text(
                str(index + 1),
                (14, y),
                FINISH_LIST_TEXT_SCALE,
                TEXT_RGBA,
                TEXT_THICKNESS,
            )
            self._draw_number_tag(drawing, cloths, number, 44, y + 4)
            drawing.draw_text(
                f"{race_seconds:.3f}",
                (84, y),
                FINISH_LIST_TEXT_SCALE,
                TEXT_RGBA,
                TEXT_THICKNESS,
            )

    def _draw_racers(self, drawing, cloths, column_lefts, buf, numbers_by_track_id):
        meta = analytics.get_relation_meta(buf)
        if not meta:
            return
        for racer in analytics.read_objects(meta):
            # mp_racer_number_tracker leaves boxes with no number untracked
            if racer["track_id"] is None:
                continue
            top_left = (int(racer["x"]), int(racer["y"]))
            bottom_right = (
                int(racer["x"] + racer["w"]),
                int(racer["y"] + racer["h"]),
            )
            number = numbers_by_track_id.get(str(racer["track_id"]))
            if number is None:
                drawing.draw_rectangle(
                    top_left, bottom_right, UNNUMBERED_RACER_RGBA, RACER_THICKNESS
                )
                continue
            drawing.draw_rectangle(
                top_left, bottom_right, cloths[number].cloth_rgba, RACER_THICKNESS
            )
            column_centre = column_lefts[number] + number_tag_width(number) // 2
            drawing.draw_segments(
                [
                    (
                        ((top_left[0] + bottom_right[0]) // 2, top_left[1]),
                        (column_centre, HEADER_HEIGHT),
                    )
                ],
                cloths[number].on_black_rgba,
                LINE_THICKNESS,
            )

    def process_in_place(self, frame, fmt, buf):
        if self.number_cloths not in NUMBER_CLOTHS:
            raise ValueError(
                f"number-cloths {self.number_cloths!r} is not one of "
                f"{sorted(NUMBER_CLOTHS)}"
            )
        if self.framerate is None:
            raise ValueError("the stream has no fixed framerate to run the race clock")
        numerator, denominator = self.framerate
        frames_per_second = numerator / denominator
        cloths = NUMBER_CLOTHS[self.number_cloths]
        height, width = frame.shape[:2]
        blobs = frameio.read_blobs(buf)
        window_blob = blobs.get(RACE_WINDOW_BLOB_NAME)
        if window_blob is not None:
            self._race_start_seconds = json.loads(window_blob)["start_seconds"]
            racer_numbers = json.loads(blobs[RACER_NUMBERS_BLOB_NAME])
            self._field = racer_numbers["field"]
        column_lefts = number_column_lefts(self._field, width) if self._field else {}
        finish = json.loads(blobs.get(FINISH_LINE_BLOB_NAME, "{}"))
        first_crossing_seconds = finish.get("first_crossing_seconds")
        drawing = drawing_for(self._drawings, frame, fmt)
        with drawing.drawing_on(frame):
            if "line" in finish:
                far, near = finish["line"]
                drawing.draw_segments(
                    [(tuple(int(v) for v in far), tuple(int(v) for v in near))],
                    LINE_RGBA,
                    LINE_THICKNESS,
                )
            if window_blob is not None:
                self._draw_racers(
                    drawing,
                    cloths,
                    column_lefts,
                    buf,
                    racer_numbers["numbers_by_track_id"],
                )
            if window_blob is not None or first_crossing_seconds is not None:
                finished = sorted(
                    (
                        (int(number), seconds - self._race_start_seconds)
                        for number, seconds in finish.get(
                            "crossing_seconds_by_racer_number", {}
                        ).items()
                    ),
                    key=lambda item: item[1],
                )
                clock_seconds = (
                    self._frame / frames_per_second - self._race_start_seconds
                    if first_crossing_seconds is None
                    else first_crossing_seconds - self._race_start_seconds
                )
                self._draw_header(
                    drawing, cloths, column_lefts, width, clock_seconds, finished
                )
                if finished:
                    self._draw_finish_list(drawing, cloths, height, finished)
        self._frame += 1


if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_racer_overlay", RacerOverlay
    )
