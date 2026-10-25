# BaseballOverlay
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import re

import backend
from backend import (
    analytics,
    GObject,
    InPlaceVideoTransform,
    CUDA_DRAW_FRAME_CAPABILITIES,
)
from sport_overlay_drawing import drawing_for

# coco class names the stock yolo weights emit
PERSON_LABEL = "person"
BAT_LABEL = "baseball bat"
GLOVE_LABEL = "baseball glove"
BALL_LABEL = "sports ball"

BATTER_ROLE = "batter"
FIELDER_ROLE = "fielder"
PLAYER_ROLE = "player"

ROLE_RGBA = {
    BATTER_ROLE: (255, 80, 80, 255),
    FIELDER_ROLE: (80, 140, 255, 255),
    PLAYER_ROLE: (235, 235, 235, 255),
}
EQUIPMENT_RGBA = {
    BAT_LABEL: (255, 200, 0, 255),
    GLOVE_LABEL: (200, 120, 255, 255),
}
BALL_RGBA = (0, 230, 0, 255)
BLACK_RGBA = (0, 0, 0, 255)

ELLIPSE_START_ANGLE = -45
ELLIPSE_END_ANGLE = 235
ELLIPSE_HEIGHT_RATIO = 0.35
ELLIPSE_THICKNESS = 2
EQUIPMENT_THICKNESS = 2
BALL_RING_THICKNESS = 2
BALL_MIN_RADIUS = 6
TRAIL_THICKNESS = 2
LABEL_SCALE = 0.5
LABEL_THICKNESS = 1
LABEL_RISE = 6
ID_BADGE_WIDTH = 40
ID_BADGE_HEIGHT = 18
ID_BADGE_DROP = 15

# a hit ball crosses under a tenth of the frame width per frame
BALL_TRAIL_MAX_STEP = 0.1
# a ball hidden behind the batter comes back within this many frames
BALL_TRAIL_MAX_GAP = 15
# a person holding a bat or glove overlaps its box by at least this much
EQUIPMENT_OVERLAP = 0.3
# a second box on the same object sits mostly inside the first
NESTED_OVERLAP = 0.7
# a ball this far inside a bat box or the batter's box is the knob or a fist
BALL_ON_BATTER = 0.5
# two bat boxes closer than this fraction of their union's diagonal are one bat
BAT_PART_GAP = 0.25


def parse_label(full_label):
    match = re.match(r"(.+)_id_(\d+)$", full_label)
    if match:
        return match.group(1), int(match.group(2))
    return full_label, None


def box_overlap(box, other):
    left = max(box[0], other[0])
    top = max(box[1], other[1])
    right = min(box[2], other[2])
    bottom = min(box[3], other[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    smaller = min(
        (box[2] - box[0]) * (box[3] - box[1]),
        (other[2] - other[0]) * (other[3] - other[1]),
    )
    return intersection / smaller if smaller > 0 else 0.0


def box_union(box, other):
    return (
        min(box[0], other[0]),
        min(box[1], other[1]),
        max(box[2], other[2]),
        max(box[3], other[3]),
    )


def same_bat(box, other):
    gap_x = max(0, max(box[0], other[0]) - min(box[2], other[2]))
    gap_y = max(0, max(box[1], other[1]) - min(box[3], other[3]))
    union = box_union(box, other)
    diagonal = ((union[2] - union[0]) ** 2 + (union[3] - union[1]) ** 2) ** 0.5
    return (gap_x**2 + gap_y**2) ** 0.5 < BAT_PART_GAP * diagonal


def merge_bat_parts(entries):
    merged = [
        dict(entry) for entry in sorted(entries, key=lambda entry: -entry["confidence"])
    ]
    index = 0
    while index < len(merged):
        kept = merged[index]
        parts = [
            other
            for other in merged[index + 1 :]
            if same_bat(kept["box"], other["box"])
        ]
        if not parts:
            index += 1
            continue
        for other in parts:
            kept["box"] = box_union(kept["box"], other["box"])
            merged.remove(other)
    return merged


def drop_nested(entries):
    kept = []
    for entry in sorted(entries, key=lambda entry: -entry["confidence"]):
        nested = any(
            box_overlap(entry["box"], other["box"]) >= NESTED_OVERLAP for other in kept
        )
        if not nested:
            kept.append(entry)
    return kept


class BaseballOverlay(InPlaceVideoTransform):
    __gstmetadata__ = (
        "Baseball Overlay",
        "Filter/Effect/Video",
        "Draws batter, fielder, bat, glove and ball markers with a ball trail "
        "from GstAnalytics detections of the coco baseball classes",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    FRAME_CAPABILITIES = CUDA_DRAW_FRAME_CAPABILITIES

    show_labels = GObject.Property(
        type=bool,
        default=True,
        nick="Show Labels",
        blurb="Write the role above each person and the class above bat and glove",
        flags=GObject.ParamFlags.READWRITE,
    )
    show_ids = GObject.Property(
        type=bool,
        default=False,
        nick="Show Track IDs",
        blurb="Draw the tracker id in a badge under each person",
        flags=GObject.ParamFlags.READWRITE,
    )
    show_equipment = GObject.Property(
        type=bool,
        default=True,
        nick="Show Equipment",
        blurb="Outline the bat and the glove",
        flags=GObject.ParamFlags.READWRITE,
    )
    ball_trail = GObject.Property(
        type=bool,
        default=True,
        nick="Ball Trail",
        blurb="Draw the path of the ball over the last trail-length sightings",
        flags=GObject.ParamFlags.READWRITE,
    )
    trail_length = GObject.Property(
        type=int,
        default=30,
        minimum=2,
        maximum=600,
        nick="Trail Length",
        blurb="Ball sightings kept in the trail",
        flags=GObject.ParamFlags.READWRITE,
    )
    min_confidence = GObject.Property(
        type=float,
        default=0.0,
        minimum=0.0,
        maximum=1.0,
        nick="Min Confidence",
        blurb="Detections under this score are not drawn",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self._frame = 0
        self._role_votes = {}
        self._trail = []
        self._trail_frame = None
        self._drawings = {}

    def _read_metadata(self, buf):
        entries = []
        meta = analytics.get_relation_meta(buf)
        if not meta:
            return entries
        for obj in analytics.read_objects(meta):
            label, track_id = parse_label(obj["label"])
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            entries.append(
                {
                    "label": label.lower(),
                    "track_id": track_id,
                    "confidence": obj["score"],
                    "box": (x, y, x + w, y + h),
                }
            )
        return entries

    def _role(self, person, bats, gloves):
        held_bat = any(
            box_overlap(person["box"], bat["box"]) >= EQUIPMENT_OVERLAP for bat in bats
        )
        held_glove = any(
            box_overlap(person["box"], glove["box"]) >= EQUIPMENT_OVERLAP
            for glove in gloves
        )
        if person["track_id"] is None:
            if held_bat:
                return BATTER_ROLE
            if held_glove:
                return FIELDER_ROLE
            return PLAYER_ROLE
        votes = self._role_votes.setdefault(
            person["track_id"], {BATTER_ROLE: 0, FIELDER_ROLE: 0}
        )
        if held_bat:
            votes[BATTER_ROLE] += 1
        if held_glove:
            votes[FIELDER_ROLE] += 1
        best = max(votes, key=votes.get)
        return best if votes[best] > 0 else PLAYER_ROLE

    def _drop_stale_trail(self):
        if self._trail and self._frame - self._trail_frame > BALL_TRAIL_MAX_GAP:
            self._trail = []

    def _extend_trail(self, point):
        if self._trail:
            elapsed = self._frame - self._trail_frame
            jump = (
                (point[0] - self._trail[-1][0]) ** 2
                + (point[1] - self._trail[-1][1]) ** 2
            ) ** 0.5
            if jump > BALL_TRAIL_MAX_STEP * self.width * elapsed:
                self._trail = []
        self._trail.append(point)
        del self._trail[: -self.trail_length]
        self._trail_frame = self._frame

    def _draw_ellipse(self, drawing, box, rgba, track_id):
        x1, y1, x2, y2 = box
        x_center = int((x1 + x2) / 2)
        y_bottom = int(y2)
        width = max(1, int(x2 - x1))
        drawing.draw_ellipse_arc(
            (x_center, y_bottom),
            (width, max(1, int(ELLIPSE_HEIGHT_RATIO * width))),
            ELLIPSE_START_ANGLE,
            ELLIPSE_END_ANGLE,
            rgba,
            ELLIPSE_THICKNESS,
        )
        if self.show_ids and track_id is not None:
            left = x_center - ID_BADGE_WIDTH // 2
            top = y_bottom - ID_BADGE_HEIGHT // 2 + ID_BADGE_DROP
            drawing.fill_rectangle(
                (left, top), (left + ID_BADGE_WIDTH, top + ID_BADGE_HEIGHT), rgba
            )
            drawing.draw_text(
                str(track_id), (left + 12, top + 14), LABEL_SCALE, BLACK_RGBA, 2
            )

    def _draw_label(self, drawing, box, text, rgba):
        x1, y1, _, _ = box
        drawing.draw_text(
            text,
            (int(x1), max(12, int(y1) - LABEL_RISE)),
            LABEL_SCALE,
            rgba,
            LABEL_THICKNESS,
        )

    def _draw_ball(self, drawing, box):
        x1, y1, x2, y2 = box
        centre = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        radius = max(BALL_MIN_RADIUS, int((x2 - x1) / 2))
        drawing.draw_ellipse_arc(
            centre, (radius, radius), 0, 360, BALL_RGBA, BALL_RING_THICKNESS
        )

    def process_in_place(self, frame, fmt, buf):
        self._frame += 1
        self._drop_stale_trail()
        entries = [
            entry
            for entry in self._read_metadata(buf)
            if entry["confidence"] >= self.min_confidence
        ]
        if not entries:
            return
        # the tracker's boxes carry an id, the detector's do not
        tracked = [entry for entry in entries if entry["track_id"] is not None]
        detected = [entry for entry in entries if entry["track_id"] is None]
        people = drop_nested(
            [entry for entry in (tracked or detected) if entry["label"] == PERSON_LABEL]
        )
        # the detector often boxes the handle and the barrel of one bat separately
        bats = merge_bat_parts(
            [entry for entry in detected if entry["label"] == BAT_LABEL]
        )
        gloves = drop_nested(
            [entry for entry in detected if entry["label"] == GLOVE_LABEL]
        )
        roles = [(person, self._role(person, bats, gloves)) for person in people]
        batter_boxes = [
            person["box"] for person, role in roles if role == BATTER_ROLE
        ] + [bat["box"] for bat in bats]
        balls = [
            entry
            for entry in detected
            if entry["label"] == BALL_LABEL
            and not any(
                box_overlap(entry["box"], box) >= BALL_ON_BATTER for box in batter_boxes
            )
        ]
        ball = max(balls, key=lambda entry: entry["confidence"], default=None)
        if ball is not None:
            x1, y1, x2, y2 = ball["box"]
            self._extend_trail((int((x1 + x2) / 2), int((y1 + y2) / 2)))

        drawing = drawing_for(self._drawings, frame, fmt)
        with drawing.drawing_on(frame):
            for person, role in roles:
                rgba = ROLE_RGBA[role]
                self._draw_ellipse(drawing, person["box"], rgba, person["track_id"])
                if self.show_labels:
                    self._draw_label(drawing, person["box"], role, rgba)
            if self.show_equipment:
                for item in bats + gloves:
                    rgba = EQUIPMENT_RGBA[item["label"]]
                    x1, y1, x2, y2 = (int(value) for value in item["box"])
                    drawing.draw_rectangle(
                        (x1, y1), (x2, y2), rgba, EQUIPMENT_THICKNESS
                    )
                    if self.show_labels:
                        self._draw_label(drawing, item["box"], item["label"], rgba)
            if self.ball_trail and len(self._trail) >= 2:
                drawing.draw_polyline(self._trail, BALL_RGBA, TRAIL_THICKNESS)
            if ball is not None:
                self._draw_ball(drawing, ball["box"])


if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_baseball_overlay", BaseballOverlay
    )
