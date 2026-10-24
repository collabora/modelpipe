# FootballOverlay
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json
import os
import re

import backend
from backend import (
    analytics,
    frameio,
    GObject,
    InPlaceVideoTransform,
    CUDA_DRAW_FRAME_CAPABILITIES,
)
from sport_overlay_drawing import (
    OpenCVSportDrawing,
    channel_order,
    drawing_for,
)
from utils.blob_headers import ALERT_META_HEADER
from utils.kick_detector import KickDetector

_PALETTE = [
    (239, 71, 111, 255),
    (255, 209, 102, 255),
    (6, 214, 160, 255),
    (17, 138, 178, 255),
    (255, 107, 107, 255),
    (78, 205, 196, 255),
    (199, 125, 255, 255),
    (255, 159, 28, 255),
    (46, 196, 182, 255),
    (118, 200, 247, 255),
]

_REFEREE_RGBA = (255, 215, 0, 255)
_BALL_RGBA = (0, 230, 0, 255)
# a hard shot moves under half of this per frame
_BALL_TRAIL_MAX_STEP = 0.04
# a ball hidden behind a player comes back within this many steps
_BALL_TRAIL_MAX_STEPS_PER_GAP = 4
_BALL_TRAIL_MAX_GAP = 25
_CAMERA_SHIFT_MIN_TRACKS = 3
_PLAYER_RGBA = (0, 200, 255, 255)
_RED_TEAM_RGBA = (255, 40, 40, 255)
_BLUE_TEAM_RGBA = (40, 90, 255, 255)
_DEFAULT_RGBA = (235, 235, 235, 255)
_BLACK_RGBA = (0, 0, 0, 255)
_HUD_BG_RGBA = (92, 41, 131, 255)
_HUD_TEXT_RGBA = (64, 186, 47, 255)
_HIGHLIGHT_RGBA = (255, 255, 255, 255)
_KICK_RGBA = (255, 255, 0, 255)
_KICK_SPEED = 0.8
_KICK_WINDOW_FRAMES = 10
_KICK_COOLDOWN_FRAMES = 25
_KICK_FLASH_FRAMES = 50
_KICK_TEXT_SCALE = 0.9
_KICK_TEXT_THICKNESS = 2
_KICK_TEXT_RISE = 12
_HUD_LEFT = 10
_HUD_TOP = 10
_HUD_BORDER_THICKNESS = 2
# a thick opencv border reaches this far outside its rectangle
_HUD_BORDER_OVERHANG = _HUD_BORDER_THICKNESS // 2
_RGBA_CHANNELS = 4
_ELLIPSE_START_ANGLE = -45
_ELLIPSE_END_ANGLE = 235
_ELLIPSE_HEIGHT_RATIO = 0.35


def _is_ball(label):
    return "ball" in label


def _is_referee(label):
    return "referee" in label or label == "ref"


class FootballOverlay(InPlaceVideoTransform):
    """
    Metadata-driven broadcast overlay (football_analysis style), streaming.

    Reads upstream GstAnalytics detection/tracking metadata and draws: an
    ellipse + optional id badge per subject, a gold ellipse for referees, a
    green triangle on the ball, fading motion trails, and a focal-player HUD
    with a headshot, accumulated ball contacts, and distance travelled.
    """

    __gstmetadata__ = (
        "Football Overlay",
        "Filter/Effect/Video",
        "Broadcast-style detection/tracking overlay (ellipses, ball triangle, "
        "trails, headshot HUD with ball contacts + distance) from GstAnalytics",
        "Marcus Edel <marcus.edel@collabora.com>",
    )

    FRAME_CAPABILITIES = CUDA_DRAW_FRAME_CAPABILITIES

    show_labels = GObject.Property(
        type=bool,
        default=True,
        nick="Show Labels",
        blurb="Draw the class name above each object",
        flags=GObject.ParamFlags.READWRITE,
    )
    show_ids = GObject.Property(
        type=bool,
        default=True,
        nick="Show Track IDs",
        blurb="Draw the track-id badge under each tracked object",
        flags=GObject.ParamFlags.READWRITE,
    )
    trails = GObject.Property(
        type=bool,
        default=True,
        nick="Show Trails",
        blurb="Draw a fading motion trail behind each tracked player",
        flags=GObject.ParamFlags.READWRITE,
    )
    ball_trail = GObject.Property(
        type=bool,
        default=False,
        nick="Show Ball Trail",
        blurb="Draw a fading motion trail behind the ball",
        flags=GObject.ParamFlags.READWRITE,
    )
    trail_length = GObject.Property(
        type=int,
        default=30,
        minimum=2,
        maximum=300,
        nick="Trail Length",
        blurb="Number of recent positions kept in each motion trail",
        flags=GObject.ParamFlags.READWRITE,
    )
    show_ball = GObject.Property(
        type=bool,
        default=False,
        nick="Show Ball",
        blurb="Draw the marker on the ball (the ball is still tracked for "
        "contact counting either way)",
        flags=GObject.ParamFlags.READWRITE,
    )
    show_hud = GObject.Property(
        type=bool,
        default=True,
        nick="Show HUD",
        blurb="Draw the focal-player HUD (headshot, label, contacts, distance)",
        flags=GObject.ParamFlags.READWRITE,
    )
    headshot_path = GObject.Property(
        type=str,
        default="data/Chinedu-Obasi_2684938.jpg",
        nick="Headshot Path",
        blurb="Image shown in the HUD (empty to disable)",
        flags=GObject.ParamFlags.READWRITE,
    )
    headshot_size = GObject.Property(
        type=int,
        default=90,
        minimum=16,
        maximum=512,
        nick="Headshot Size",
        blurb="Headshot square size in pixels",
        flags=GObject.ParamFlags.READWRITE,
    )
    player_label = GObject.Property(
        type=str,
        default="Player #8",
        nick="Player Label",
        blurb="Static label drawn in the HUD",
        flags=GObject.ParamFlags.READWRITE,
    )
    contact_pad_ratio = GObject.Property(
        type=float,
        default=0.25,
        minimum=0.0,
        maximum=5.0,
        nick="Contact Pad Ratio",
        blurb="Ball counts as a contact within this fraction of the player box size",
        flags=GObject.ParamFlags.READWRITE,
    )
    contact_gap_frames = GObject.Property(
        type=int,
        default=5,
        minimum=0,
        maximum=1000,
        nick="Contact Gap Frames",
        blurb="Min frames between counted contacts for the same player",
        flags=GObject.ParamFlags.READWRITE,
    )
    player_height = GObject.Property(
        type=float,
        default=1.8,
        minimum=0.1,
        maximum=10.0,
        nick="Player Height (m)",
        blurb="Assumed real-world height used to convert pixels to metres",
        flags=GObject.ParamFlags.READWRITE,
    )
    min_confidence = GObject.Property(
        type=float,
        default=0.0,
        minimum=0.0,
        maximum=1.0,
        nick="Min Confidence",
        blurb="Skip detections whose confidence is below this threshold",
        flags=GObject.ParamFlags.READWRITE,
    )
    class_names = GObject.Property(
        type=str,
        default="",
        nick="Class Names",
        blurb="Comma-separated names to map numeric labels (label_N) from the "
        "onnx/objectdetector path, e.g. 'ball,goalkeeper,player,referee'",
        flags=GObject.ParamFlags.READWRITE,
    )
    team_colors = GObject.Property(
        type=bool,
        default=True,
        nick="Team Colors",
        blurb="Colour players by jersey team (red/blue, per-track majority vote); "
        "off draws all players one colour",
        flags=GObject.ParamFlags.READWRITE,
    )
    draw_from_detections = GObject.Property(
        type=bool,
        default=False,
        nick="Draw From Detections",
        blurb="Draw ellipses on the raw per-frame detection boxes instead of the "
        "tracker's boxes -- no Kalman drift, coasted phantoms or track-split "
        "doubles. Team colour is then classified per frame; the HUD still uses "
        "tracker metadata if present",
        flags=GObject.ParamFlags.READWRITE,
    )
    merge_iou = GObject.Property(
        type=float,
        default=0.5,
        minimum=0.0,
        maximum=1.0,
        nick="Merge IoU",
        blurb="Collapse overlapping boxes (across classes) into one before "
        "drawing, so one player isn't circled twice; a box is merged when its "
        "IoU or containment with a kept box exceeds this (0 disables)",
        flags=GObject.ParamFlags.READWRITE,
    )
    position_smoothing = GObject.Property(
        type=float,
        default=0.5,
        minimum=0.0,
        maximum=0.95,
        nick="Position Smoothing",
        blurb="Temporal EMA on drawn box positions (0=off, higher=smoother but "
        "more lag). Boxes are associated frame-to-frame by proximity, so this "
        "damps detection jitter and the steps from a detection interval > 1",
        flags=GObject.ParamFlags.READWRITE,
    )
    highlight_focal = GObject.Property(
        type=bool,
        default=True,
        nick="Highlight Focal Player",
        blurb="Mark the focal player (the one shown in the HUD) on the pitch "
        "with a chevron above their head and a bolder ellipse",
        flags=GObject.ParamFlags.READWRITE,
    )
    focal_track_id = GObject.Property(
        type=int,
        default=-1,
        minimum=-1,
        maximum=100000,
        nick="Focal Track ID",
        blurb="Pin the focal/highlighted player to this track id; -1 = auto "
        "(the player tracked the most, with hysteresis so it stays stable)",
        flags=GObject.ParamFlags.READWRITE,
    )
    detect_kicks = GObject.Property(
        type=bool,
        default=False,
        nick="Detect Kicks",
        blurb="Mark each kick on the picture and attach a GST-ALERT blob to its "
        "frame, so a mp_alertrecorder downstream records it",
        flags=GObject.ParamFlags.READWRITE,
    )

    @GObject.Property(
        type=float,
        default=_KICK_SPEED,
        minimum=0.0,
        maximum=100.0,
        nick="Kick Speed",
        blurb="Ball speed, in ball sizes per frame, the ball must pass after a "
        "player contact to count as a kick",
        flags=GObject.ParamFlags.READWRITE,
    )
    def kick_speed(self):
        return self._kick_detector.speed_threshold

    @kick_speed.setter
    def kick_speed(self, value):
        self._kick_detector.speed_threshold = value

    @GObject.Property(
        type=int,
        default=0,
        minimum=0,
        nick="Kick Count",
        blurb="Kicks detected so far",
        flags=GObject.ParamFlags.READABLE,
    )
    def kick_count(self):
        return self._kick_count

    @GObject.Property(
        type=str,
        default="",
        nick="Last Kick",
        blurb="JSON of the last kick (frame, player, speed, direction in degrees "
        "anticlockwise from screen right), empty before the first",
        flags=GObject.ParamFlags.READABLE,
    )
    def last_kick(self):
        return self._last_kick

    def __init__(self):
        super().__init__()
        self._format_seen = None
        self._order = channel_order("RGBA")
        # per-track state, accumulated across frames
        self._trail = {}
        self._ball_trail = []
        self._ball_trail_frame = None
        self._ball_trail_pending = None
        self._last_pt = {}
        self._last_centre = {}
        self._distance_px = {}
        self._heights = []
        self._widths = []
        self._ell_w = {}  # track_id -> smoothed ellipse half-width (px)
        self._contacts = {}
        self._last_contact_frame = {}
        self._frames_seen = {}
        self._track_label = {}
        self._class_votes = {}  # track_id -> {label: count}, for stable class
        self._team_votes = {}  # track_id -> {"red": n, "blue": n}, jersey team
        self._frame = 0
        self._focal = None  # current focal track id (sticky, for hysteresis)
        self._headshot = None
        self._headshot_loaded = False
        self._hud_bitmap = None
        self._hud_bitmap_key = None
        self._drawings = {}
        # Position-smoothing slots: {"box": np[x1,y1,x2,y2]} kept across frames
        # and matched by proximity, so the drawn ellipse can be low-passed.
        self._smooth_slots = []
        self._kick_detector = KickDetector(
            _KICK_SPEED, _KICK_WINDOW_FRAMES, _KICK_COOLDOWN_FRAMES
        )
        self._kick_count = 0
        self._last_kick = ""
        self._ball_box = None
        self._kick_flash_text = ""
        self._kick_flash_until = 0

    def _use_format(self, fmt):
        self._order = channel_order(fmt)
        self._headshot_loaded = False  # re-load in the new channel order
        self._format_seen = fmt

    def _map_label(self, label):
        if self.class_names:
            m = re.match(r"label_(\d+)$", label)
            if m:
                names = [s.strip() for s in self.class_names.split(",") if s.strip()]
                i = int(m.group(1))
                if 0 <= i < len(names):
                    return names[i]
        return label

    def _parse_label(self, full_label):
        core = full_label
        m = re.match(r"stream_\d+_(.*)$", full_label)
        if m:
            core = m.group(1)
        m = re.match(r"(.+)_id_(\d+)$", core)
        if m:
            return self._map_label(m.group(1)), int(m.group(2))
        m = re.match(r"id_(\d+)$", core)
        if m:
            return "object", int(m.group(1))
        return self._map_label(core or "object"), None

    def _read_metadata(self, buf):
        entries = []
        meta = analytics.get_relation_meta(buf)
        if not meta:
            return entries
        for obj in analytics.read_objects(meta):
            label, track_id = self._parse_label(obj["label"])
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

    @staticmethod
    def _point_to_bbox_distance(px, py, box):
        x1, y1, x2, y2 = box
        dx = max(x1 - px, 0.0, px - x2)
        dy = max(y1 - py, 0.0, py - y2)
        return (dx * dx + dy * dy) ** 0.5

    def _ball_contact(self, players, ball_box):
        """Closest player to the ball, if within contact_pad_ratio of its size."""
        bx = (ball_box[0] + ball_box[2]) / 2.0
        by = (ball_box[1] + ball_box[3]) / 2.0
        best_tid, best_d, best_box = None, float("inf"), None
        for tid, box in players.items():
            d = self._point_to_bbox_distance(bx, by, box)
            if d < best_d:
                best_tid, best_d, best_box = tid, d, box
        if best_box is None:
            return None
        w = best_box[2] - best_box[0]
        h = best_box[3] - best_box[1]
        if best_d > self.contact_pad_ratio * max(w, h):
            return None
        return best_tid

    def _update_tracks(self, entries, det_ball_box=None):
        self._frame += 1
        active = set()
        players = {}
        ball_box = None
        # Accumulate per-track class votes first so the stable label below
        # already reflects this frame.
        for e in entries:
            tid = e["track_id"]
            if tid is None:
                continue
            v = self._class_votes.setdefault(tid, {})
            v[e["label"]] = v.get(e["label"], 0) + 1
        self._follow_camera(entries)
        smoothing = float(self.position_smoothing)
        for e in entries:
            tid = e["track_id"]
            if tid is None:
                continue
            label = self._stable_label(tid, e["label"])
            if _is_ball(label):
                ball_box = e["box"]
                continue
            active.add(tid)
            players[tid] = e["box"]
            self._track_label[tid] = label
            self._frames_seen[tid] = self._frames_seen.get(tid, 0) + 1
            x1, y1, x2, y2 = e["box"]
            self._last_centre[tid] = ((x1 + x2) / 2, (y1 + y2) / 2)
            foot = ((x1 + x2) / 2, y2)
            prev = self._last_pt.get(tid)
            if prev is not None:
                foot = (
                    smoothing * prev[0] + (1 - smoothing) * foot[0],
                    smoothing * prev[1] + (1 - smoothing) * foot[1],
                )
            if y2 - y1 > 0:
                self._heights.append(y2 - y1)
                if len(self._heights) > 600:
                    self._heights = self._heights[-600:]
            self._update_ellipse_width(tid, x2 - x1)
            if prev is not None:
                self._distance_px[tid] = (
                    self._distance_px.get(tid, 0.0)
                    + ((foot[0] - prev[0]) ** 2 + (foot[1] - prev[1]) ** 2) ** 0.5
                )
            self._last_pt[tid] = foot
            trail = self._trail.setdefault(tid, [])
            trail.append(foot)
            if len(trail) > self.trail_length:
                del trail[: -self.trail_length]

        # Fall back to the detected ball if no tracked ball this frame.
        if ball_box is None:
            ball_box = det_ball_box

        if ball_box is not None:
            x1, y1, x2, y2 = ball_box
            self._extend_ball_trail((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        elif (
            self._ball_trail
            and self._frame - self._ball_trail_frame > _BALL_TRAIL_MAX_GAP
        ):
            self._ball_trail = []

        # Ball contacts (debounced per player), like football_analyzer.
        contact_track_id = None
        if ball_box is not None and players:
            tid = self._ball_contact(players, ball_box)
            contact_track_id = tid
            if tid is not None:
                last = self._last_contact_frame.get(tid)
                if last is None or (self._frame - last) > self.contact_gap_frames:
                    self._contacts[tid] = self._contacts.get(tid, 0) + 1
                self._last_contact_frame[tid] = self._frame

        for tid in list(self._trail.keys()):
            if tid not in active:
                del self._trail[tid]
                self._last_pt.pop(tid, None)
                self._last_centre.pop(tid, None)
                self._ell_w.pop(tid, None)
        if ball_box is not None:
            self._ball_box = ball_box
        kick = None
        if self.detect_kicks:
            kicker = self._kicker(contact_track_id)
            kick = self._kick_detector.update(self._frame, ball_box, kicker)
        return active, kick

    def _kicker(self, contact_track_id):
        if contact_track_id is None:
            return None
        label = self._track_label.get(contact_track_id, "")
        if self._is_referee_track(contact_track_id, label):
            return None
        return contact_track_id

    def _record_kick(self, buf, kick):
        self._kick_count += 1
        self._last_kick = json.dumps(kick._asdict())
        self._kick_flash_text = f"KICK #{kick.player}"
        self._kick_flash_until = self._frame + _KICK_FLASH_FRAMES
        payload = [
            {
                "event": "kick",
                "player": kick.player,
                "frame": kick.frame,
                "speed": kick.speed,
            }
        ]
        frameio.append_blob(buf, ALERT_META_HEADER, json.dumps(payload).encode())
        self.logger.info(f"kick {self._last_kick}")

    def _draw_kick_flash(self, drawing):
        if self._frame > self._kick_flash_until or self._ball_box is None:
            return
        x1, y1, _, _ = self._ball_box
        drawing.draw_text(
            self._kick_flash_text,
            (int(x1), max(_KICK_TEXT_RISE, int(y1) - _KICK_TEXT_RISE)),
            _KICK_TEXT_SCALE,
            _KICK_RGBA,
            _KICK_TEXT_THICKNESS,
        )

    def _follow_camera(self, entries):
        # a pan moves the pitch under the trails
        moves = []
        for e in entries:
            prev = self._last_centre.get(e["track_id"])
            if prev is None or _is_ball(e["label"]):
                continue
            x1, y1, x2, y2 = e["box"]
            moves.append(((x1 + x2) / 2 - prev[0], (y1 + y2) / 2 - prev[1]))
        if len(moves) < _CAMERA_SHIFT_MIN_TRACKS:
            return
        dx = sorted(m[0] for m in moves)[len(moves) // 2]
        dy = sorted(m[1] for m in moves)[len(moves) // 2]
        for trail in self._trail.values():
            trail[:] = [(x + dx, y + dy) for x, y in trail]
        self._ball_trail = [(x + dx, y + dy) for x, y in self._ball_trail]
        if self._ball_trail_pending is not None:
            (x, y), frame = self._ball_trail_pending
            self._ball_trail_pending = ((x + dx, y + dy), frame)
        for tid, (x, y) in self._last_pt.items():
            self._last_pt[tid] = (x + dx, y + dy)

    def _update_ellipse_width(self, track_id, raw_w):
        # Smooth (and outlier-reject) the per-track ellipse width so a single
        # oversized box -- two players merged, or a drifting keep-alive
        # prediction -- can't balloon the circle for one frame.
        if raw_w <= 0:
            return
        if self._widths:
            self._widths.append(raw_w)
            if len(self._widths) > 600:
                self._widths = self._widths[-600:]
            srt = sorted(self._widths)
            med = srt[len(srt) // 2]
            clamped = min(max(raw_w, 0.5 * med), 1.8 * med)
        else:
            self._widths.append(raw_w)
            clamped = raw_w
        prev = self._ell_w.get(track_id)
        # EMA: slow enough to keep the circle size steady frame-to-frame, fast
        # enough to still follow real perspective changes as players move.
        self._ell_w[track_id] = (
            clamped if prev is None else 0.25 * clamped + 0.75 * prev
        )

    def _px_per_meter(self):
        import numpy as np

        if not self._heights:
            return None
        return float(np.median(self._heights)) / max(0.1, self.player_height)

    def _focal_track(self):
        # Pin to an explicit track id if requested.
        if self.focal_track_id >= 0:
            return (
                self.focal_track_id
                if self.focal_track_id in self._frames_seen
                else self._focal
            )
        keys = set(self._frames_seen)
        if not keys:
            return None

        # Only consider *sustained* tracks. Otherwise a track that flickered for
        # a few frames -- common when detection/tracking churns -- can win on a
        # single ball contact and then show ~0 distance (it was barely tracked).
        # The floor scales with elapsed frames, with a small absolute minimum.
        floor = max(10, int(0.2 * self._frame))
        candidates = [t for t in keys if self._frames_seen.get(t, 0) >= floor] or list(
            keys
        )

        # Rank by ball contacts (the player most involved with the ball), with
        # frames-seen as a tiebreak / pre-contact fallback (before anyone has
        # touched the ball, the most-tracked player is shown).
        def score(t):
            return (self._contacts.get(t, 0), self._frames_seen.get(t, 0))

        best = max(candidates, key=score)
        # Stability: keep the current focal unless a challenger has *strictly
        # more* contacts, so the highlight/HUD don't flip on ties or noise.
        cur = self._focal
        if (
            cur is not None
            and cur in candidates
            and self._contacts.get(best, 0) <= self._contacts.get(cur, 0)
        ):
            best = cur
        self._focal = best
        return best

    def _stable_label(self, track_id, fallback=""):
        # Majority-voted class over the track's history — smooths frame-to-frame
        # misclassifications (e.g. a player briefly tagged 'referee'), so the
        # gold referee marking doesn't flicker.
        votes = self._class_votes.get(track_id)
        if not votes:
            return fallback
        return max(votes, key=votes.get)

    def _team_color(self, track_id):
        # Confident kit colour from a track's accumulated jersey votes, else
        # None. Red/blue -> team; "ref" (distinctive non-team kit) -> gold.
        # Requires a minimum number of votes AND a clear majority, so a few
        # noisy frames can't decide the colour.
        if track_id is None:
            return None
        c = self._team_votes.get(track_id)
        if not c:
            return None
        red, blue, ref = c.get("red", 0), c.get("blue", 0), c.get("ref", 0)
        total = red + blue + ref
        if total < 4:
            return None
        colors = {_RED_TEAM_RGBA: red, _BLUE_TEAM_RGBA: blue, _REFEREE_RGBA: ref}
        color, n = max(colors.items(), key=lambda kv: kv[1])
        return color if n >= 0.6 * total else None

    def _is_referee_track(self, track_id, fallback_label):
        # A track is a referee only if referee *clearly dominates* its class
        # votes. Referees are rare, so a mostly-player track with a few stray
        # 'referee' mislabels stays a player (won't get the gold circle).
        votes = self._class_votes.get(track_id) if track_id is not None else None
        if not votes:
            return _is_referee(fallback_label)
        total = sum(votes.values())
        ref = sum(c for lbl, c in votes.items() if _is_referee(lbl))
        return total > 0 and ref >= 3 and ref >= 0.6 * total

    def _color_for(self, label, track_id):
        # Colour by the track's *accumulated* jersey team (robust to per-frame
        # noise). Referee/player only decides the fallback when the team is
        # undecided: a referee keeps gold (stays visible), a player isn't drawn.
        if _is_ball(label):
            return _BALL_RGBA
        if self.team_colors:
            team = self._team_color(track_id)
            if team is not None:
                return team
            return _REFEREE_RGBA if self._is_referee_track(track_id, label) else None
        return _REFEREE_RGBA if _is_referee(label) else _PLAYER_RGBA

    @staticmethod
    def _overlap(a, b):
        # max(IoU, intersection-over-smaller-area): catches both heavy overlap
        # and a small duplicate box sitting inside a larger one.
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        iou = inter / union if union > 0.0 else 0.0
        smaller = min(area_a, area_b)
        contain = inter / smaller if smaller > 0.0 else 0.0
        return max(iou, contain)

    @staticmethod
    def _feet_close(a, b):
        # True when two boxes' foot points (bottom-centre, where the ellipse is
        # drawn) are within ~0.4 of the smaller box width. The ellipse is ~2x the
        # box width, so near-coincident feet = one player circled twice even when
        # the boxes' IoU is low. Genuinely adjacent players are ~a full width
        # apart at the feet, so they're not merged.
        fax, fay = (a[0] + a[2]) / 2.0, a[3]
        fbx, fby = (b[0] + b[2]) / 2.0, b[3]
        ref = max(1.0, min(a[2] - a[0], b[2] - b[0]))
        return ((fax - fbx) ** 2 + (fay - fby) ** 2) ** 0.5 < 0.4 * ref

    def _merge_overlaps(self, entries):
        # Class-agnostic greedy suppression: keep the most confident box, drop
        # any later box that overlaps it past merge_iou OR sits at the same feet.
        # Collapses a player circled twice (e.g. player+goalkeeper on one person,
        # or two offset boxes) into one. Only the most confident ball is kept.
        if self.merge_iou <= 0.0 or len(entries) < 2:
            return entries
        ordered = sorted(entries, key=lambda e: e["confidence"], reverse=True)
        kept = []
        ball_kept = False
        for e in ordered:
            if _is_ball(e["label"]):
                if not ball_kept:
                    kept.append(e)
                ball_kept = True
                continue
            if any(
                not _is_ball(k["label"])
                and (
                    self._overlap(e["box"], k["box"]) >= self.merge_iou
                    or self._feet_close(e["box"], k["box"])
                )
                for k in kept
            ):
                continue
            kept.append(e)
        return kept

    def _assign_track_ids(self, draw_entries, track_entries):
        # Give each drawn box a stable track id: track-mode boxes already carry
        # one; detection-mode boxes borrow the id of the best-overlapping track
        # (greedy, each track used once) so detection circles can use the
        # tracker's persistent id for the badge and the accumulated colour.
        ids = [e["track_id"] for e in draw_entries]
        if not track_entries:
            return ids
        pairs = []
        for di, e in enumerate(draw_entries):
            if e["track_id"] is not None or _is_ball(e["label"]):
                continue
            for t in track_entries:
                if _is_ball(t["label"]):
                    continue
                ov = self._overlap(e["box"], t["box"])
                if ov >= 0.3:
                    pairs.append((ov, di, t["track_id"]))
        pairs.sort(key=lambda p: p[0], reverse=True)
        used_draw, used_track = set(), set()
        for _ov, di, tid in pairs:
            if di in used_draw or tid in used_track:
                continue
            ids[di] = tid
            used_draw.add(di)
            used_track.add(tid)
        return ids

    def _smooth_boxes(self, np, entries):
        # Temporal EMA on the boxes we're about to draw. Each box is matched to
        # the nearest slot from last frame (by centre, within a size-relative
        # gate) and pulled toward the new detection; slots not matched this
        # frame are dropped (no phantoms). Damps jitter and interval steps. The
        # ball is passed through unsmoothed so it never lags.
        import numpy as np

        a = float(self.position_smoothing)
        if a <= 0.0 or not entries:
            return entries
        used = set()
        out = []
        for e in entries:
            if _is_ball(e["label"]):
                out.append(e)
                continue
            box = np.array(e["box"], dtype=np.float64)
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            # Generous gate so a coherent interval-step jump still associates
            # (and glides) without grabbing a different nearby player.
            gate = 1.5 * max(box[2] - box[0], box[3] - box[1], 1.0)
            best, best_d = None, gate
            for idx, slot in enumerate(self._smooth_slots):
                if idx in used:
                    continue
                sb = slot["box"]
                d = (
                    ((sb[0] + sb[2]) / 2.0 - cx) ** 2
                    + ((sb[1] + sb[3]) / 2.0 - cy) ** 2
                ) ** 0.5
                if d < best_d:
                    best, best_d = idx, d
            if best is None:
                self._smooth_slots.append({"box": box.copy()})
                used.add(len(self._smooth_slots) - 1)
                smoothed = box
            else:
                used.add(best)
                slot = self._smooth_slots[best]
                slot["box"] = a * slot["box"] + (1.0 - a) * box
                smoothed = slot["box"]
            ne = dict(e)
            ne["box"] = (
                float(smoothed[0]),
                float(smoothed[1]),
                float(smoothed[2]),
                float(smoothed[3]),
            )
            out.append(ne)
        self._smooth_slots = [s for i, s in enumerate(self._smooth_slots) if i in used]
        return out

    def _detection_color(self, drawing, label, box):
        # Colour a raw detection box (no track id) by its jersey team, classified
        # from this frame -- referees included. When the jersey isn't clearly a
        # team colour, a referee falls back to gold (so real refs stay visible)
        # and a player isn't drawn (matching the track-mode behaviour).
        ref = _is_referee(label)
        if not self.team_colors:
            return _REFEREE_RGBA if ref else _PLAYER_RGBA
        vote = drawing.classify_jersey(box)
        if vote == "red":
            return _RED_TEAM_RGBA
        if vote == "blue":
            return _BLUE_TEAM_RGBA
        if vote == "ref":
            return _REFEREE_RGBA
        return _REFEREE_RGBA if ref else None

    def _load_headshot(self, cv2, np):
        import cv2
        import numpy as np

        if self._headshot_loaded:
            return self._headshot
        self._headshot_loaded = True
        self._headshot = None
        path = self.headshot_path
        if not path or not os.path.exists(path):
            if path:
                self.logger.warning(f"headshot not found: {path}")
            return None
        img = cv2.imread(path)  # BGR
        if img is None:
            return None
        sz = int(self.headshot_size)
        img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_AREA)
        rgb = img[:, :, ::-1]  # BGR -> RGB
        alpha = np.full((sz, sz, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, alpha], axis=2).astype(np.uint8)  # logical RGBA

        self._headshot = np.ascontiguousarray(rgba[:, :, list(self._order)])
        return self._headshot

    def _extend_ball_trail(self, point):
        if self._ball_trail and not self._ball_could_move(
            self._ball_trail[-1], self._ball_trail_frame, point
        ):
            # a stray box must not wipe the trail
            pending = self._ball_trail_pending
            if pending is None or not self._ball_could_move(*pending, point):
                self._ball_trail_pending = (point, self._frame)
                return
            self._ball_trail = [pending[0]]
        self._ball_trail_pending = None
        self._ball_trail.append(point)
        del self._ball_trail[: -self.trail_length]
        self._ball_trail_frame = self._frame

    def _ball_could_move(self, start, start_frame, end):
        elapsed = max(1, self._frame - start_frame)
        if elapsed > _BALL_TRAIL_MAX_GAP:
            return False
        jump = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        steps = min(elapsed, _BALL_TRAIL_MAX_STEPS_PER_GAP)
        return jump <= _BALL_TRAIL_MAX_STEP * self.width * steps

    def _draw_trail(self, drawing, points, rgba):
        if len(points) < 2:
            return
        drawing.draw_polyline(points, rgba, 2)

    def _draw_ellipse(self, drawing, box, rgba, track_id):
        x1, y1, x2, y2 = box
        y_bottom = int(y2)
        x_center = int((x1 + x2) / 2)
        # Prefer the per-track smoothed width so the ellipse stays stable even
        # when a single detection box is momentarily oversized.
        smoothed = self._ell_w.get(track_id)
        width = max(1, int(smoothed if smoothed is not None else x2 - x1))
        drawing.draw_ellipse_arc(
            (x_center, y_bottom),
            (width, max(1, int(_ELLIPSE_HEIGHT_RATIO * width))),
            _ELLIPSE_START_ANGLE,
            _ELLIPSE_END_ANGLE,
            rgba,
            2,
        )
        if self.show_ids and track_id is not None:
            rect_w, rect_h = 40, 18
            x1r = x_center - rect_w // 2
            x2r = x_center + rect_w // 2
            y1r = y_bottom - rect_h // 2 + 15
            y2r = y_bottom + rect_h // 2 + 15
            drawing.fill_rectangle((x1r, y1r), (x2r, y2r), rgba)
            tx = x1r + 12 - (10 if track_id > 99 else 0)
            drawing.draw_text(str(track_id), (tx, y1r + 14), 0.5, _BLACK_RGBA, 2)

    def _draw_triangle(self, drawing, box, rgba):
        x1, y1, x2, y2 = box
        x = int((x1 + x2) / 2)
        y = int(y1)
        drawing.draw_triangle(
            [(x, y), (x - 10, y - 20), (x + 10, y - 20)], rgba, _BLACK_RGBA, 2
        )

    def _draw_focal_marker(self, drawing, box):
        # Broadcast-style "selected player" chevron floating above the head,
        # plus a bolder ellipse, to flag the focal (HUD) player on the pitch.
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        tip_y = int(y1) - 10
        s = 16
        drawing.draw_triangle(
            [
                (cx, tip_y),
                (cx - s, tip_y - int(s * 1.5)),
                (cx + s, tip_y - int(s * 1.5)),
            ],
            _HIGHLIGHT_RGBA,
            _BLACK_RGBA,
            2,
        )
        # Bolder ring at the feet to reinforce the selection.
        x_center = int((x1 + x2) / 2)
        width = max(1, int(x2 - x1))
        drawing.draw_ellipse_arc(
            (x_center, int(y2)),
            (width, max(1, int(_ELLIPSE_HEIGHT_RATIO * width))),
            _ELLIPSE_START_ANGLE,
            _ELLIPSE_END_ANGLE,
            _HIGHLIGHT_RGBA,
            4,
        )

    def _draw_label(self, drawing, box, label, rgba):
        x1, y1, _, _ = box
        drawing.draw_text(label, (int(x1), max(12, int(y1) - 6)), 0.5, rgba, 1)

    def _hud_pixels(self, contacts, distance_m, rgba, headshot):
        distance_text = f"{distance_m:.1f}"
        key = (
            self.player_label,
            contacts,
            distance_text,
            rgba,
            self._format_seen,
            id(headshot),
        )
        if key != self._hud_bitmap_key:
            self._hud_bitmap = self._render_hud(contacts, distance_text, rgba, headshot)
            self._hud_bitmap_key = key
        return self._hud_bitmap

    def _render_hud(self, contacts, distance_text, rgba, headshot):
        import numpy as np

        x = y = _HUD_BORDER_OVERHANG
        if headshot is not None:
            hh, hw = headshot.shape[:2]
            w, h = hw + 280, max(110, hh + 20)
            text_x = x + hw + 20
        else:
            w, h = 320, 100
            text_x = x + 12
        bitmap_size = (
            h + 1 + 2 * _HUD_BORDER_OVERHANG,
            w + 1 + 2 * _HUD_BORDER_OVERHANG,
            _RGBA_CHANNELS,
        )
        drawing = OpenCVSportDrawing(self._format_seen)
        # the thick border leaves its corner pixels unpainted
        bitmap = np.empty(bitmap_size, dtype=np.uint8)
        bitmap[:] = drawing.buffer_colour(rgba)
        with drawing.drawing_on(bitmap):
            drawing.fill_rectangle((x, y), (x + w, y + h), _HUD_BG_RGBA)
            drawing.draw_rectangle((x, y), (x + w, y + h), rgba, _HUD_BORDER_THICKNESS)
            if headshot is not None:
                hy, hx = y + 10, x + 10
                drawing.blit(headshot, hx, hy)
                drawing.draw_rectangle((hx, hy), (hx + hw, hy + hh), rgba, 2)
            drawing.draw_text(
                self.player_label, (text_x, y + 28), 0.7, _HUD_TEXT_RGBA, 2
            )
            drawing.draw_text(
                f"Ball contacts: {contacts}", (text_x, y + 58), 0.6, _HUD_TEXT_RGBA, 1
            )
            drawing.draw_text(
                f"Distance: {distance_text} m", (text_x, y + 85), 0.6, _HUD_TEXT_RGBA, 1
            )
        return bitmap

    def process_in_place(self, frame, fmt, buf):
        import cv2
        import numpy as np

        if fmt != self._format_seen:
            self._use_format(fmt)
        all_entries = self._read_metadata(buf)
        # The buffer carries both the detector's boxes (track_id None) and
        # the tracker's boxes (track_id set). Tracking state/HUD always use
        # the tracked entries; what we *draw* depends on draw_from_detections.
        track_entries = [e for e in all_entries if e["track_id"] is not None]
        det_entries = [e for e in all_entries if e["track_id"] is None]

        # Ball position for contact counting: prefer a tracked ball, else
        # fall back to the strongest ball *detection* (the ball is small and
        # fast, so it often isn't tracked) -- so contacts still get counted.
        det_ball_box = None
        best_ball = -1.0
        for e in det_entries:
            if _is_ball(e["label"]) and e["confidence"] > best_ball:
                best_ball, det_ball_box = e["confidence"], e["box"]

        # Per-track state (votes, contacts, distance, focal) from the tracker.
        active, kick = self._update_tracks(
            track_entries if track_entries else all_entries, det_ball_box
        )
        if kick is not None:
            self._record_kick(buf, kick)

        if self.draw_from_detections:
            draw_entries = list(det_entries)
            # Bridge missed detections: the detector occasionally drops a
            # player for a frame, which would flicker the circle. The tracker
            # is still coasting that player (Kalman keep-alive), so draw any
            # confirmed track that has no detection this frame -- detections
            # still drive everything they cover; tracks only fill the gaps.
            if track_entries:
                covered = set()
                for d in det_entries:
                    if _is_ball(d["label"]):
                        continue
                    for t in track_entries:
                        if t["track_id"] in covered or _is_ball(t["label"]):
                            continue
                        if self._overlap(d["box"], t["box"]) >= 0.3:
                            covered.add(t["track_id"])
                draw_entries += [
                    t
                    for t in track_entries
                    if not _is_ball(t["label"]) and t["track_id"] not in covered
                ]
        else:
            draw_entries = track_entries if track_entries else det_entries
        # min-confidence gates only what we *draw* (tracks carry conf 1.0, so
        # they're unaffected); the contact math above used the raw detections.
        if self.min_confidence > 0.0:
            draw_entries = [
                e for e in draw_entries if e["confidence"] >= self.min_confidence
            ]
        # Collapse overlapping boxes so one player isn't circled twice,
        # then low-pass the positions so the circle glides.
        draw_entries = self._merge_overlaps(draw_entries)
        draw_entries = self._smooth_boxes(np, draw_entries)
        if not all_entries:
            return
        drawing = drawing_for(self._drawings, frame, fmt)
        with drawing.drawing_on(frame):
            self._draw_frame(cv2, np, drawing, active, track_entries, draw_entries)

    def _draw_frame(self, cv2, np, drawing, active, track_entries, draw_entries):
        # Jersey team voting first, so trails/ellipses use this frame's
        # vote (track mode; detection mode classifies per box at draw).
        # Referees are voted on too -- their colour comes from the jersey
        # (gold only as the fallback), not the class label.
        if self.team_colors:
            for e in track_entries:
                tid = e["track_id"]
                lab = self._stable_label(tid, e["label"])
                if _is_ball(lab):
                    continue
                vote = drawing.classify_jersey(e["box"])
                if vote:
                    tv = self._team_votes.setdefault(
                        tid, {"red": 0, "blue": 0, "ref": 0}
                    )
                    tv[vote] = tv.get(vote, 0) + 1

        if self.trails:
            for tid in active:
                rgba = self._color_for(self._track_label.get(tid, ""), tid)
                if rgba is None:
                    continue
                self._draw_trail(drawing, self._trail.get(tid, []), rgba)
        if self.ball_trail:
            self._draw_trail(drawing, self._ball_trail, _BALL_RGBA)

        # Which drawn box is the focal (HUD) player? Match the focal
        # track's box to the nearest drawn box so we can highlight it
        # even when drawing from detections (no track id on the box).
        focal_idx = None
        if self.highlight_focal:
            focal_tid = self._focal_track()
            focal_box = None
            if focal_tid is not None:
                for t in track_entries:
                    if t["track_id"] == focal_tid:
                        focal_box = t["box"]
                        break
            if focal_box is not None:
                best = 0.0
                for i, e in enumerate(draw_entries):
                    if _is_ball(e["label"]):
                        continue
                    ov = self._overlap(e["box"], focal_box)
                    if ov > best:
                        best, focal_idx = ov, i

        # Stable track id per drawn box (detection boxes borrow the id of
        # the track they overlap) -- used for the id badge and to look up
        # the track's accumulated colour.
        draw_ids = self._assign_track_ids(draw_entries, track_entries)

        for i, e in enumerate(draw_entries):
            box = e["box"]
            badge_id = draw_ids[i]
            # Use the track's stable identity (class + accumulated team
            # votes) for colour whenever the box maps to a track -- in
            # detection mode that's the box's matched track id. This
            # makes colour robust to per-frame label/jersey noise. Only
            # an unmatched detection falls back to this frame's guess.
            color_tid = e["track_id"] if e["track_id"] is not None else badge_id
            if color_tid is not None:
                label = self._stable_label(color_tid, e["label"])
            else:
                label = e["label"]
            if _is_ball(label):
                if self.show_ball:
                    self._draw_triangle(drawing, box, _BALL_RGBA)
                continue
            if color_tid is not None:
                rgba = self._color_for(label, color_tid)
            else:
                rgba = self._detection_color(drawing, label, box)
            if rgba is None:
                continue
            self._draw_ellipse(drawing, box, rgba, badge_id)
            if i == focal_idx:
                self._draw_focal_marker(drawing, box)
            if self.show_labels:
                self._draw_label(drawing, box, label, rgba)

        if self.show_hud:
            focal = self._focal_track()
            if focal is not None:
                ppm = self._px_per_meter()
                dist_m = (self._distance_px.get(focal, 0.0) / ppm) if ppm else 0.0
                hud_rgba = (
                    self._color_for(self._track_label.get(focal, ""), focal)
                    or _DEFAULT_RGBA
                )
                hud = self._hud_pixels(
                    self._contacts.get(focal, 0),
                    dist_m,
                    hud_rgba,
                    self._load_headshot(cv2, np),
                )
                drawing.blit(
                    hud,
                    _HUD_LEFT - _HUD_BORDER_OVERHANG,
                    _HUD_TOP - _HUD_BORDER_OVERHANG,
                )
        self._draw_kick_flash(drawing)


if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_football_overlay", FootballOverlay
    )
