# Multi-Object Tracker
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import backend
from backend import (
    analytics,
    GObject,
    InPlaceVideoTransform,
    METADATA_FRAME_CAPABILITIES,
)
from backend.analytics import stream_label


def covered_fraction(boxes, by):
    import numpy as np

    if len(boxes) == 0 or len(by) == 0:
        return np.empty((len(boxes), len(by)))
    a = np.array(boxes, dtype=float)
    b = np.array(by, dtype=float)
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    yy2 = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    return inter / np.maximum(a[:, 2] * a[:, 3], 1e-7)[:, None]


def merge_covered_detections(detections, ratio):
    kept = []
    for det in sorted(detections, key=lambda d: -d[4]):
        if kept and covered_fraction([det[:4]], [k[:4] for k in kept]).max() > ratio:
            continue
        kept.append(det)
    return kept


class KalmanBoxTracker:
    """Simple Kalman filter tracker for a single bounding box."""

    _id_counter = 0

    def __init__(self, bbox):
        import numpy as np

        self.id = KalmanBoxTracker._id_counter
        KalmanBoxTracker._id_counter += 1
        # State: [x_center, y_center, w, h, vx, vy, vw, vh]
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        self.state = np.array([cx, cy, bbox[2], bbox[3], 0, 0, 0, 0], dtype=np.float64)
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.matched_detection = None

    def predict(self):
        """Advance state by one frame using constant velocity model."""
        self.state[:4] += self.state[4:]
        self.age += 1
        self.time_since_update += 1
        self.matched_detection = None
        return self._get_bbox()

    def update(self, bbox):
        """Update state with observed bounding box [x, y, w, h]."""
        import numpy as np

        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        observed = np.array([cx, cy, bbox[2], bbox[3]], dtype=np.float64)
        # Simple exponential smoothing for velocity
        alpha = 0.5
        new_vel = observed - self.state[:4]
        self.state[4:] = alpha * new_vel + (1 - alpha) * self.state[4:]
        self.state[:4] = observed
        self.hits += 1
        self.time_since_update = 0

    def _get_bbox(self):
        """Return [x, y, w, h] from state."""
        import numpy as np

        w = max(self.state[2], 0)
        h = max(self.state[3], 0)
        x = self.state[0] - w / 2.0
        y = self.state[1] - h / 2.0
        return np.array([x, y, w, h])

    def get_bbox(self):
        return self._get_bbox()


def iou_batch(bb_det, bb_trk):
    """Compute IoU between two sets of [x, y, w, h] bounding boxes."""
    import numpy as np

    if len(bb_det) == 0 or len(bb_trk) == 0:
        return np.empty((len(bb_det), len(bb_trk)))

    det = np.array(bb_det)
    trk = np.array(bb_trk)

    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    det_xy = np.column_stack(
        [det[:, 0], det[:, 1], det[:, 0] + det[:, 2], det[:, 1] + det[:, 3]]
    )
    trk_xy = np.column_stack(
        [trk[:, 0], trk[:, 1], trk[:, 0] + trk[:, 2], trk[:, 1] + trk[:, 3]]
    )

    xx1 = np.maximum(det_xy[:, None, 0], trk_xy[None, :, 0])
    yy1 = np.maximum(det_xy[:, None, 1], trk_xy[None, :, 1])
    xx2 = np.minimum(det_xy[:, None, 2], trk_xy[None, :, 2])
    yy2 = np.minimum(det_xy[:, None, 3], trk_xy[None, :, 3])

    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)

    area_det = det[:, 2] * det[:, 3]
    area_trk = trk[:, 2] * trk[:, 3]

    union = area_det[:, None] + area_trk[None, :] - inter
    return inter / np.maximum(union, 1e-7)


class SortTracker:
    """SORT/ByteTrack multi-object tracker using IoU + Kalman filtering."""

    def __init__(
        self,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        keep_alive=2,
        new_track_conf=0.25,
        camera_motion=True,
        dup_iou=0.8,
        merge_overlap=0.5,
        distance_gate=1.0,
        new_track_max_overlap=0.1,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        # ByteTrack-style activation gate: a brand-new track is only started
        # from a confident detection. Weak/ghost boxes can still *continue* an
        # existing track (matched above) but won't spawn phantom circles.
        self.new_track_conf = new_track_conf
        # Keep emitting a confirmed track (with its Kalman-predicted box) for up
        # to keep_alive frames after a missed detection — bridges flicker so the
        # overlay doesn't blink when the detector drops a box for a frame or two.
        self.keep_alive = keep_alive
        # Camera-motion compensation: estimate the global image shift from the
        # tracks that matched, then re-try matching the leftovers with their
        # predictions shifted by it. Re-attaches players during a pan instead of
        # leaving the old track behind and spawning a duplicate.
        self.camera_motion = camera_motion
        # Two confirmed tracks overlapping more than this IoU are duplicates;
        # the weaker one is dropped (ByteTrack's remove_duplicate_stracks).
        self.dup_iou = dup_iou
        self.merge_overlap = merge_overlap
        self.distance_gate = distance_gate
        self.new_track_max_overlap = new_track_max_overlap
        self.trackers = []

    @staticmethod
    def _center(bbox):
        return (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)

    def _estimate_motion(self, matches, predicted, det_bboxes):
        """Fit a global 2D similarity transform (translation + uniform scale +
        rotation) mapping each matched track's predicted centre to its observed
        centre. Returns a callable box->warped-box, or None if it can't be
        estimated. Uses RANSAC (via OpenCV) so players moving against the camera
        consensus are rejected as outliers; falls back to a robust median
        translation if OpenCV is unavailable or the fit is degenerate."""
        import numpy as np

        if len(matches) < 3:
            return None
        src = np.array(
            [self._center(predicted[ti]) for _, ti in matches], dtype=np.float32
        )
        dst = np.array(
            [self._center(det_bboxes[di]) for di, _ in matches], dtype=np.float32
        )

        M = None
        try:
            import cv2

            M, _ = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0
            )
        except Exception:
            M = None

        if M is not None:
            scale = float(np.hypot(M[0, 0], M[0, 1]))
            # Reject implausible fits (e.g. from too few/noisy correspondences).
            if 0.5 <= scale <= 2.0:

                def warp(box):
                    cx, cy = self._center(box)
                    ncx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
                    ncy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
                    nw, nh = box[2] * scale, box[3] * scale
                    return np.array([ncx - nw / 2.0, ncy - nh / 2.0, nw, nh])

                return warp

        # Fallback: robust median translation (pan/tilt only).
        delta = dst - src
        tx, ty = float(np.median(delta[:, 0])), float(np.median(delta[:, 1]))
        if abs(tx) < 1.0 and abs(ty) < 1.0:
            return None
        return lambda box: np.array([box[0] + tx, box[1] + ty, box[2], box[3]])

    def _associate(self, det_bboxes, det_idxs, trk_idxs, trk_boxes, detections):
        """Hungarian-match a subset of detections to a subset of trackers,
        applying updates to matched trackers. Returns list of (det_i, trk_i)."""
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        if not det_idxs or not trk_idxs:
            return []
        dets = [det_bboxes[d] for d in det_idxs]
        trks = [trk_boxes[t] for t in trk_idxs]
        affinity = iou_batch(dets, trks)
        if affinity.size == 0:
            return []
        if self.distance_gate > 0:
            affinity = np.maximum(affinity, self._closeness(dets, trks))
        row_ind, col_ind = linear_sum_assignment(1.0 - affinity)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if affinity[r, c] >= self.iou_threshold:
                di, ti = det_idxs[r], trk_idxs[c]
                self.trackers[ti].update(detections[di][:4])
                self.trackers[ti].label_quark = detections[di][5]
                self.trackers[ti].matched_detection = detections[di]
                matches.append((di, ti))
        return matches

    # 1 at the track's centre, 0 at distance_gate box sizes away
    def _closeness(self, dets, trks):
        import numpy as np

        det = np.array(dets, dtype=float)
        trk = np.array(trks, dtype=float)
        det_centre = det[:, :2] + det[:, 2:] / 2.0
        trk_centre = trk[:, :2] + trk[:, 2:] / 2.0
        distance = np.linalg.norm(
            det_centre[:, None, :] - trk_centre[None, :, :], axis=2
        )
        size = np.maximum(np.maximum(trk[:, 2], trk[:, 3]), 1.0)[None, :]
        return np.clip(1.0 - distance / (self.distance_gate * size), 0.0, 1.0)

    def _suppress_duplicates(self):
        """Drop the weaker of any two confirmed tracks sitting on the same box."""
        n = len(self.trackers)
        if n < 2:
            return
        boxes = [t.get_bbox() for t in self.trackers]
        iou_matrix = iou_batch(boxes, boxes)
        remove = set()
        for i in range(n):
            if i in remove:
                continue
            for j in range(i + 1, n):
                if j in remove:
                    continue
                if iou_matrix[i, j] > self.dup_iou:
                    ti, tj = self.trackers[i], self.trackers[j]
                    # Keep the better track: matched more recently, then more
                    # hits; drop the other (usually the freshly-spawned dup).
                    ki = (ti.time_since_update, -ti.hits)
                    kj = (tj.time_since_update, -tj.hits)
                    remove.add(j if ki <= kj else i)
        if remove:
            self.trackers = [t for k, t in enumerate(self.trackers) if k not in remove]

    def _on_a_matched_track(self, box, matched_boxes):
        if not matched_boxes or self.new_track_max_overlap >= 1.0:
            return False
        return iou_batch([box], matched_boxes).max() > self.new_track_max_overlap

    def update(self, detections):
        """
        Update tracks with new detections.

        Args:
            detections: list of [x, y, w, h, score, label_quark] arrays

        Returns:
            list of (track_id, bbox, label_quark, matched detection or None)
            for confirmed tracks
        """
        import numpy as np

        # Predict new locations for existing tracks
        to_remove = []
        for i, trk in enumerate(self.trackers):
            if np.any(np.isnan(trk.predict())):
                to_remove.append(i)
        for i in reversed(to_remove):
            self.trackers.pop(i)

        if self.merge_overlap > 0:
            detections = merge_covered_detections(detections, self.merge_overlap)
        det_bboxes = [d[:4] for d in detections] if len(detections) > 0 else []
        n_det = len(det_bboxes)
        n_trk = len(self.trackers)
        # Predicted box per tracker, captured before any update this frame.
        predicted = [self.trackers[i].get_bbox() for i in range(n_trk)]

        # 1) First association on the raw predictions.
        matches = self._associate(
            det_bboxes, list(range(n_det)), list(range(n_trk)), predicted, detections
        )
        matched_det = {di for di, _ in matches}
        matched_trk = {ti for _, ti in matches}

        # 2) Camera-motion compensation: fit a global image transform (pan, zoom
        # and rotation) from the tracks that matched, apply it to the leftover
        # predictions, and re-match. This recovers tracks during camera moves
        # instead of leaving them behind and spawning duplicates.
        if self.camera_motion:
            warp = self._estimate_motion(matches, predicted, det_bboxes)
            if warp is not None:
                rem_trk = [i for i in range(n_trk) if i not in matched_trk]
                rem_det = [d for d in range(n_det) if d not in matched_det]
                if rem_trk and rem_det:
                    shifted = {i: warp(predicted[i]) for i in rem_trk}
                    m2 = self._associate(
                        det_bboxes, rem_det, rem_trk, shifted, detections
                    )
                    matched_det.update(di for di, _ in m2)
                    matched_trk.update(ti for _, ti in m2)

        # Create new tracks for unmatched detections, but only from confident
        # ones (ByteTrack activation gate) so weak/ghost boxes don't start a
        # phantom track that gets drawn as a stray circle.
        matched_boxes = [self.trackers[ti].get_bbox() for ti in matched_trk]
        for d_idx in range(n_det):
            if d_idx not in matched_det:
                if detections[d_idx][4] < self.new_track_conf:
                    continue
                if self._on_a_matched_track(det_bboxes[d_idx], matched_boxes):
                    continue
                trk = KalmanBoxTracker(detections[d_idx][:4])
                trk.label_quark = detections[d_idx][5]
                trk.matched_detection = detections[d_idx]
                self.trackers.append(trk)

        # Remove dead tracks, then drop duplicate tracks sitting on one object.
        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]
        self._suppress_duplicates()

        # Return confirmed tracks, including ones that missed a detection this
        # frame (predicted box) for up to keep_alive frames — prevents flicker.
        results = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits and trk.time_since_update <= self.keep_alive:
                results.append(
                    (trk.id, trk.get_bbox(), trk.label_quark, trk.matched_detection)
                )
        return results


class TrackerTransform(InPlaceVideoTransform):
    __gstmetadata__ = (
        "Multi-Object Tracker",
        "Transform",
        "Assigns persistent track IDs to detected objects using ByteTrack/SORT",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    FRAME_CAPABILITIES = METADATA_FRAME_CAPABILITIES

    tracker_type = GObject.Property(
        type=str,
        default="bytetrack",
        nick="Tracker Type",
        blurb="Tracking algorithm to use: 'bytetrack' or 'sort'",
        flags=GObject.ParamFlags.READWRITE,
    )

    max_age = GObject.Property(
        type=int,
        default=30,
        minimum=1,
        maximum=1000,
        nick="Max Age",
        blurb="Maximum number of frames to keep a lost track before deletion",
        flags=GObject.ParamFlags.READWRITE,
    )

    min_hits = GObject.Property(
        type=int,
        default=3,
        minimum=1,
        maximum=100,
        nick="Min Hits",
        blurb="Minimum detections before a track is confirmed",
        flags=GObject.ParamFlags.READWRITE,
    )

    iou_threshold = GObject.Property(
        type=float,
        default=0.3,
        minimum=0.0,
        maximum=1.0,
        nick="IoU Threshold",
        blurb="Minimum match score for detection-to-track assignment: the IoU, "
        "or the closeness within distance-gate, whichever is higher",
        flags=GObject.ParamFlags.READWRITE,
    )

    distance_gate = GObject.Property(
        type=float,
        default=1.0,
        minimum=0.0,
        maximum=10.0,
        nick="Distance Gate",
        blurb="Also match a detection whose centre is within this many box sizes "
        "of the track's predicted centre, scored 1 at the centre down to 0 at "
        "the gate, so a small box that moved its own width still matches "
        "(0 = overlap only)",
        flags=GObject.ParamFlags.READWRITE,
    )

    merge_overlap = GObject.Property(
        type=float,
        default=0.5,
        minimum=0.0,
        maximum=1.0,
        nick="Merge Overlap",
        blurb="Drop a detection when a higher-scoring one covers more than this "
        "fraction of its box, a second box on the same object that NMS kept "
        "(0 = keep every detection)",
        flags=GObject.ParamFlags.READWRITE,
    )

    new_track_max_overlap = GObject.Property(
        type=float,
        default=0.1,
        minimum=0.0,
        maximum=1.0,
        nick="New Track Max Overlap",
        blurb="A detection overlapping a track that matched this frame by more "
        "than this IoU starts no new track, it is a second box on that object "
        "(1 = always start one)",
        flags=GObject.ParamFlags.READWRITE,
    )

    keep_alive = GObject.Property(
        type=int,
        default=2,
        minimum=0,
        maximum=1000,
        nick="Keep Alive",
        blurb="Frames to keep emitting a confirmed track (Kalman-predicted box) "
        "after a missed detection; bridges flicker (0 = only matched frames)",
        flags=GObject.ParamFlags.READWRITE,
    )

    new_track_confidence = GObject.Property(
        type=float,
        default=0.25,
        minimum=0.0,
        maximum=1.0,
        nick="New Track Confidence",
        blurb="Minimum detection confidence to START a new track (ByteTrack "
        "activation gate); weak boxes still continue existing tracks but "
        "won't spawn phantom/duplicate circles",
        flags=GObject.ParamFlags.READWRITE,
    )

    camera_motion = GObject.Property(
        type=bool,
        default=True,
        nick="Camera Motion Compensation",
        blurb="Estimate the global image shift from matched tracks and re-match "
        "leftovers shifted by it, so a panning camera re-attaches players "
        "instead of leaving the old track behind and spawning a duplicate",
        flags=GObject.ParamFlags.READWRITE,
    )

    duplicate_iou = GObject.Property(
        type=float,
        default=0.8,
        minimum=0.0,
        maximum=1.0,
        nick="Duplicate IoU",
        blurb="Two confirmed tracks overlapping more than this are treated as "
        "duplicates and the weaker one is dropped",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self._passing_tracked_through = False
        self._tracker = None

    def _ensure_tracker(self):
        if self._tracker is None:
            self._tracker = SortTracker(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold,
                keep_alive=self.keep_alive,
                new_track_conf=self.new_track_confidence,
                camera_motion=self.camera_motion,
                dup_iou=self.duplicate_iou,
                merge_overlap=self.merge_overlap,
                distance_gate=self.distance_gate,
                new_track_max_overlap=self.new_track_max_overlap,
            )
        return self._tracker

    def _read_detections(self, buf):
        """Extract detections from upstream object-detection metadata."""
        detections = []
        meta = analytics.get_relation_meta(buf)
        if not meta:
            return detections

        for obj in analytics.read_objects(meta):
            detections.append(
                [
                    obj["x"],
                    obj["y"],
                    obj["w"],
                    obj["h"],
                    obj["score"],
                    stream_label(obj["stream"], obj["label"]),
                    obj,
                ]
            )
        return detections

    def process_in_place(self, frame, fmt, buf):
        tracker = self._ensure_tracker()
        detections = self._read_detections(buf)
        if detections and all(row[6]["track_id"] is not None for row in detections):
            if not self._passing_tracked_through:
                self.logger.warning(
                    "detections arrive with track ids, mp_tracker passes them through"
                )
                self._passing_tracked_through = True
            return
        # an empty frame still ages the tracks
        tracked = tracker.update(detections)
        if not tracked:
            return
        meta = analytics.get_relation_meta(buf) or analytics.add_relation_meta(buf)
        if not meta:
            raise RuntimeError("failed to add analytics relation metadata")
        for track_id, bbox, label, matched in tracked:
            if matched is not None:
                analytics.track_object(meta, matched[6], track_id)
                continue
            # a track that missed its detection this frame shows its predicted box
            x, y, w, h = bbox
            box = analytics.add_object(meta, label, int(x), int(y), int(w), int(h), 1.0)
            analytics.relate(meta, box, analytics.add_tracking(meta, track_id))
        self.logger.info(
            f"Tracker: {len(detections)} detections -> {len(tracked)} confirmed tracks"
        )


if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("mp_tracker", TrackerTransform)
