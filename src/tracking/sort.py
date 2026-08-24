"""
sort.py — SORT: Simple Online and Realtime Tracking
====================================================
SORT (Bewley et al., 2016, https://arxiv.org/abs/1602.00763) is the
minimal-yet-surprisingly-strong multi-object tracker:

  1. PREDICT   every existing track one frame ahead (Kalman filter)
  2. ASSOCIATE new detections to predicted tracks via IoU + Hungarian
  3. UPDATE    matched tracks with their detection; spawn tracks from
               unmatched detections; age out tracks that keep missing

Why it works: cats move smoothly between consecutive video frames, so a
box's *predicted* position overlaps heavily with its next detection — even
if the detector misses the cat for a few frames (occlusion), the Kalman
prediction keeps the box coasting along its velocity until it reappears.

Track lifecycle:
  - New track      : created for any unmatched detection
  - Confirmed track: reported in outputs once it has been hit min_hits times
                     (suppresses spurious single-frame false positives)
  - Deleted track  : missed for more than max_age consecutive frames

References:
  - Paper:    https://arxiv.org/abs/1602.00763
  - Original: https://github.com/abewley/sort
"""

import numpy as np
import torch

from src.tracking.kalman_filter import (
    KalmanBoxFilter,
    xyxy_to_cxcyah,
    cxcyah_to_xyxy,
)
from src.tracking.hungarian import assign
from src.utils.boxes import compute_iou


# ─── Track: a single tracked object ────────────────────────────────
class Track:
    """
    One tracked object: owns its Kalman state and lifecycle bookkeeping.

    Attributes:
        id                (int):   unique track identifier (1, 2, 3, ...)
        mean / covariance (ndarray): Kalman filter state [8] and [8, 8]
        score             (float): confidence of the most recent detection
        hits              (int):   total number of matched detections so far
        hit_streak        (int):   consecutive frames WITH a matched detection
                                     (reset to 0 on the first miss)
        age               (int):   total frames since creation (incl. misses)
        time_since_update (int):   consecutive frames without a match
        history           (list):  last N boxes [x1,y1,x2,y2], newest last
                                     (used by the demo to draw motion trails)

    Class attribute `_count` guarantees globally unique IDs.
    """

    _count = 0  # class-level counter -> unique ids across all Track objects

    def __init__(self, bbox_xyxy, score: float = 1.0):
        """
        Args:
            bbox_xyxy (sequence): initial detection [x1, y1, x2, y2]
            score     (float):   detector confidence for this detection
        """
        bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.float64)

        Track._count += 1
        self.id = Track._count

        # Kalman filter expects center format [cx, cy, w, h]
        self.kf = KalmanBoxFilter()
        self.mean, self.covariance = self.kf.initiate(xyxy_to_cxcyah(bbox_xyxy))

        self.score = float(score)
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.time_since_update = 0

        self.history = [bbox_xyxy.tolist()]  # motion trail for visualization

    # ── Lifecycle methods ─────────────────────────────────────────
    def predict(self) -> np.ndarray:
        """
        Advance the Kalman state one frame. Called for EVERY track each
        frame, matched or not.

        Returns:
            ndarray: predicted box [x1, y1, x2, y2] (before correction)
        """
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1

        if self.time_since_update > 0:
            # We are already coasting on predictions only; the streak of
            # confirmed detections is broken.
            self.hit_streak = 0
        self.time_since_update += 1

        return self.to_xyxy()

    def update(self, bbox_xyxy, score: float = None):
        """
        Correct this track's state with a freshly associated detection.

        Args:
            bbox_xyxy (sequence): detection box [x1, y1, x2, y2]
            score     (float):   optional updated confidence
        """
        bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.float64)

        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, xyxy_to_cxcyah(bbox_xyxy),
        )
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        if score is not None:
            self.score = float(score)

        self.history.append(bbox_xyxy.tolist())
        if len(self.history) > 30:          # keep trail bounded
            self.history.pop(0)

    @property
    def is_confirmed(self) -> bool:
        """Has this track been detected enough times to be trusted?"""
        return self.hits >= 3 or self.age <= 3

    # ── Geometry helpers ──────────────────────────────────────────
    def to_xyxy(self) -> np.ndarray:
        """Current Kalman estimate as [x1, y1, x2, y2]."""
        return cxcyah_to_xyxy(self.mean)


# ─── SORT tracker ──────────────────────────────────────────────────
class SORTTracker:
    """
    Frame-by-frame multi-object tracker.

    Usage:
        tracker = SORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        for frame in video:
            detections = detector.predict(frame)          # [N, 4] xyxy boxes
            outputs = tracker.update(detections, scores)  # active tracks
            # outputs: list of dicts {'id', 'box', 'score'}

    Args:
        max_age       (int): frames a track survives WITHOUT a matching
                             detection before deletion (occlusion tolerance).
                             Default 30 (~1 second at 30 fps).
        min_hits      (int): consecutive detections required before a track
                             appears in outputs (false-positive filter).
        iou_threshold (float): minimum IoU between a predicted track box and
                               a detection for them to be associable.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks = []
        self.frame_count = 0

    # ── Core per-frame API ────────────────────────────────────────
    def update(self, detections, scores=None):
        """
        Process one frame: predict → associate → update → prune.

        Args:
            detections (ndarray): [N, 4] boxes in [x1,y1,x2,y2]; may be empty
                                  when the detector finds nothing.
            scores     (ndarray): [N] confidences (optional).

        Returns:
            list[dict]: one entry per ACTIVE confirmed track this frame:
                {'id': int, 'box': ndarray[4], 'score': float}
                A track is reported only if time_since_update == 0 (it was
                actually re-detected this frame) AND it has enough hits.
        """
        self.frame_count += 1

        detections = np.asarray(detections, dtype=np.float64).reshape(-1, 4)
        num_dets = detections.shape[0]

        if scores is None:
            scores = np.ones(num_dets)
        else:
            scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        # ── 1. PREDICT: advance every track one frame ─────────────
        for trk in self.tracks:
            trk.predict()
        predicted_boxes = np.stack(
            [trk.to_xyxy() for trk in self.tracks]
        ) if self.tracks else np.zeros((0, 4))

        # ── 2. ASSOCIATE detections ↔ predicted track boxes ───────
        matches, unmatched_dets, unmatched_trks = self._associate(
            detections, predicted_boxes,
        )

        # ── 3a. UPDATE matched tracks with their detection ────────
        for det_idx, trk_idx in matches:
            trk = self.tracks[trk_idx]
            trk.update(detections[det_idx],
                       score=scores[det_idx] if det_idx < len(scores) else None)
            self._on_track_matched(trk, det_idx)

        # ── 3b. Spawn new tracks from unmatched detections ────────
        for det_idx in unmatched_dets:
            self.tracks.append(self._new_track(
                detections[det_idx], scores[det_idx], det_idx,
            ))

        # ── 3c. Delete stale tracks ───────────────────────────────
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]

        # ── 4. Collect output for THIS frame ──────────────────────
        output = []
        for trk in self.tracks:
            if trk.time_since_update == 0 and (
                trk.hit_streak >= self.min_hits
                or self.frame_count <= self.min_hits
            ):
                output.append({
                    "id":    trk.id,
                    "box":   trk.to_xyxy(),
                    "score": trk.score,
                    "trail": np.asarray(trk.history, dtype=np.float64),
                })
        return output

    # ── Extension points (overridden by DeepSORTTracker) ───────────
    def _new_track(self, bbox_xyxy, score: float, det_idx: int = None) -> Track:
        """Factory for new tracks (DeepSORT swaps in an appearance-aware
        subclass here and uses det_idx to attach the detection's embedding)."""
        return Track(bbox_xyxy, score)

    def _on_track_matched(self, track: Track, det_idx: int):
        """Called right after a track consumes detection `det_idx`.
        Base implementation does nothing (SORT has no appearance memory)."""
        pass

    # ── Association logic (overridden by DeepSORT) ─────────────────
    def _associate(self, detections: np.ndarray, track_boxes: np.ndarray):
        """
        Match detections to predicted track boxes using IoU cost +
        Hungarian assignment.

        Returns:
            (matches, unmatched_det_indices, unmatched_track_indices)
            matches: list of (det_idx, trk_idx)
        """
        num_trks = len(self.tracks)
        if num_trks == 0 or detections.shape[0] == 0:
            return [], list(range(detections.shape[0])), list(range(num_trks))

        # IoU matrix [num_trks, num_dets] — reuse Week 2 utility
        iou_matrix = compute_iou(
            torch.as_tensor(track_boxes, dtype=torch.float32),
            torch.as_tensor(detections, dtype=torch.float32),
        ).numpy()

        cost_matrix = 1.0 - iou_matrix

        valid_mask = iou_matrix >= self.iou_threshold

        # assign() solves rows=tracks × cols=detections and returns
        # (matches, unmatched_rows, unmatched_cols) in (trk, det) order;
        # convert to the (det_idx, trk_idx) contract of update().
        matches, unmatched_rows, unmatched_cols = assign(
            cost_matrix, valid_mask=valid_mask,
        )
        matches = [(c, r) for r, c in matches]
        return matches, unmatched_cols, unmatched_rows


# ─── Quick sanity test ─────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(7)

    print("SORT tracker sanity test")
    print("Scenario: one cat walks right for 40 frames, disappears for")
    print("5 frames (detector miss), then reappears on its old path.\n")

    tracker = SORTTracker(max_age=10, min_hits=3, iou_threshold=0.3)

    def cat_box(t):
        """Ground-truth-ish cat box moving right, 12 px/frame."""
        cx = 100 + 12 * t
        cy = 200
        return np.array([cx - 40, cy - 30, cx + 40, cy + 30], dtype=float)

    seen_ids = {}
    for t in range(50):
        if 20 <= t < 25:                       # occluded → no detections
            dets, scs = np.zeros((0, 4)), None
        else:
            gt = cat_box(t)
            jitter = rng.normal(0, 1.5, 4)     # small detector noise
            dets = np.clip((gt + jitter)[None, :], 0, None)
            scs = np.array([0.9])

        out = tracker.update(dets, scs)
        for o in out:
            seen_ids.setdefault(o["id"], []).append(t)

    assert len(seen_ids) == 1, \
        f"Expected exactly ONE persistent track, got {sorted(seen_ids)}"

    tid = next(iter(seen_ids))
    frames = seen_ids[tid]
    pre_miss = [t for t in frames if t < 20]
    post_miss = [t for t in frames if t >= 25]
    assert len(pre_miss) >= 15, "track should confirm before occlusion"
    assert post_miss and post_miss[0] <= 27, \
        "Kalman prediction should carry the ID through the 5-frame gap"

    print(f"  Single track #{tid} persisted across the occlusion:")
    print(f"    first seen frame {frames[0]}, last seen frame {frames[-1]}")
    print(f"    frames reported: {len(frames)} / {50 - 5} visible frames")

    # ID-switch test: two cats walking side by side must stay separate
    tracker2 = SORTTracker(max_age=5, min_hits=2, iou_threshold=0.3)
    id_sets = {}
    for t in range(30):
        box_a = np.array([[100 + 8 * t - 30, 100 - 25,
                           100 + 8 * t + 30, 100 + 25]], dtype=float)
        box_b = np.array([[100 + 8 * t - 30, 300 - 25,
                           100 + 8 * t + 30, 300 + 25]], dtype=float)
        out = tracker2.update(np.concatenate([box_a, box_b]), None)
        for o in out:
            id_sets.setdefault(o["id"], set()).add(
                "A" if abs(o["box"][1] - 75) < 150 else "B"
            )

    assert len(id_sets) == 2, \
        f"Two parallel cats should get two distinct IDs, got {len(id_sets)}"
    for tid_, who in id_sets.items():
        assert len(who) == 1, f"track #{tid_} switched identities: {who}"
    print("  Two side-by-side cats kept distinct IDs (no switches)")

    print("\nsort.py sanity checks passed!")
