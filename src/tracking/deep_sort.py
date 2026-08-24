"""
deep_sort.py — DeepSORT: SORT + Appearance Features
====================================================
Pure SORT fails when cats cross paths or pass behind each other: during
the crossing, IoU alone cannot tell "which detection belongs to which
track", so IDs get swapped.  DeepSORT (Wojke et al., 2017,
https://arxiv.org/abs/1703.07402) fixes this by adding an *appearance*
cue: a small CNN embeds each detection crop into a 128-D vector that is
(spatially) stable for the same individual over time.

Per-frame pipeline (extends SORT):

  1. PREDICT   Kalman predict for all tracks          (inherited)
  2. EXTRACT   appearance embedding for every detection (this module)
  3. CASCADE MATCHING:
       a. Appearance stage — cost combines cosine distance to the track's
          feature gallery with the Mahalanobis motion distance:

              d(i, j) = λ · d_mahalanobis(i, j) + (1 − λ) · d_cosine(r_i, a_j)

          Pairs are FORBIDDEN unless both:
            - Mahalanobis d² ≤ χ² gate   (physically plausible motion)
            - cosine distance ≤ threshold (looks like the same cat)
       b. IoU stage — leftover tracks/detections fall back to plain SORT
          matching (handles new objects and badly-lit detections).
  4. UPDATE    matched tracks + append their embedding to the gallery
               (last K features kept; match against the MINIMUM distance,
               which tolerates gradual appearance drift like turning heads).
  5. LIFECYCLE unmatched detections → new tracks, stale tracks → deleted.

The appearance extractor is a small ResNet-flavored CNN (~1M params):
Input crop [B, 3, 128, 128] → conv stages → global average pooling →
Linear → 128-D → L2-normalized output.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.tracking.kalman_filter import (
    CHI2INV95,
    KalmanBoxFilter,
    xyxy_to_cxcyah,
)
from src.tracking.hungarian import assign
from src.tracking.sort import Track, SORTTracker
from src.utils.boxes import compute_iou
from src.data.augmentations import IMAGENET_MEAN, IMAGENET_STD


# ─── Appearance feature extractor ──────────────────────────────────
class ConvBNAct(nn.Module):
    """Conv2d → BatchNorm → LeakyReLU (same recipe as our backbone)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AppearanceExtractor(nn.Module):
    """
    Small CNN producing a 128-D appearance embedding from a detection crop.

    Architecture (input 3×128×128):
        ConvBNAct(3→32, s1)      → 32×128×128
        ConvBNAct(32→64, s2)     → 64×64×64
        ConvBNAct(64→128, s2)    → 128×32×32
        ConvBNAct(128→256, s2)   → 256×16×16
        ConvBNAct(256→256, s2)   → 256×8×8
        Global average pool      → 256
        Linear(256→128)          → 128-D embedding
        (L2 normalization applied at extraction time)

    ~1M parameters — cheap enough to run on every detection of every frame
    without meaningfully slowing down inference.

    NOTE: This extractor works out-of-the-box with random weights (embeddings
    are still spatially consistent enough to help), and gets substantially
    better after contrastive/triplet fine-tuning on cat crops (stretch goal).
    """

    INPUT_SIZE = 128  # crops are resized to 128×128 before the CNN

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 32, stride=1),
            ConvBNAct(32, 64, stride=2),     # 128 → 64
            ConvBNAct(64, 128, stride=2),    # 64  → 32
            ConvBNAct(128, 256, stride=2),   # 32  → 16
            ConvBNAct(256, 256, stride=2),   # 16  → 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): [B, 3, H, W] normalized crop batch

        Returns:
            Tensor: [B, embedding_dim] embeddings (NOT yet L2-normalized)
        """
        x = self.features(x)
        x = self.pool(x).flatten(1)          # [B, 256]
        return self.fc(x)                    # [B, embedding_dim]

    @torch.no_grad()
    def extract(self, frame_bgr: np.ndarray, boxes_xyxy) -> np.ndarray:
        """
        Crop detections out of a video frame and embed them.

        Args:
            frame_bgr (ndarray): H×W×3 uint8 BGR frame (OpenCV convention)
            boxes_xyxy (ndarray): [N, 4] boxes in frame pixel coordinates

        Returns:
            ndarray: [N, embedding_dim], L2-normalized float32 embeddings
                     (shape [0, D] when there are no boxes)
        """
        boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        if boxes.shape[0] == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        H, W = frame_bgr.shape[:2]
        size = self.INPUT_SIZE

        crops = []
        for x1, y1, x2, y2 in boxes:
            # Clamp the crop to image bounds and guard against degenerate
            # zero-area boxes from edge-touching detections
            xa, xb = int(max(0, round(x1))), int(min(W, max(round(x2), round(x1) + 1)))
            ya, yb = int(max(0, round(y1))), int(min(H, max(round(y2), round(y1) + 1)))
            crop = frame_bgr[ya:yb, xa:xb]

            if crop.size == 0:
                crop = np.zeros((size, size, 3), dtype=np.uint8)
            elif crop.shape[0] != size or crop.shape[1] != size:
                crop = cv2.resize(crop, (size, size))

            # BGR → RGB, scale, ImageNet-normalize (matches training pipeline)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - np.asarray(IMAGENET_MEAN)) / np.asarray(IMAGENET_STD)
            crops.append(rgb.transpose(2, 0, 1))          # HWC → CHW

        batch = torch.from_numpy(np.stack(crops)).float()

        device = next(self.parameters()).device
        emb = self.forward(batch.to(device))
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # L2 normalize

        return emb.cpu().numpy().astype(np.float32)


# ─── Appearance-aware track ────────────────────────────────────────
class AppearanceTrack(Track):
    """
    A SORT Track that also remembers what its cat LOOKS like.

    The gallery stores the last `budget` embeddings observed for this track.
    Matching uses the MINIMUM cosine distance across the gallery (per the
    DeepSORT paper): if ANY stored view of this cat resembles the new
    detection, it counts as a match. This tolerates pose/lighting changes
    that would break matching against a single averaged feature.
    """

    def __init__(self, bbox_xyxy, score: float = 1.0,
                 feature=None, budget: int = 100):
        super().__init__(bbox_xyxy, score)
        self.feature_budget = budget
        self.features = []                       # list of [D] ndarrays
        if feature is not None:
            f = np.asarray(feature, dtype=np.float64).ravel()
            if f.size > 0:
                self.features.append(f)

    def add_feature(self, feature):
        """Append a fresh embedding; trim gallery to the budget."""
        f = np.asarray(feature, dtype=np.float64).ravel()
        if f.size == 0 or not np.all(np.isfinite(f)):
            return
        self.features.append(f)
        if len(self.features) > self.feature_budget:
            self.features.pop(0)                 # drop oldest view

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 1.0                           # undefined → maximal distance
        return float(1.0 - np.dot(a, b) / (na * nb))

    def appearance_distance(self, feature, mode: str = "min") -> float:
        """
        Distance between a detection embedding and this track's gallery.

        Args:
            feature (ndarray): [D] detection embedding
            mode    (str):     'min' (DeepSORT default), 'mean' or 'ema'

        Returns:
            float in [0, 2]; 0 = identical appearance
        """
        f = np.asarray(feature, dtype=np.float64).ravel()
        if len(self.features) == 0:
            return 1.0
        dists = [self._cosine_distance(g, f) for g in self.features]
        if mode == "mean":
            return float(np.mean(dists))
        return float(np.min(dists))


# ─── DeepSORT tracker ──────────────────────────────────────────────
class DeepSORTTracker(SORTTracker):
    """
    SORT + cascaded appearance matching. Drop-in replacement:

        tracker = DeepSORTTracker(extractor=AppearanceExtractor())
        outputs = tracker.update(detections, scores, frame=frame_bgr)

    Extra Args (on top of SORT's max_age/min_hits/iou_threshold):
        extractor         (nn.Module): AppearanceExtractor instance (or None
                                       to run in pure-SORT fallback mode via
                                       externally supplied `features`)
        device            (str):       where the extractor runs
        embedding_budget  (int):       gallery length per track (K in paper)
        max_cosine_distance (float):   appearance gate — pairs further apart
                                       than this are never matched (paper: 0.2)
        lambda_mahalanobis (float):     λ weight of the Mahalanobis term in
                                        the stage-1 cost (md formula)
        mahalanobis_gate   (float):     χ² threshold on squared Mahalanobis
                                        distance (95% quantile, dof=4)
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        extractor: nn.Module = None,
        device: str = "cpu",
        embedding_budget: int = 100,
        max_cosine_distance: float = 0.25,
        lambda_mahalanobis: float = 0.5,
        mahalanobis_gate: float = CHI2INV95[4],
    ):
        super().__init__(max_age, min_hits, iou_threshold)

        self.extractor = extractor.to(device) if extractor is not None else None
        self.embedding_budget   = embedding_budget
        self.max_cosine_distance = max_cosine_distance
        self.lambda_mahalanobis  = lambda_mahalanobis
        self.mahalanobis_gate    = mahalanobis_gate

        # Stateless math helper for Mahalanobis distances
        self._kf_math = KalmanBoxFilter()

        # Embeddings for the CURRENT frame, set by update() before the
        # parent class invokes _associate()
        self._frame_features = None

    # ── Public API ────────────────────────────────────────────────
    def update(self, detections, scores=None, frame=None, features=None):
        """
        Process one video frame.

        Args:
            detections (ndarray): [N, 4] xyxy boxes from the detector
            scores     (ndarray): [N] confidences
            frame      (ndarray): BGR uint8 video frame — used to extract
                                  appearance embeddings when `features` is
                                  not given
            features   (ndarray): optional precomputed [N, D] embeddings
                                  (skips the internal extractor)

        Returns:
            list[dict]: same format as SORTTracker.update()
        """
        detections = np.asarray(detections, dtype=np.float64).reshape(-1, 4)

        if features is not None:
            self._frame_features = np.asarray(features, dtype=np.float64)
        elif frame is not None and self.extractor is not None \
                and detections.shape[0] > 0:
            self._frame_features = self.extractor.extract(frame, detections)
        else:
            self._frame_features = None

        return super().update(detections, scores)

    # ── Extension-point implementations ───────────────────────────
    def _new_track(self, bbox_xyxy, score: float, det_idx: int = None):
        feature = None
        if self._frame_features is not None and det_idx is not None \
                and det_idx < len(self._frame_features):
            feature = self._frame_features[det_idx]
        return AppearanceTrack(bbox_xyxy, score, feature,
                               budget=self.embedding_budget)

    def _on_track_matched(self, track, det_idx: int):
        if isinstance(track, AppearanceTrack) \
                and self._frame_features is not None \
                and det_idx < len(self._frame_features):
            track.add_feature(self._frame_features[det_idx])

    # ── Cascaded association ──────────────────────────────────────
    def _associate(self, detections: np.ndarray, track_boxes: np.ndarray):
        """
        Stage 1: appearance cost (Mahalanobis + cosine) with double gating.
        Stage 2: IoU fallback for everything left over.
        """
        num_trks = len(self.tracks)
        num_dets = detections.shape[0]

        if num_trks == 0 or num_dets == 0:
            return [], list(range(num_dets)), list(range(num_trks))

        feats = self._frame_features

        # ── Stage 1: appearance-based matching ────────────────────
        matches_s1 = []
        if feats is not None and feats.shape[0] == num_dets:
            cost = np.zeros((num_trks, num_dets))
            valid = np.zeros((num_trks, num_dets), dtype=bool)

            centers = xyxy_to_cxcyah(detections)
            for ti, trk in enumerate(self.tracks):
                # Motion plausibility of every detection for this track [num_dets]
                maha_sq = self._kf_math.gating_distance(
                    trk.mean, trk.covariance, centers, gated=False,
                )
                maha_norm = np.clip(maha_sq / self.mahalanobis_gate, 0.0, 1.0)

                for dj in range(num_dets):
                    cos_d = trk.appearance_distance(feats[dj])
                    # md formula: d = λ·d_mahalanobis + (1−λ)·d_cosine
                    cost[ti, dj] = (
                        self.lambda_mahalanobis * maha_norm[dj]
                        + (1.0 - self.lambda_mahalanobis) * cos_d
                    )
                    valid[ti, dj] = (
                        maha_sq[dj] <= self.mahalanobis_gate
                        and cos_d <= self.max_cosine_distance
                    )

            matches_s1, unmatched_d, unmatched_t = assign(cost, valid_mask=valid)
        else:
            # No appearance info available → everything goes to stage 2
            unmatched_d = list(range(num_dets))
            unmatched_t = list(range(num_trks))

        # ── Stage 2: IoU fallback on leftovers ────────────────────
        if unmatched_t and unmatched_d:
            rem_track_boxes = track_boxes[unmatched_t]
            rem_detections  = detections[unmatched_d]

            iou_matrix = compute_iou_matrix(rem_track_boxes, rem_detections)
            sub_cost = 1.0 - iou_matrix
            sub_valid = iou_matrix >= self.iou_threshold

            sub_matches, rem_unmatched_d, rem_unmatched_t = assign(
                sub_cost, valid_mask=sub_valid,
            )

            # Map local indices back to original indices
            for local_d, local_t in sub_matches:
                matches_s1.append((unmatched_d[local_d], unmatched_t[local_t]))
            unmatched_d = [unmatched_d[i] for i in rem_unmatched_d]
            unmatched_t = [unmatched_t[i] for i in rem_unmatched_t]

        return matches_s1, unmatched_d, unmatched_t


# ─── Shared helper ─────────────────────────────────────────────────
def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """NumPy IoU matrix between two box sets ([N,4] vs [M,4])."""
    return compute_iou(
        torch.as_tensor(np.asarray(boxes_a), dtype=torch.float32),
        torch.as_tensor(np.asarray(boxes_b), dtype=torch.float32),
    ).numpy()


# ─── Quick sanity test ─────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(3)

    print("DeepSORT sanity tests")

    # Test 1: appearance extractor output shape + unit norm
    ext = AppearanceExtractor().eval()
    dummy_frame = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    dummy_boxes = np.array([[10., 20., 110., 120.],
                            [300., 200., 420., 310.]])
    embs = ext.extract(dummy_frame, dummy_boxes)
    assert embs.shape == (2, 128), f"bad embedding shape {embs.shape}"
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"not unit norm: {norms}"
    empty = ext.extract(dummy_frame, np.zeros((0, 4)))
    assert empty.shape == (0, 128)
    print("  Test 1 (extractor shapes + L2 norm): OK "
          f"({sum(p.numel() for p in ext.parameters()):,} params)")

    # Test 2: two cats CROSS paths in the same lane — identity must survive.
    # Cat A walks left→right, Cat B walks right→left, identical sizes, so
    # around the midpoint IoU alone is ambiguous. We shuffle detection order
    # every frame so the tracker cannot cheat via input ordering.
    tracker = DeepSORTTracker(
        max_age=5, min_hits=2, iou_threshold=0.25,
        max_cosine_distance=0.4, lambda_mahalanobis=0.5,
    )

    eA = np.zeros(128); eA[0] = 1.0        # cat A's constant "appearance"
    eB = np.zeros(128); eB[1] = 1.0        # cat B's constant "appearance"

    def catA_box(t):
        cx = 80 + 12 * t                   # moves right
        return np.array([cx - 40, 70., cx + 40, 130.])

    def catB_box(t):
        cx = 440 - 12 * t                  # moves left
        return np.array([cx - 40, 70., cx + 40, 130.])

    id_to_cx = {}
    for t in range(35):
        boxes = np.stack([
            catA_box(t) + rng.normal(0, 0.8, 4),
            catB_box(t) + rng.normal(0, 0.8, 4),
        ])
        order = rng.permutation(2)         # shuffle which det comes first
        feats = np.stack([eA, eB])[order]
        out = tracker.update(boxes[order], None, features=feats)
        for o in out:
            id_to_cx.setdefault(o["id"], []).append(float(o["box"][0]))

    assert len(id_to_cx) == 2, \
        f"expected 2 persistent tracks through the crossing, got {len(id_to_cx)}"
    for tid_, cxs in id_to_cx.items():
        steps = np.diff(np.asarray(cxs))
        # Each cat moves at a constant ±12 px/frame. An identity swap would
        # flip the direction mid-sequence, mixing +12 and -12 steps.
        assert np.all(steps > 5) or np.all(steps < -5), \
            f"track #{tid_} flipped direction → identity swap suspected ({steps})"
    print(f"  Test 2 (crossing cats keep IDs): OK — "
          f"{sorted(id_to_cx)} kept constant velocity through the crossing")

    # Test 3: gallery mechanics — budget trimming & min-distance matching
    gallery_track = AppearanceTrack(np.array([0., 0., 50., 50.]), 0.9,
                                    feature=eA, budget=3)
    for k in range(6):                     # push 6 views into a 3-slot gallery
        v = np.zeros(128); v[k % 4] = 1.0
        gallery_track.add_feature(v)
    assert len(gallery_track.features) == 3, "gallery exceeded its budget"
    assert gallery_track.appearance_distance(eA) < 0.01, \
        "original appearance should still match via the retained view"
    far = np.zeros(128); far[127] = 1.0
    assert gallery_track.appearance_distance(far) > 1.0 - 1e-9, \
        "orthogonal appearance should be maximally distant"
    print("  Test 3 (gallery budget + min-distance): OK")

    print("\ndeep_sort.py sanity checks passed!")
