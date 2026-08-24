"""
src.tracking — Multi-Object Tracking Engine (Week 5)
=====================================================
Components:
    kalman_filter.KalmanBoxFilter  — 8-D constant-velocity Kalman filter
    hungarian.hungarian_algorithm  — optimal assignment (from scratch)
    hungarian.assign               — gated assignment wrapper
    sort.Track / SORTTracker       — Kalman + IoU + Hungarian tracking
    deep_sort.AppearanceExtractor  — 128-D CNN embeddings for crops
    deep_sort.AppearanceTrack      — Track with an embedding gallery
    deep_sort.DeepSORTTracker      — SORT + cascaded appearance matching
    demo.run                       — end-to-end video inference script

Quick start:
    from src.tracking import DeepSORTTracker, AppearanceExtractor

    tracker = DeepSORTTracker(extractor=AppearanceExtractor())
    outputs = tracker.update(detections_xyxy, scores, frame=bgr_frame)
"""

from src.tracking.kalman_filter import (
    CHI2INV95,
    KalmanBoxFilter,
    cxcyah_to_xyxy,
    xyxy_to_cxcyah,
)
from src.tracking.hungarian import assign, hungarian_algorithm
from src.tracking.sort import SORTTracker, Track
from src.tracking.deep_sort import (
    AppearanceExtractor,
    AppearanceTrack,
    DeepSORTTracker,
)

__all__ = [
    "CHI2INV95",
    "KalmanBoxFilter",
    "xyxy_to_cxcyah",
    "cxcyah_to_xyxy",
    "hungarian_algorithm",
    "assign",
    "Track",
    "SORTTracker",
    "AppearanceExtractor",
    "AppearanceTrack",
    "DeepSORTTracker",
]
