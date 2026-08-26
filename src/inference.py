"""
inference.py — Reusable End-to-End Inference Pipeline (Week 6, Day 1–2)
========================================================================
Class-based version of tracking/demo.py: the full frame → detect → track
→ render loop wrapped in a CatDetectorTracker object that can be driven
programmatically instead of through a CLI.

    pipeline = CatDetectorTracker(checkpoint="checkpoints/best.pt")
    outputs, canvas = pipeline.step(frame)               # streaming use
    summary = pipeline.run_video("in.mp4", "out.mp4")   # whole clip

Per-frame flow (reuses demo.py helpers so both entry points stay in
sync by construction):
    BGR frame ──▶ resize + normalize ──▶ DeepMeowDetector.predict ──▶
    rescale boxes ──▶ DeepSORTTracker.update ──▶ boxes/IDs/trails/FPS

Extras over demo.py:
    - step() streaming API (webcams / live loops / custom loops)
    - conf_threshold & nms_iou adjustable between frames ("slider")
    - run_video() returns a summary dict (frames, FPS, peak tracks)
    - reset() to start a fresh tracking session on a new video
"""

import sys
import time
from pathlib import Path

# Allow launching from any working directory:
# put the repo root on sys.path so `from src...` resolves.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np
import torch

from src.models.detector import DeepMeowDetector
from src.optimize import OptimizedPredictor
from src.tracking.demo import draw_tracks, preprocess_frame, scale_boxes_back
from src.tracking.deep_sort import AppearanceExtractor, DeepSORTTracker


class CatDetectorTracker:
    """
    End-to-end cat detection + tracking pipeline.

    Args:
        checkpoint     (str):  path to a trained .pt checkpoint; None runs
                               random weights (plumbing smoke test only).
        device         (str):  'cuda' / 'cpu'; auto-detected when None.
        conf_threshold (float): detector confidence threshold ("slider" —
                               safe to change between frames).
        nms_iou        (float): NMS IoU threshold.
        max_detections (int):  keep only the top-K highest-score detections
                               per frame before tracking. Protects the tracker
                               from pathological false-positive floods (e.g.
                               random-weight smoke tests) that would turn the
                               O(tracks × detections) matching into minutes
                               per frame. Real clips have ≪ 100 cats/frame.
        tracker_kwargs (dict): forwarded to DeepSORTTracker, e.g.
                               {'max_age': 30, 'min_hits': 3,
                                'iou_threshold': 0.3}.
        use_ema        (bool): prefer EMA weights inside the checkpoint.
        optimize       (bool): use the optimized detector path (TorchScript
                               trace + cached anchors + fused decode, see
                               src/optimize.py) — same detections, faster.
        half           (bool): FP16 weights/input on CUDA (optimize path);
                               None = auto (on for CUDA, off for CPU).
        trace          (bool): TorchScript-trace the conv stack.
    """

    def __init__(self, checkpoint=None, device=None,
                 conf_threshold=0.25, nms_iou=0.45, max_detections=100,
                 tracker_kwargs=None, use_ema=True,
                 optimize=False, half=None, trace=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── Detector ──────────────────────────────────────────────
        self.model = DeepMeowDetector(num_classes=1, input_size=416)
        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location=self.device)
            state = ckpt["ema_model_state"] if (
                isinstance(ckpt, dict) and "ema_model_state" in ckpt and use_ema
            ) else ckpt.get("model_state", ckpt)
            self.model.load_state_dict(state)
            print(f"Loaded checkpoint: {checkpoint}")
        else:
            print("WARNING: no checkpoint given -> random weights "
                  "(detections will be meaningless)")
        self.model.to(self.device).eval()

        # ── Tracker (+ appearance branch) ─────────────────────────
        self._tracker_kwargs = dict(tracker_kwargs or {})
        extractor = AppearanceExtractor().to(self.device).eval()
        self.tracker = DeepSORTTracker(
            extractor=extractor, device=self.device, **self._tracker_kwargs)

        # Runtime-adjustable knobs: assign any time, takes effect next frame
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou
        self.max_detections = int(max_detections)

        # ── Optional optimized detector path (Week 6, Day 3–4) ────
        # Wraps the SAME loaded model; conf/nms thresholds are passed
        # per call so the "slider" keeps working in fast mode too.
        if half is None:
            half = self.device == "cuda"
        self._fast = OptimizedPredictor(
            self.model, device=self.device,
            half=half, trace=trace,
        ) if optimize else None

        self._fps = 0.0

    # ── Detection ────────────────────────────────────────────────────
    def _cap_detections(self, boxes, scores):
        """Keep only the top-K highest-score detections (K = max_detections)."""
        if boxes.shape[0] > self.max_detections:
            top = np.argsort(scores)[::-1][:self.max_detections]
            boxes, scores = boxes[top], scores[top]
        return boxes, scores

    def detect(self, frame_bgr: np.ndarray):
        """BGR frame → (boxes [N,4] xyxy @frame scale, scores [N])."""
        if self._fast is not None:
            boxes, scores = self._fast.detect(
                frame_bgr, self.conf_threshold, self.nms_iou)
            return self._cap_detections(
                boxes.cpu().numpy(), scores.cpu().numpy())

        inp = preprocess_frame(frame_bgr, self.model.input_size).to(self.device)
        with torch.no_grad():
            dets = self.model.predict(
                inp,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.nms_iou)[0]
        boxes = dets["boxes"].cpu().numpy()
        scores = dets["scores"].cpu().numpy()
        if boxes.shape[0] > 0:
            boxes = scale_boxes_back(boxes, frame_bgr.shape, self.model.input_size)
        return self._cap_detections(boxes, scores)

    # ── Streaming API: one frame in → tracks (+canvas) out ──────────
    def step(self, frame_bgr: np.ndarray, draw: bool = True):
        """
        Process a single frame.

        Returns:
            (outputs, canvas):
                outputs: list of active-track dicts
                         ({'id', 'box', 'score', 'trail'})
                canvas:  annotated BGR copy, or None when draw=False
        """
        t0 = time.time()
        boxes, scores = self.detect(frame_bgr)
        outputs = self.tracker.update(boxes, scores, frame=frame_bgr)
        dt = time.time() - t0

        inst = 1.0 / max(dt, 1e-6)
        self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst

        canvas = draw_tracks(frame_bgr, outputs, self._fps) if draw else None
        return outputs, canvas

    @property
    def fps(self) -> float:
        """Smoothed per-frame processing rate (detect + track + draw)."""
        return self._fps

    def reset(self):
        """Drop all track state (call when switching to a new video)."""
        extractor = AppearanceExtractor().to(self.device).eval()
        self.tracker = DeepSORTTracker(
            extractor=extractor, device=self.device, **self._tracker_kwargs)
        self._fps = 0.0

    # ── Whole-video convenience ──────────────────────────────────────
    def run_video(self, video_path, output_path,
                  max_frames: int = 0, verbose: bool = True) -> dict:
        """
        Annotate an entire video file.

        Args:
            video_path / output_path: input clip and annotated mp4 destination
            max_frames: process only the first N frames (0 = all)
            verbose: print progress every 50 frames

        Returns:
            summary dict: {frames, elapsed_s, fps, peak_tracks, output}
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {video_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in, (width, height))

        n, t_start, peak = 0, time.time(), 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and n >= max_frames):
                break
            outputs, canvas = self.step(frame, draw=True)
            writer.write(canvas)
            n += 1
            peak = max(peak, len(outputs))
            if verbose and n % 50 == 0:
                print(f"  frame {n}/{total} | {len(outputs)} active tracks "
                      f"| {self.fps:.1f} FPS")

        cap.release()
        writer.release()
        elapsed = time.time() - t_start
        summary = {
            "frames": n,
            "elapsed_s": round(elapsed, 2),
            "fps": round(n / max(elapsed, 1e-6), 1),
            "peak_tracks": peak,
            "output": str(output_path),
        }
        if verbose:
            print(f"Done: {n} frames in {elapsed:.1f}s "
                  f"({summary['fps']} FPS overall incl. I/O)")
            print(f"Tracked video saved to: {output_path}")
        return summary


# ─── Quick sanity test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("CatDetectorTracker smoke test (random weights, synthetic clip)")

    # Synthetic two-blob clip so the test never needs data or weights
    clip = Path("results/_smoke_clip.mp4")
    clip.parent.mkdir(parents=True, exist_ok=True)
    W, H, N = 320, 240, 40
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
    rng = np.random.default_rng(7)
    for t in range(N):
        frame = np.full((H, W, 3), 50, np.uint8)
        cx = 60 + 4 * t
        cv2.ellipse(frame, (cx, 120), (25, 18), 0, 0, 360, (90, 200, 240), -1)
        frame += rng.integers(0, 10, frame.shape, dtype=np.uint8)
        vw.write(frame)
    vw.release()

    pipeline = CatDetectorTracker(checkpoint=None)
    summary = pipeline.run_video(str(clip), "results/_smoke_tracked.mp4",
                                 verbose=False)

    assert summary["frames"] == N, f"processed {summary['frames']}/{N} frames"
    assert Path(summary["output"]).exists(), "no output video written"
    assert pipeline.fps > 0, "FPS estimate not updated"
    print(f"  OK: {summary}")

    # Same clip through the optimized path (traced; fp16 only on CUDA)
    pipeline.reset()
    fast = CatDetectorTracker(checkpoint=None, optimize=True)
    fast_summary = fast.run_video(str(clip), "results/_smoke_tracked_fast.mp4",
                                  verbose=False)
    assert fast_summary["frames"] == N, "optimized path dropped frames"
    print(f"  OK (optimize=True): {fast_summary}")

    clip.unlink(missing_ok=True)
    print("inference.py smoke checks passed!")
