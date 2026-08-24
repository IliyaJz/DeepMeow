"""
demo.py — End-to-End Tracking Demo (Detector + DeepSORT on Video)
===================================================================
Glue script for the Week 5 deliverable: run the DeepMeow detector frame
by frame over a cat video, feed detections through the DeepSORT tracker,
and render an annotated output video with persistent track IDs + motion
trails.

Pipeline per frame:
    BGR frame ──▶ letterbox-resize to 416×416 ──▶ DeepMeowDetector.predict
             ──▶ [x1,y1,x2,y2, score] detections ──▶ DeepSORTTracker.update
             ──▶ draw boxes + IDs + trails ──▶ write to output video

Usage (from the repo root):
    python src/tracking/demo.py \
        --video data/videos/cats.mp4 \
        --checkpoint checkpoints/best.pth \
        --out results/tracked_demo.mp4

If you don't have a trained checkpoint yet, pass `--random-weights` to
verify the plumbing end-to-end (detections will be garbage but tracking
logic still runs).

The tracker keeps IDs across short occlusions; each box shows
"cat #<id>" and a fading trail of recent positions.
"""

import argparse
import sys
import time
from pathlib import Path

# Allow launching from any working directory:
# put the repo root on sys.path so `from src...` resolves.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np
import torch

from src.models.detector import DeepMeowDetector
from src.tracking.deep_sort import AppearanceExtractor, DeepSORTTracker


# ─── Frame preprocessing (must mirror training normalization) ────────
def preprocess_frame(frame_bgr: np.ndarray, input_size: int = 416) -> torch.Tensor:
    """
    Resize a BGR video frame to the network input and normalize it.

    Uses plain resize (not letterbox) because training images were resized
    the same way in augmentations.py — consistency beats elegance here.

    Args:
        frame_bgr  (ndarray): H×W×3 uint8 OpenCV frame
        input_size (int):     target square resolution (multiple of 32)

    Returns:
        Tensor: [1, 3, S, S] normalized RGB tensor
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (input_size, input_size))
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)  # [1,3,H,W]


def scale_boxes_back(boxes: np.ndarray, frame_shape, input_size: int) -> np.ndarray:
    """Map boxes predicted at input_size back onto original frame pixels."""
    h, w = frame_shape[:2]
    scale = np.array([w / input_size, h / input_size,
                      w / input_size, h / input_size])
    return boxes * scale


# ─── Drawing helpers ─────────────────────────────────────────────────
PALETTE = [
    (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
    (0, 114, 178), (213, 94, 0), (204, 121, 167), (100, 100, 100),
]


def color_for(track_id: int):
    return PALETTE[track_id % len(PALETTE)]


def draw_tracks(frame: np.ndarray, outputs: list, fps: float) -> np.ndarray:
    """Draw boxes, IDs, confidence and motion trails onto a frame copy."""
    canvas = frame.copy()
    H, W = canvas.shape[:2]

    for out in outputs:
        x1, y1, x2, y2 = out["box"].astype(int)
        tid = out["id"]
        color = color_for(tid)

        # Motion trail (last ~30 centers), oldest = dimmest
        trail = out.get("trail")
        if trail is not None and len(trail) > 1:
            pts = np.stack([
                ((trail[:, 0] + trail[:, 2]) / 2).astype(int),
                ((trail[:, 1] + trail[:, 3]) / 2).astype(int),
            ], axis=1)
            for k in range(1, len(pts)):
                alpha = k / len(pts)                     # fade-in with age
                c = tuple(int(ch * alpha) for ch in color)
                cv2.line(canvas, tuple(pts[k - 1]), tuple(pts[k]), c, 2)

        # Box + label
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"cat #{tid} {out['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_label = max(y1 - 6, th + 4)
        cv2.rectangle(canvas, (x1, y_label - th - 4), (x1 + tw + 4, y_label + 2),
                      color, -1)
        cv2.putText(canvas, label, (x1 + 2, y_label - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    # FPS overlay
    cv2.putText(canvas, f"{fps:.1f} FPS", (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2, cv2.LINE_AA)
    return canvas


# ─── Main loop ────────────────────────────────────────────────────────
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Detector ────────────────────────────────────────────────
    model = DeepMeowDetector(num_classes=1, input_size=416)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt["ema_model_state"] if (
            isinstance(ckpt, dict) and "ema_model_state" in ckpt and args.use_ema
        ) else ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
    elif args.random_weights:
        print("WARNING: running with RANDOM WEIGHTS (--random-weights); "
              "detections will be meaningless but the pipeline is testable.")
    else:
        raise SystemExit("Provide --checkpoint PATH or --random-weights.")
    model.to(device).eval()

    # ── 2. Tracker with appearance extractor ───────────────────────
    extractor = AppearanceExtractor().to(device).eval()
    tracker = DeepSORTTracker(
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.track_iou,
        extractor=extractor,
        device=device,
    )

    # ── 3. Video I/O ───────────────────────────────────────────────
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps_in   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        (width, height),
    )
    print(f"Input: {args.video} ({width}x{height} @ {fps_in:.0f} fps, "
          f"{total} frames)")
    print(f"Output: {args.out}\n")

    frame_idx, t_start = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_idx >= args.max_frames):
            break

        # Detect at 416×416, then map boxes back to original resolution.
        # t0 starts before preprocessing so the FPS overlay reflects the
        # FULL per-frame pipeline (preprocess + detect + track), not just
        # the tracker.
        t0 = time.time()
        inp = preprocess_frame(frame, model.input_size).to(device)
        with torch.no_grad():
            dets = model.predict(inp,
                                 conf_threshold=args.conf,
                                 iou_threshold=args.nms_iou)[0]

        det_boxes = dets["boxes"].cpu().numpy()
        det_scores = dets["scores"].cpu().numpy()

        if det_boxes.shape[0] > 0:
            det_boxes = scale_boxes_back(det_boxes, frame.shape, model.input_size)

        # Track (appearance embeddings are extracted from the full frame)
        outputs = tracker.update(det_boxes, det_scores, frame=frame)
        dt = time.time() - t0

        fps_now = 1.0 / max(dt, 1e-6)
        canvas = draw_tracks(frame, outputs, fps_now)
        writer.write(canvas)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}/{total} | "
                  f"{len(outputs)} active tracks | {fps_now:.1f} FPS")

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    print(f"\nDone: {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / max(elapsed, 1e-6):.1f} FPS overall incl. I/O)")
    print(f"Tracked video saved to: {args.out}")


# ─── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepMeow Week-5 demo: detect cats and track them "
                    "with DeepSORT (persistent IDs + trails)")
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--out", default="results/tracked_demo.mp4",
                        help="output annotated video path")
    parser.add_argument("--checkpoint", default=None,
                        help="trained detector checkpoint (.pth)")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="prefer EMA weights from the checkpoint")
    parser.add_argument("--random-weights", action="store_true",
                        help="run untrained detector (plumbing test only)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="detector confidence threshold")
    parser.add_argument("--nms-iou", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--max-age", type=int, default=30,
                        help="frames a track survives without detection")
    parser.add_argument("--min-hits", type=int, default=3,
                        help="hits before a track ID is reported")
    parser.add_argument("--track-iou", type=float, default=0.3,
                        help="IoU fallback association threshold")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="process only the first N frames (0 = all)")

    run(parser.parse_args())
