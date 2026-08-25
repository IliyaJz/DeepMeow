"""
optimize.py — Inference Optimization (Week 6, Day 3–4)
=======================================================
Speeds up the video inference path without changing what is detected.
Targets: ≥ 30 FPS at 416×416 on a modern GPU, ≥ 5 FPS on CPU.

Optimizations implemented (one class, composable flags):
  1. TorchScript trace of the conv stack (backbone → FPN → heads) —
     fuses ~200 eager module calls into one scripted graph
  2. FP16 weights + input on CUDA — ~2× conv throughput on tensor cores
  3. Anchors precomputed ONCE — DeepMeowDetector.predict() regenerates
     10,647 anchor rows on the CPU for every single frame; they are
     constant per input resolution
  4. Fused decode — all three scales decoded in one vectorized pass on
     the GPU, confidence-thresholded there; only surviving boxes cross
     to CPU (predict() moves the whole [A,4] anchor set per scale)
  5. Pre-allocated input buffer — persistent CUDA tensor filled with
     copy_() instead of a fresh allocation per frame

Equivalence contract: detect() returns the same boxes/scores as the
reference path (preprocess → predict → scale_boxes_back), because it
replays the exact same math: same anchor row ordering (row r of the
flattened [H,W,3,6] predictions pairs with row r of generate_anchors'
(a, y, x)-ordered anchor list — a fixed permutation the trained weights
have absorbed), same sigmoid/scale/NMS semantics.

Usage:
    fast = OptimizedPredictor(model, device="cuda", half=True)
    boxes, scores = fast.detect(frame_bgr, conf_threshold=0.25)

    timing = benchmark_detect(fast.detect, frame_bgr)
"""

import time

import numpy as np
import torch
import torch.nn as nn

from src.models.detector import DeepMeowDetector
from src.tracking.demo import preprocess_frame
from src.utils.boxes import decode_boxes, generate_anchors, nms


class _ConvStack(nn.Module):
    """backbone → FPN → heads as a single traceable module (raw preds)."""

    def __init__(self, model: DeepMeowDetector):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.head

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        f3, f4, f5 = self.neck(p3, p4, p5)
        return self.head(f3, f4, f5)


class OptimizedPredictor:
    """
    Fast detection wrapper around a trained DeepMeowDetector.

    Args:
        model   (DeepMeowDetector): the detector (already loaded weights)
        device  (str):  'cuda' / 'cpu'
        half    (bool): FP16 weights+input (CUDA only; ignored on CPU)
        trace   (bool): TorchScript-trace the conv stack
        warmup  (int):  warm-up forward passes run at construction so the
                        first real frame doesn't pay CUDA init costs
    """

    def __init__(self, model: DeepMeowDetector, device: str = "cpu",
                 half: bool = False, trace: bool = True, warmup: int = 3):
        self.device = device
        self.input_size = model.input_size
        self.num_classes = model.num_classes
        self.half = bool(half and device == "cuda")
        was_training = model.training
        model.eval()  # freeze BN statistics before tracing

        conv_stack = _ConvStack(model).to(device).eval()

        # ── 1. TorchScript trace of the conv stack ────────────────
        example = torch.zeros(1, 3, self.input_size, self.input_size,
                              device=device)
        if trace:
            with torch.no_grad():
                self.conv = torch.jit.trace(conv_stack, example)
        else:
            self.conv = conv_stack

        # ── 2. FP16 ───────────────────────────────────────────────
        if self.half:
            self.conv = self.conv.half()

        # ── 3. Anchors cached once (identical ordering to predict) ─
        self._anchors = []
        for key, size in [("f3", None), ("f4", None), ("f5", None)]:
            H = self.input_size // model.STRIDES[key]
            anchors = generate_anchors(
                H, model.ANCHOR_SIZES[key], model.STRIDES[key],
                device=device,
            ).float()
            self._anchors.append(anchors)

        # ── 5. Pre-allocated input buffer ─────────────────────────
        buf_dtype = torch.float16 if self.half else torch.float32
        self._in_buf = torch.zeros(1, 3, self.input_size, self.input_size,
                                   device=device, dtype=buf_dtype)

        # Warm-up (also triggers any lazy CUDA work)
        with torch.no_grad():
            for _ in range(max(0, warmup)):
                self.conv(self._in_buf)
        if device == "cuda":
            torch.cuda.synchronize()

        if was_training:
            model.train()

    @torch.no_grad()
    def detect(self, frame_bgr, conf_threshold: float = 0.25,
               iou_threshold: float = 0.45):
        """
        BGR frame → (boxes [N,4] float32 xyxy @frame scale, scores [N]).

        Same contract as CatDetectorTracker.detect(); labels are omitted
        because the project is single-class (every detection is 'cat').
        """
        # 5. preprocess into the persistent buffer (H2D + cast in one op)
        self._in_buf.copy_(preprocess_frame(frame_bgr, self.input_size))

        # 1+2. scripted (possibly fp16) forward: raw [B,H,W,3,6] per scale
        pred3, pred4, pred5 = self.conv(self._in_buf)

        # 4. fused decode: threshold on GPU, keep only survivors
        kept_boxes, kept_scores = [], []
        for preds, anchors in zip((pred3, pred4, pred5), self._anchors):
            p = preds[0].reshape(-1, 6).float()          # (h, w, a) row order
            boxes = decode_boxes(p[:, :4], anchors)
            obj = torch.sigmoid(p[:, 4])
            cls = torch.sigmoid(p[:, 5:])
            conf = (obj.unsqueeze(1) * cls).max(dim=1).values
            mask = conf >= conf_threshold
            if mask.any():
                kept_boxes.append(boxes[mask])
                kept_scores.append(conf[mask])

        if not kept_boxes:
            return (torch.zeros((0, 4)), torch.zeros(0))

        boxes = torch.cat(kept_boxes, dim=0)
        scores = torch.cat(kept_scores, dim=0)

        keep = nms(boxes, scores, iou_threshold=iou_threshold)
        boxes, scores = boxes[keep], scores[keep]

        # map back to original frame pixels (same math as scale_boxes_back)
        h, w = frame_bgr.shape[:2]
        scale = boxes.new_tensor([w / self.input_size, h / self.input_size,
                                  w / self.input_size, h / self.input_size])
        return boxes * scale, scores


# ─── Benchmarking helper ──────────────────────────────────────────────
def benchmark_detect(predict_fn, frame_bgr, n: int = 50, warmup: int = 10) -> dict:
    """
    Time a detect(frame_bgr) callable end to end (preprocess → device →
    decode → CPU transfer), which is what video playback actually pays.

    Returns {'mean_ms', 'fps', 'n'}.
    """
    is_cuda = torch.cuda.is_available()
    for _ in range(warmup):
        predict_fn(frame_bgr)
    if is_cuda:
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n):
        predict_fn(frame_bgr)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    mean_ms = elapsed / n * 1000.0
    return {"mean_ms": round(mean_ms, 2), "fps": round(1000.0 / mean_ms, 1),
            "n": n}


# ─── Quick sanity test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("OptimizedPredictor sanity tests")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    model = DeepMeowDetector(num_classes=1, input_size=416).eval()

    frame = np.random.default_rng(0).integers(
        0, 255, size=(360, 640, 3), dtype=np.uint8)

    # Reference path: preprocess → predict → scale back (what demo.py does)
    from src.tracking.demo import scale_boxes_back

    def reference_detect(frame_bgr, conf=0.01, iou=0.45):
        inp = preprocess_frame(frame_bgr, model.input_size).to(device)
        with torch.no_grad():
            dets = model.predict(inp, conf_threshold=conf,
                                 iou_threshold=iou)[0]
        boxes = dets["boxes"].cpu().numpy()
        if boxes.shape[0] > 0:
            boxes = scale_boxes_back(boxes, frame_bgr.shape, model.input_size)
        return boxes, dets["scores"].cpu().numpy()

    # Test 1: fp32 traced predictor must match the reference path closely
    fast32 = OptimizedPredictor(model, device=device, half=False, trace=True)
    rb, rs = reference_detect(frame)
    fb, fs = fast32.detect(frame, conf_threshold=0.01)

    assert rb.shape[0] == fb.shape[0], \
        f"detection count mismatch: reference {rb.shape[0]} vs fast {fb.shape[0]}"
    if rb.shape[0] > 0:
        assert np.allclose(rb, fb.cpu().numpy(), atol=1e-3), \
            "box coordinates diverge between reference and traced path"
        assert np.allclose(rs, fs.cpu().numpy(), atol=1e-3), \
            "scores diverge between reference and traced path"
    print(f"  Test 1 (fp32 trace == reference): OK "
          f"({rb.shape[0]} detections, max |Δbox| "
          f"{np.abs(rb - fb.cpu().numpy()).max() if rb.shape[0] else 0:.2e})")

    # Test 2: fp16 runs and stays numerically sane (looser tolerance)
    if device == "cuda":
        fast16 = OptimizedPredictor(model, device=device, half=True)
        hb, hs = fast16.detect(frame, conf_threshold=0.01)
        assert hb.shape[0] == rb.shape[0], \
            f"fp16 detection count {hb.shape[0]} != fp32 {rb.shape[0]}"
        if rb.shape[0] > 0:
            assert np.abs(rs - hs.cpu().numpy()).max() < 0.05, \
                "fp16 scores drifted too far from fp32"
        print("  Test 2 (fp16 sanity): OK")

    # Test 3: speed comparison
    variants = [("reference (eager fp32)", lambda f: reference_detect(f)),
                ("optimized (traced fp32)", lambda f: fast32.detect(f, 0.01))]
    if device == "cuda":
        fast16 = OptimizedPredictor(model, device=device, half=True)
        variants.append(("optimized (traced fp16)", lambda f: fast16.detect(f, 0.01)))

    print(f"  Test 3 (speed, {640}x{360} frame):")
    for name, fn in variants:
        t = benchmark_detect(fn, frame, n=30)
        print(f"    {name:26s} {t['mean_ms']:7.2f} ms  ~{t['fps']:6.1f} FPS")

    print("optimize.py sanity checks passed!")
