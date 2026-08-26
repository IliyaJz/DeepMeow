"""
clips.py — Demo Clip Resolver
=============================
Ensures a demo video exists at the expected path so downstream sections
(tracked demo, inference pipeline, benchmarks) can run unconditionally.

Resolution order:
  1. Use the file if it already exists (repo checkout or previous run)
  2. Copy the first video found on Google Drive (Colab; mounts if needed)
  3. Download from YouTube via yt-dlp (when `yt_url` is given)
  4. Synthesize a placeholder clip so the pipeline always runs
"""

import glob
import os
import shutil
import subprocess


def _first_drive_video():
    """First video found anywhere in MyDrive (mounts Drive if needed)."""
    try:
        from google.colab import drive          # Colab-only import
    except ImportError:
        return None

    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")

    hits = (glob.glob("/content/drive/MyDrive/*.mp4")
            + glob.glob("/content/drive/MyDrive/**/*.mp4", recursive=True)
            + glob.glob("/content/drive/MyDrive/**/*.mov", recursive=True)
            + glob.glob("/content/drive/MyDrive/**/*.avi", recursive=True))
    hits = sorted(set(hits))
    if not hits:
        print("No video files found anywhere in MyDrive.")
        return None

    print("Videos found on Drive:")
    for i, h in enumerate(hits):
        print(f"  [{i}] {h}")
    return hits[0]                              # <- change index to pick another


def _synthesize(path, width=640, height=360, frames=120):
    """Two moving 'cats' (ellipses) on noise — no data or weights needed."""
    import cv2
    import numpy as np

    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                         (width, height))
    rng = np.random.default_rng(0)
    for t in range(frames):
        frame = np.full((height, width, 3), 40, np.uint8)
        for cx0, cy0, vx, vy, col in [(150, 180, 3, -1, (80, 180, 240)),
                                      (500, 120, -2, 1, (60, 200, 120))]:
            cx, cy = cx0 + vx * t, cy0 + vy * t
            cv2.ellipse(frame, (int(cx), int(cy)), (45, 30), 0, 0, 360, col, -1)
            cv2.circle(frame, (int(cx) + 38, int(cy) - 28), 14, col, -1)
            cv2.circle(frame, (int(cx) + 44, int(cy) - 31), 2, (30, 30, 30), -1)
        frame += rng.integers(0, 12, frame.shape, dtype=np.uint8)
        vw.write(frame)
    vw.release()


def ensure_clip(path="data/videos/cats.mp4", yt_url=""):
    """
    Guarantee a readable video at `path`; return its absolute path.

    Args:
        path   (str): destination the demo expects
        yt_url (str): optional YouTube URL downloaded via yt-dlp when the
                      file is missing and Drive has nothing to offer
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if os.path.exists(path):
        print(f"Using existing video: {path}")
    elif yt_url:
        subprocess.run(["pip", "install", "-q", "yt-dlp"], check=True)
        subprocess.run(["yt-dlp", "-f", "bestvideo[ext=mp4]/best",
                        "-o", path, yt_url], check=True)
    else:
        src = _first_drive_video()
        if src:
            shutil.copy(src, path)
            print(f"Copied {src} -> {path}")
        else:
            _synthesize(path)
            print(f"Synthesized placeholder clip: {path}")

    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Ready: {os.path.abspath(path)} ({n} frames, {w}x{h})")
    return os.path.abspath(path)


# ─── Quick sanity test ────────────────────────────────────────────────
if __name__ == "__main__":
    out = ensure_clip("results/_clip_resolve_test.mp4")
    assert os.path.exists(out)
    os.remove(out)
    print("clips.py sanity checks passed!")
