"""
mosaic.py — Mosaic & Mixup Data Augmentation
=============================================
This module implements two powerful augmentation strategies used in modern detectors
(popularized by YOLOv4/v5) to improve generalization and detection of small objects.

Why do we need these?
  Standard augmentations (flip, crop, color jitter) work on single images.
  But they don't teach the model to handle:
    - Multiple objects at different scales in the same image
    - Objects appearing in unusual contexts
    - Objects near image borders

Mosaic augmentation:
  Combines 4 training images into a single mosaic image.
  The 4 images are placed in each quadrant around a random center point.
  This forces the model to detect objects that are:
    - Cropped at edges (partial visibility)
    - Shown at 1/4 of normal size (good for small-object detection)
    - Appearing alongside objects from different scenes (diverse context)

  Before mosaic:
    [img1]  [img2]  [img3]  [img4]  <- 4 separate training samples

  After mosaic:
    +-------+-------+
    | img1  | img2  |
    +-------+-------+
    | img3  | img4  |
    +-------+-------+
    <- 1 combined 416x416 image with all 4 sets of GT boxes

Mixup augmentation:
  Linearly blends two images and their labels:
    mixed_image = lambda * img_A + (1 - lambda) * img_B
  where lambda ~ Beta(alpha, alpha), typically alpha = 0.2 ~ 0.4

  This acts as a soft regularization — the model learns to predict
  "partial confidence" for overlapping classes, which improves calibration.
"""

import random
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image


# ─── Mosaic Augmentation ──────────────────────────────────────────
class MosaicAugmentation:
    """
    Applies 4-image Mosaic augmentation to a batch of images and boxes.

    The output is always (output_size x output_size), matching the
    model's expected input resolution.
    """

    def __init__(self, output_size: int = 416):
        """
        Args:
            output_size (int): Side length of the square output image.
                               Should match the detector's input_size (416).
        """
        self.output_size = output_size

    def __call__(self, images: list, targets: list) -> tuple:
        """
        Build one mosaic from 4 (image, target) pairs.

        Args:
            images  (list): List of 4 PIL Images (any size)
            targets (list): List of 4 target dicts:
                              {'boxes': Tensor[N,4] in [x1,y1,x2,y2], 'labels': Tensor[N]}

        Returns:
            mosaic_image (np.ndarray): Combined image [H, W, 3] uint8
            mosaic_target (dict):      Combined {'boxes': Tensor[M,4], 'labels': Tensor[M]}
        """
        assert len(images) == 4, "Mosaic requires exactly 4 images"
        S = self.output_size

        # Pick a random center point in the middle half of the output
        # (avoids placing the center too close to the edge)
        cx = random.randint(S // 4, 3 * S // 4)
        cy = random.randint(S // 4, 3 * S // 4)

        # Initialize the output canvas (gray background)
        mosaic = np.full((S, S, 3), 114, dtype=np.uint8)

        all_boxes  = []
        all_labels = []

        # Each of the 4 images goes into one quadrant
        # Quadrant layout:
        #   0: top-left    1: top-right
        #   2: bottom-left 3: bottom-right
        for idx, (img, tgt) in enumerate(zip(images, targets)):
            # Convert PIL to numpy
            img_np = np.array(img.convert("RGB"))
            ih, iw = img_np.shape[:2]

            # Determine where this image sits on the mosaic canvas
            if idx == 0:    # Top-left: image aligns to (cx, cy) at its bottom-right
                x1c, y1c = cx - iw, cy - ih   # canvas top-left (may be negative)
                x2c, y2c = cx,      cy
            elif idx == 1:  # Top-right: image aligns to (cx, cy) at its bottom-left
                x1c, y1c = cx,      cy - ih
                x2c, y2c = cx + iw, cy
            elif idx == 2:  # Bottom-left: image aligns to (cx, cy) at its top-right
                x1c, y1c = cx - iw, cy
                x2c, y2c = cx,      cy + ih
            else:           # Bottom-right: image aligns to (cx, cy) at its top-left
                x1c, y1c = cx,      cy
                x2c, y2c = cx + iw, cy + ih

            # Clip canvas region to [0, S]
            x1c_clip = max(x1c, 0)
            y1c_clip = max(y1c, 0)
            x2c_clip = min(x2c, S)
            y2c_clip = min(y2c, S)

            # Corresponding crop region in the source image
            x1i = x1c_clip - x1c
            y1i = y1c_clip - y1c
            x2i = x1i + (x2c_clip - x1c_clip)
            y2i = y1i + (y2c_clip - y1c_clip)

            if x2c_clip > x1c_clip and y2c_clip > y1c_clip:
                mosaic[y1c_clip:y2c_clip, x1c_clip:x2c_clip] = \
                    img_np[y1i:y2i, x1i:x2i]

            # Transform bounding boxes to mosaic coordinates
            boxes  = tgt["boxes"].clone().float()   # [N, 4]
            labels = tgt["labels"].clone()           # [N]

            if len(boxes) > 0:
                # Shift boxes by the image's position on the canvas
                boxes[:, 0] += x1c   # x1
                boxes[:, 1] += y1c   # y1
                boxes[:, 2] += x1c   # x2
                boxes[:, 3] += y1c   # y2

                # Clip boxes to the visible canvas area [0, S]
                boxes[:, 0].clamp_(0, S)
                boxes[:, 1].clamp_(0, S)
                boxes[:, 2].clamp_(0, S)
                boxes[:, 3].clamp_(0, S)

                # Remove degenerate boxes (zero or negative area after clipping)
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                valid = (w > 2) & (h > 2)

                if valid.any():
                    all_boxes.append(boxes[valid])
                    all_labels.append(labels[valid])

        if len(all_boxes) > 0:
            mosaic_boxes  = torch.cat(all_boxes,  dim=0)
            mosaic_labels = torch.cat(all_labels, dim=0)
        else:
            mosaic_boxes  = torch.zeros((0, 4))
            mosaic_labels = torch.zeros(0, dtype=torch.long)

        mosaic_target = {"boxes": mosaic_boxes, "labels": mosaic_labels}
        return mosaic, mosaic_target


# ─── Mixup Augmentation ───────────────────────────────────────────
def apply_mixup(img_a: np.ndarray, tgt_a: dict,
                img_b: np.ndarray, tgt_b: dict,
                alpha: float = 0.2) -> tuple:
    """
    Blend two images and merge their ground-truth boxes.

    The blend ratio lambda is sampled from a Beta(alpha, alpha) distribution.
    For alpha=0.2, lambda is usually very close to 0 or 1
    (i.e., one image dominates), which is the desired behavior —
    we want real-looking images, not 50/50 ghost blends.

    Args:
        img_a (np.ndarray): First image  [H, W, 3] uint8
        tgt_a (dict):       First targets  {'boxes': Tensor[N,4], 'labels': Tensor[N]}
        img_b (np.ndarray): Second image [H, W, 3] uint8
        tgt_b (dict):       Second targets {'boxes': Tensor[M,4], 'labels': Tensor[M]}
        alpha (float):      Beta distribution shape parameter (0.2 is standard)

    Returns:
        mixed_img    (np.ndarray): Blended image [H, W, 3] uint8
        mixed_target (dict):       Merged {'boxes': Tensor[N+M,4], 'labels': Tensor[N+M]}
    """
    # Sample blend ratio from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Resize img_b to match img_a's size if needed
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    # Blend pixel values
    mixed = (lam * img_a.astype(np.float32) +
             (1 - lam) * img_b.astype(np.float32)).astype(np.uint8)

    # Merge all GT boxes from both images (no blending on boxes — just concatenate)
    if len(tgt_a["boxes"]) > 0 and len(tgt_b["boxes"]) > 0:
        mixed_boxes  = torch.cat([tgt_a["boxes"],  tgt_b["boxes"]],  dim=0)
        mixed_labels = torch.cat([tgt_a["labels"], tgt_b["labels"]], dim=0)
    elif len(tgt_a["boxes"]) > 0:
        mixed_boxes, mixed_labels = tgt_a["boxes"], tgt_a["labels"]
    elif len(tgt_b["boxes"]) > 0:
        mixed_boxes, mixed_labels = tgt_b["boxes"], tgt_b["labels"]
    else:
        mixed_boxes  = torch.zeros((0, 4))
        mixed_labels = torch.zeros(0, dtype=torch.long)

    return mixed, {"boxes": mixed_boxes, "labels": mixed_labels}


# ─── Quick sanity test ─────────────────────────────────────────────
if __name__ == "__main__":
    from PIL import Image

    # Create 4 dummy images (different solid colors)
    mosaic_aug = MosaicAugmentation(output_size=416)

    images = [Image.fromarray(np.full((200, 200, 3), c, dtype=np.uint8))
              for c in [100, 150, 180, 220]]

    targets = [
        {"boxes": torch.tensor([[20., 20., 80., 80.]]),
         "labels": torch.tensor([0])}
        for _ in range(4)
    ]

    mosaic_img, mosaic_tgt = mosaic_aug(images, targets)

    print(f"Mosaic image shape: {mosaic_img.shape}  (expect (416, 416, 3))")
    print(f"Mosaic boxes shape: {mosaic_tgt['boxes'].shape}")
    print("mosaic.py sanity check passed!")
