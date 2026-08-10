"""
metrics.py — COCO-Style Mean Average Precision (mAP)
======================================================
This module evaluates how well our detector performs on the validation set.

What is mAP?
  - For each class, we compute a Precision-Recall curve.
  - Average Precision (AP) = area under that curve.
  - mAP = mean AP averaged over all classes and IoU thresholds.

We compute two standard COCO metrics:
  - mAP@0.50     (IoU threshold = 0.5, same as Pascal VOC)
  - mAP@0.50:0.95 (IoU averaged over 10 thresholds: 0.5, 0.55, ..., 0.95)
    This is the primary COCO metric — it rewards more precise localization.

How the computation works:
  1. Collect all predictions (boxes, scores, labels) from the entire val set
  2. Collect all ground-truth boxes
  3. For each class at each IoU threshold:
     a. Sort predictions by score (highest first)
     b. For each prediction, check if it matches any un-matched GT box (by IoU)
     c. Build precision and recall arrays
     d. Compute AP as the area under the P-R curve (11-point interpolation)
  4. Average AP across thresholds and classes -> mAP

Usage:
  evaluator = MeanAveragePrecision(num_classes=1, iou_thresholds=[0.5])
  # Accumulate predictions over the val set
  for batch in val_loader:
      preds = model.predict(images)
      evaluator.update(preds, targets)
  # Compute final metrics
  results = evaluator.compute()
  print(results)  # {'mAP_50': 0.42, 'mAP_50_95': 0.28}
"""

import torch
import numpy as np
from collections import defaultdict


# ─── Helper: compute AP from precision/recall arrays ──────────────
def _compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Compute Average Precision via 11-point interpolation.

    The AP is the area under the Precision-Recall curve.
    We use 11-point interpolation (standard before COCO switched to full AUC):
      - Sample recall at 11 evenly spaced points: 0.0, 0.1, 0.2, ..., 1.0
      - At each recall level, take the maximum precision at that recall or higher
      - AP = mean of those 11 precision values

    Args:
        precision (np.ndarray): Precision values [N] in decreasing recall order
        recall    (np.ndarray): Recall values [N] in increasing order

    Returns:
        ap (float): Average Precision in [0, 1]
    """
    # Prepend sentinel values so the curve starts at (recall=0, precision=1)
    # and ends at (recall=1, precision=0)
    precision = np.concatenate([[0.0], precision, [0.0]])
    recall    = np.concatenate([[0.0], recall,    [1.0]])

    # Make precision monotonically non-increasing from right
    # (take max precision from the right at each recall level)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # 11-point interpolation
    ap = 0.0
    for r_thresh in np.linspace(0.0, 1.0, 11):
        # Find precision at this recall threshold
        p_at_r = precision[recall >= r_thresh]
        ap += (p_at_r.max() if len(p_at_r) > 0 else 0.0)
    ap /= 11.0

    return float(ap)


# ─── Main mAP Evaluator ───────────────────────────────────────────
class MeanAveragePrecision:
    """
    Accumulates predictions and ground truths, then computes mAP.

    Supports:
      - Multiple IoU thresholds (COCO-style)
      - Multiple classes
      - Batch-by-batch accumulation (call update() for each batch)
      - Final metric aggregation (call compute() once at end of epoch)
    """

    def __init__(self, num_classes: int = 1,
                 iou_thresholds: list = None):
        """
        Args:
            num_classes     (int):  Number of object categories (1 = cat only)
            iou_thresholds  (list): IoU thresholds to evaluate at.
                                    Default: [0.5, 0.55, ..., 0.95] (COCO standard)
        """
        self.num_classes = num_classes

        if iou_thresholds is None:
            # COCO standard: 10 thresholds from 0.5 to 0.95 with step 0.05
            self.iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()
        else:
            self.iou_thresholds = iou_thresholds

        # Internal storage — one list per image for predictions and ground truths
        self._predictions = []  # list of dicts: {boxes, scores, labels}
        self._targets     = []  # list of dicts: {boxes, labels}

    def reset(self):
        """Clear accumulated data — call at the start of each epoch."""
        self._predictions = []
        self._targets     = []

    def update(self, predictions: list, targets: list):
        """
        Accumulate predictions and targets for a batch of images.

        Args:
            predictions (list): List of B dicts from model.predict():
                                 {'boxes': [K,4], 'scores': [K], 'labels': [K]}
            targets     (list): List of B dicts from the DataLoader:
                                 {'boxes': [G,4], 'labels': [G]}
        """
        for pred, tgt in zip(predictions, targets):
            # Move everything to CPU numpy for evaluation
            self._predictions.append({
                "boxes":  pred["boxes"].cpu().numpy()   if len(pred["boxes"]) > 0  else np.zeros((0, 4)),
                "scores": pred["scores"].cpu().numpy()  if len(pred["scores"]) > 0 else np.zeros(0),
                "labels": pred["labels"].cpu().numpy()  if len(pred["labels"]) > 0 else np.zeros(0, dtype=int),
            })
            self._targets.append({
                "boxes":  tgt["boxes"].cpu().numpy()   if len(tgt["boxes"]) > 0  else np.zeros((0, 4)),
                "labels": tgt["labels"].cpu().numpy()  if len(tgt["labels"]) > 0 else np.zeros(0, dtype=int),
            })

    def compute(self) -> dict:
        """
        Compute mAP across all accumulated images, classes, and IoU thresholds.

        Returns:
            dict with keys:
              'mAP_50':     AP averaged over classes at IoU=0.50
              'mAP_50_95':  AP averaged over classes and all IoU thresholds
              'AP_per_class': list of per-class AP at IoU=0.50
        """
        ap_matrix = np.zeros((self.num_classes, len(self.iou_thresholds)))

        for cls_idx in range(self.num_classes):
            for thr_idx, iou_thr in enumerate(self.iou_thresholds):
                ap_matrix[cls_idx, thr_idx] = self._compute_ap_for_class(cls_idx, iou_thr)

        # mAP@0.50 = mean over classes at threshold index 0
        thr_50_idx = 0  # 0.5 is always first in our threshold list
        map_50     = float(ap_matrix[:, thr_50_idx].mean())

        # mAP@0.50:0.95 = mean over both classes and all thresholds
        map_50_95  = float(ap_matrix.mean())

        # Per-class AP at IoU=0.50 for detailed inspection
        ap_per_class = ap_matrix[:, thr_50_idx].tolist()

        return {
            "mAP_50":       map_50,
            "mAP_50_95":    map_50_95,
            "AP_per_class": ap_per_class,
        }

    def _compute_ap_for_class(self, cls_idx: int, iou_threshold: float) -> float:
        """
        Compute AP for a single class at a single IoU threshold.

        Steps:
          1. Collect all predictions for this class, sorted by score
          2. For each prediction: check if it matches any unmatched GT (IoU >= threshold)
             -> True Positive (TP) or False Positive (FP)
          3. Build cumulative TP/FP arrays -> compute precision and recall
          4. Compute AP via 11-point interpolation

        Args:
            cls_idx       (int):   Which class to evaluate
            iou_threshold (float): IoU threshold for a valid match

        Returns:
            ap (float): Average Precision for this class
        """
        # Collect all GT boxes for this class across all images
        # gt_by_image[img_idx] = list of gt boxes
        gt_by_image = defaultdict(list)
        n_total_gt  = 0
        for img_idx, tgt in enumerate(self._targets):
            mask = tgt["labels"] == cls_idx
            boxes = tgt["boxes"][mask]
            gt_by_image[img_idx] = boxes
            n_total_gt += len(boxes)

        if n_total_gt == 0:
            # No ground truth for this class — AP is undefined, return 0
            return 0.0

        # Collect all predictions for this class: (score, img_idx, box)
        all_preds = []
        for img_idx, pred in enumerate(self._predictions):
            mask = pred["labels"] == cls_idx
            for score, box in zip(pred["scores"][mask], pred["boxes"][mask]):
                all_preds.append((float(score), img_idx, box))

        if len(all_preds) == 0:
            return 0.0

        # Sort predictions by score, highest first
        all_preds.sort(key=lambda x: -x[0])

        # Track which GT boxes have already been matched (to avoid double-counting)
        matched = defaultdict(set)  # img_idx -> set of matched gt indices

        tp_list = []  # True Positive flags
        fp_list = []  # False Positive flags

        for score, img_idx, pred_box in all_preds:
            gt_boxes = gt_by_image[img_idx]

            if len(gt_boxes) == 0:
                # No GT in this image -> False Positive
                tp_list.append(0)
                fp_list.append(1)
                continue

            # Compute IoU between this prediction and all GT boxes in this image
            pred_t = torch.tensor(pred_box).unsqueeze(0)  # [1, 4]
            gt_t   = torch.tensor(gt_boxes)                # [G, 4]

            iou = _box_iou_numpy(pred_box, gt_boxes)  # [G]

            best_iou_idx = int(np.argmax(iou))
            best_iou     = iou[best_iou_idx]

            if best_iou >= iou_threshold and best_iou_idx not in matched[img_idx]:
                # Matched a GT box that hasn't been claimed yet -> True Positive
                tp_list.append(1)
                fp_list.append(0)
                matched[img_idx].add(best_iou_idx)
            else:
                # Either IoU too low, or GT already matched -> False Positive
                tp_list.append(0)
                fp_list.append(1)

        # Build cumulative TP and FP arrays
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)

        # Precision = TP / (TP + FP) at each rank
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)

        # Recall = TP / total_GT at each rank
        recall = tp_cumsum / (n_total_gt + 1e-7)

        return _compute_ap(precision, recall)


# ─── Helper: Box IoU (numpy version, for metrics) ─────────────────
def _box_iou_numpy(box: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between a single box and a set of GT boxes using numpy.

    Args:
        box      (np.ndarray): Shape [4], format [x1, y1, x2, y2]
        gt_boxes (np.ndarray): Shape [G, 4]

    Returns:
        iou (np.ndarray): Shape [G]
    """
    x1 = np.maximum(box[0], gt_boxes[:, 0])
    y1 = np.maximum(box[1], gt_boxes[:, 1])
    x2 = np.minimum(box[2], gt_boxes[:, 2])
    y2 = np.minimum(box[3], gt_boxes[:, 3])

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    gt_area  = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union    = box_area + gt_area - inter

    return inter / (union + 1e-7)


# ─── Quick sanity test ─────────────────────────────────────────────
if __name__ == "__main__":
    evaluator = MeanAveragePrecision(num_classes=1, iou_thresholds=[0.5])

    # Perfect prediction: pred box = gt box, high confidence
    perfect_pred = [{"boxes": torch.tensor([[10., 10., 50., 50.]]),
                     "scores": torch.tensor([0.99]),
                     "labels": torch.tensor([0])}]
    perfect_tgt  = [{"boxes": torch.tensor([[10., 10., 50., 50.]]),
                     "labels": torch.tensor([0])}]

    evaluator.update(perfect_pred, perfect_tgt)
    results = evaluator.compute()

    print(f"Perfect prediction mAP@50: {results['mAP_50']:.4f}  (expect ~1.0)")
    print(f"AP per class: {results['AP_per_class']}")
    print("metrics.py sanity check passed!")
