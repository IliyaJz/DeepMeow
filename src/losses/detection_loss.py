"""
detection_loss.py — Multi-Task Detection Loss
===============================================
This module defines the loss functions that train our detector.

Detection training is a multi-task problem. We need to simultaneously:
  1. Teach the model to localize objects accurately (box regression loss)
  2. Teach the model to score "is there an object here?" (objectness loss)
  3. Teach the model to classify objects correctly (classification loss)

Total loss:
    L = lambda_box * L_box + lambda_obj * L_obj + lambda_cls * L_cls

where:
  L_box: CIoU loss between predicted and ground-truth bounding boxes
         (computed only for POSITIVE anchors — anchors matched to a GT box)

  L_obj: Focal loss for objectness binary classification
         Positive anchors target = 1 (object here)
         Negative anchors target = 0 (background)
         Ignore anchors (IoU in 0.4-0.5 range) are skipped

  L_cls: Binary cross-entropy for class prediction
         (computed only for positive anchors)

Key challenge: extreme class imbalance
  In a 52x52 grid with 3 anchors per cell, there are 8,112 negative anchors
  for every 1-3 positive anchors (matching actual cats in the image).
  Without correction, the model learns to always predict "no object".
  Solution: Focal Loss, which down-weights easy negatives automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.boxes import (
    generate_anchors, decode_boxes, compute_ciou, match_anchors_to_gt, encode_boxes
)


# ─── Focal Loss ───────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification (Lin et al., 2017).

    Standard binary cross-entropy treats all examples equally:
        BCE = -[y * log(p) + (1-y) * log(1-p)]

    But with ~99% negatives (background), the model is overwhelmed
    by easy negative examples and ignores the rare positive examples.

    Focal Loss down-weights easy examples with a modulating factor (1 - p_t)^gamma:
        FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        p_t = p     if y == 1 (positive example, correct prediction confidence)
              1-p   if y == 0 (negative example)
        alpha_t = alpha   for positives
                  1-alpha for negatives (balance pos/neg contribution)
        gamma = focusing parameter (typically 2.0):
                high gamma = more weight on hard examples
                gamma = 0  = standard BCE

    When the model is already confident (p_t near 1), (1-p_t)^gamma is tiny,
    so those easy examples contribute almost nothing to the total loss.
    Hard examples (uncertain predictions) still get full weight.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha (float): Balance factor for positive/negative samples.
                           0.25 = positives get 0.25 weight, negatives get 0.75
            gamma (float): Focusing parameter. gamma=0 → standard BCE.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs  (Tensor): Raw logits (before sigmoid), shape [N]
            targets (Tensor): Binary targets (0.0 or 1.0), shape [N]

        Returns:
            loss (Tensor): Scalar focal loss value
        """
        # Compute standard BCE but keep individual values (reduction='none')
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # p_t: probability of the correct class
        p   = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)

        # alpha_t: per-sample class balance weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Modulating factor: down-weights easy examples
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Final focal loss
        focal = alpha_t * modulating_factor * bce
        return focal.mean()


# ─── Main Detection Loss ──────────────────────────────────────────
class DetectionLoss(nn.Module):
    """
    Multi-task detection loss that combines:
      - CIoU box regression loss (positive anchors only)
      - Focal objectness loss (all anchors, ignoring the 0.4-0.5 zone)
      - Binary cross-entropy classification loss (positive anchors only)

    This loss is computed independently for each of the 3 FPN scales
    and then summed together.
    """

    # Anchor sizes (in pixels at 416x416 input) from configs/default.yaml
    # Each scale has 3 anchor sizes: (width, height)
    ANCHOR_SIZES = {
        # For F3 (52x52), stride=8 — small objects
        "f3": [[10, 13], [16, 30], [33, 23]],
        # For F4 (26x26), stride=16 — medium objects
        "f4": [[30, 61], [62, 45], [59, 119]],
        # For F5 (13x13), stride=32 — large objects
        "f5": [[116, 90], [156, 198], [373, 326]],
    }
    STRIDES = {"f3": 8, "f4": 16, "f5": 32}

    def __init__(self, num_classes: int = 1, input_size: int = 416,
                 lambda_box: float = 5.0, lambda_obj: float = 1.0,
                 lambda_cls: float = 0.5):
        """
        Args:
            num_classes (int):    Number of object classes (1 for cat-only)
            input_size  (int):    Input image resolution (416)
            lambda_box  (float):  Weight for bounding box regression loss
            lambda_obj  (float):  Weight for objectness loss
            lambda_cls  (float):  Weight for classification loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_size  = input_size
        self.lambda_box  = lambda_box
        self.lambda_obj  = lambda_obj
        self.lambda_cls  = lambda_cls

        self.focal_loss  = FocalLoss(alpha=0.25, gamma=2.0)
        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def _compute_scale_loss(
        self,
        predictions: torch.Tensor,   # [B, H, W, num_anchors, 6]
        targets: list,                # list of B target dicts {boxes:[N,4], labels:[N]}
        scale_key: str,               # "f3", "f4", or "f5"
        device: torch.device,
    ):
        """
        Compute loss for a single FPN scale.

        Args:
            predictions: Raw head output for this scale
            targets:     List of per-image target dicts
            scale_key:   Which scale we're computing ("f3"/"f4"/"f5")
            device:      Torch device

        Returns:
            loss_box, loss_obj, loss_cls — scalar tensors for this scale
        """
        batch_size = predictions.shape[0]
        H, W       = predictions.shape[1], predictions.shape[2]
        num_anchors = predictions.shape[3]
        stride      = self.STRIDES[scale_key]
        anchor_list = self.ANCHOR_SIZES[scale_key]

        # Generate anchors for this scale
        # anchors: [H*W*num_anchors, 4] in [x1,y1,x2,y2] absolute pixels
        anchors = generate_anchors(
            feature_map_size=H,
            anchor_sizes=anchor_list,
            stride=stride,
            device=device,
        )  # [H*W*3, 4]

        # Flatten predictions: [B, H*W*num_anchors, 6]
        # .contiguous() is called here as a defensive measure — permute() inside the
        # head reorders dimensions without moving data in memory, which can cause
        # .view() to fail. Calling .contiguous() forces a memory copy into
        # a standard row-major layout before we reshape.
        preds_flat = predictions.contiguous().view(batch_size, -1, 4 + 1 + self.num_classes)

        total_box = torch.tensor(0.0, device=device)
        total_obj = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)

        for i in range(batch_size):
            pred_i   = preds_flat[i]       # [A, 6], A = H*W*num_anchors
            gt_boxes = targets[i]["boxes"].to(device)   # [G, 4]
            gt_labels = targets[i]["labels"].to(device)  # [G]

            # ── Match each anchor to a GT box (or background) ──────
            # labels: [A] — which GT index each anchor is matched to
            #         -1  = background (negative)
            #         -2  = ignore zone
            #         >=0 = matched GT index
            labels, max_iou = match_anchors_to_gt(anchors, gt_boxes)

            # ── Masks ───────────────────────────────────────────────
            pos_mask = labels >= 0    # [A] bool — positive anchors
            neg_mask = labels == -1   # [A] bool — negative anchors

            # ── Objectness targets ──────────────────────────────────
            # Positive → target 1.0, Negative → target 0.0, Ignore → skipped
            obj_targets = torch.zeros(anchors.shape[0], device=device)
            obj_targets[pos_mask] = 1.0

            # We compute objectness loss only on pos + neg anchors (skip ignore zone)
            valid_obj_mask = pos_mask | neg_mask
            if valid_obj_mask.sum() > 0:
                loss_obj_i = self.focal_loss(
                    pred_i[valid_obj_mask, 4],    # predicted objectness logits
                    obj_targets[valid_obj_mask],  # 0/1 targets
                )
                total_obj = total_obj + loss_obj_i

            # ── Box regression & classification (positives only) ────
            num_pos = pos_mask.sum().item()
            if num_pos > 0:
                # Get the anchors and predictions for positive anchors
                pos_anchors = anchors[pos_mask]          # [P, 4]
                pos_preds   = pred_i[pos_mask]           # [P, 6]
                pos_gt_idx  = labels[pos_mask]           # [P] — which GT box

                # Get the corresponding GT boxes
                matched_gt  = gt_boxes[pos_gt_idx]       # [P, 4]

                # ── Box regression loss ─────────────────────────────
                # Decode the predicted offsets to absolute box coordinates
                decoded_boxes = decode_boxes(pos_preds[:, :4], pos_anchors)  # [P, 4]

                # CIoU between predicted and matched GT boxes
                ciou = compute_ciou(decoded_boxes, matched_gt)  # [P]
                loss_box_i = (1.0 - ciou).mean()  # CIoU loss = 1 - CIoU
                total_box  = total_box + loss_box_i

                # ── Classification loss ─────────────────────────────
                # For single-class detection, target is always 1.0 for cat
                # (we already know it's a cat — the objectness head filters bg)
                cls_targets = torch.zeros(
                    num_pos, self.num_classes, device=device
                )
                # Set the correct class index to 1.0
                for j, gt_idx in enumerate(pos_gt_idx):
                    cls_targets[j, gt_labels[gt_idx]] = 1.0

                loss_cls_i = self.cls_loss_fn(
                    pos_preds[:, 5:],  # class logits [P, num_classes]
                    cls_targets,       # one-hot targets [P, num_classes]
                )
                total_cls = total_cls + loss_cls_i

        # Average over batch
        return (
            total_box / batch_size,
            total_obj / batch_size,
            total_cls / batch_size,
        )

    def forward(self, predictions: tuple, targets: list, device: torch.device):
        """
        Compute total detection loss across all 3 FPN scales.

        Args:
            predictions (tuple): (pred3, pred4, pred5) from MultiScaleHead
                                  Each is [B, H, W, 3, 6]
            targets     (list):  List of B dicts, each {'boxes': [N,4], 'labels': [N]}
            device      (device): Torch device

        Returns:
            total_loss (Tensor): Scalar — weighted sum of all losses
            loss_dict  (dict):   Breakdown of individual losses for logging
        """
        pred3, pred4, pred5 = predictions

        # Compute loss for each FPN scale
        box3, obj3, cls3 = self._compute_scale_loss(pred3, targets, "f3", device)
        box4, obj4, cls4 = self._compute_scale_loss(pred4, targets, "f4", device)
        box5, obj5, cls5 = self._compute_scale_loss(pred5, targets, "f5", device)

        # Sum across scales
        loss_box = box3 + box4 + box5
        loss_obj = obj3 + obj4 + obj5
        loss_cls = cls3 + cls4 + cls5

        # Weighted sum (lambda weights from config)
        total_loss = (
            self.lambda_box * loss_box +
            self.lambda_obj * loss_obj +
            self.lambda_cls * loss_cls
        )

        # Return breakdown for logging / debugging
        loss_dict = {
            "total": total_loss.item(),
            "box":   loss_box.item(),
            "obj":   loss_obj.item(),
            "cls":   loss_cls.item(),
        }

        return total_loss, loss_dict
