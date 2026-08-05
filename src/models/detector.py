"""
detector.py — Full Detection Model
=====================================
This module wires together all components we built in Week 1 and Week 2
into a single end-to-end model.

Full pipeline:
  Input image [B, 3, 416, 416]
      ↓
  Backbone  → (P3, P4, P5)  multi-scale feature maps
      ↓
  FPN Neck  → (F3, F4, F5)  semantically enriched, uniform 256-channel maps
      ↓
  Detection Head → (pred3, pred4, pred5)  raw anchor predictions per scale
      ↓
  [During training]  → DetectionLoss returns scalar loss + breakdown dict
  [During inference] → Decode + NMS → final bounding boxes

Usage:
  # Training
  model = DeepMeowDetector(num_classes=1)
  loss, breakdown = model(images, targets)  # returns loss

  # Inference
  model.eval()
  boxes, scores, labels = model.predict(images, conf_threshold=0.25)
"""

import torch
import torch.nn as nn

from src.models.backbone import Backbone
from src.models.neck import FPN
from src.models.head import MultiScaleHead
from src.losses.detection_loss import DetectionLoss
from src.utils.boxes import generate_anchors, decode_boxes, nms


class DeepMeowDetector(nn.Module):
    """
    DeepMeow: End-to-end single-shot cat detector.

    Architecture:
      Backbone (custom ResNet-style)  →  40M params
      FPN Neck (top-down pathway)     →  ~1M params
      Multi-Scale Detection Head      →  ~3M params
      Total:                          ~44M params

    All components are defined in:
      src/models/backbone.py
      src/models/neck.py
      src/models/head.py
      src/losses/detection_loss.py
    """

    # Anchor sizes and strides — must match detection_loss.py
    ANCHOR_SIZES = {
        "f3": [[10, 13], [16, 30], [33, 23]],
        "f4": [[30, 61], [62, 45], [59, 119]],
        "f5": [[116, 90], [156, 198], [373, 326]],
    }
    STRIDES = {"f3": 8, "f4": 16, "f5": 32}
    FEATURE_MAP_SIZES = {"f3": 52, "f4": 26, "f5": 13}

    def __init__(
        self,
        num_classes: int = 1,
        input_size: int = 416,
        fpn_channels: int = 256,
        num_anchors_per_scale: int = 3,
        lambda_box: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_cls: float = 0.5,
    ):
        """
        Args:
            num_classes           (int):   Number of object categories (1 = cat only)
            input_size            (int):   Input image resolution (must be multiple of 32)
            fpn_channels          (int):   Output channels for each FPN scale
            num_anchors_per_scale (int):   Number of anchor boxes per grid cell
            lambda_box/obj/cls    (float): Loss weighting factors
        """
        super().__init__()

        self.num_classes  = num_classes
        self.input_size   = input_size

        # ── Sub-modules ────────────────────────────────────────────
        self.backbone = Backbone()
        self.neck     = FPN(out_channels=fpn_channels)
        self.head     = MultiScaleHead(
            in_channels=fpn_channels,
            num_anchors_per_scale=num_anchors_per_scale,
            num_classes=num_classes,
        )

        # ── Loss function (used during training) ───────────────────
        self.criterion = DetectionLoss(
            num_classes=num_classes,
            input_size=input_size,
            lambda_box=lambda_box,
            lambda_obj=lambda_obj,
            lambda_cls=lambda_cls,
        )

    def forward(self, images: torch.Tensor, targets: list = None):
        """
        Forward pass — behaves differently in train vs eval mode.

        Training (targets provided):
            Returns (total_loss, loss_dict)

        Inference (targets=None):
            Returns raw predictions (pred3, pred4, pred5)
            Use model.predict() for post-processed boxes.

        Args:
            images  (Tensor): [B, 3, H, W] normalized image batch
            targets (list):   List of B dicts: {'boxes': [N,4], 'labels': [N]}
                              Pass None during inference.

        Returns:
            If training: (loss_tensor, loss_breakdown_dict)
            If inference: (pred3, pred4, pred5) raw prediction tuples
        """
        device = images.device

        # ── 1. Backbone: extract multi-scale features ──────────────
        p3, p4, p5 = self.backbone(images)

        # ── 2. FPN Neck: enrich features with top-down context ─────
        f3, f4, f5 = self.neck(p3, p4, p5)

        # ── 3. Detection Head: produce raw anchor predictions ───────
        pred3, pred4, pred5 = self.head(f3, f4, f5)

        # ── 4a. Training: compute loss ─────────────────────────────
        if self.training and targets is not None:
            loss, loss_dict = self.criterion(
                predictions=(pred3, pred4, pred5),
                targets=targets,
                device=device,
            )
            return loss, loss_dict

        # ── 4b. Inference: return raw predictions ──────────────────
        return pred3, pred4, pred5

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Run inference and return cleaned-up detections.

        Steps:
          1. Forward pass → raw predictions (pred3, pred4, pred5)
          2. Decode offsets → absolute bounding boxes
          3. Apply confidence threshold (filter low-score predictions)
          4. Apply NMS (remove duplicate detections)

        Args:
            images         (Tensor): [B, 3, H, W] normalized images
            conf_threshold (float):  Minimum objectness * class probability
            iou_threshold  (float):  NMS IoU threshold

        Returns:
            List of B dicts, each containing:
              'boxes'  (Tensor): [K, 4] detected boxes [x1,y1,x2,y2]
              'scores' (Tensor): [K]    confidence scores
              'labels' (Tensor): [K]    predicted class indices
        """
        self.eval()
        device = images.device

        pred3, pred4, pred5 = self.forward(images)

        batch_results = []

        for b in range(images.shape[0]):
            all_boxes, all_scores, all_labels = [], [], []

            # Process each FPN scale
            for scale_key, preds in [("f3", pred3), ("f4", pred4), ("f5", pred5)]:
                H, W = preds.shape[1], preds.shape[2]
                stride = self.STRIDES[scale_key]
                anchor_list = self.ANCHOR_SIZES[scale_key]

                # Generate anchors for this scale
                anchors = generate_anchors(H, anchor_list, stride, device=device)
                # anchors: [H*W*3, 4]

                # Flatten this image's predictions: [H*W*3, 6]
                pred_b = preds[b].view(-1, 4 + 1 + self.num_classes)

                # Decode box offsets → absolute coordinates
                boxes = decode_boxes(pred_b[:, :4], anchors)  # [A, 4]

                # Compute confidence = sigmoid(objectness) * sigmoid(class)
                objectness   = torch.sigmoid(pred_b[:, 4])          # [A]
                class_scores = torch.sigmoid(pred_b[:, 5:])         # [A, C]
                conf, labels = (objectness.unsqueeze(1) * class_scores).max(dim=1)

                # Filter by confidence threshold
                keep_mask = conf >= conf_threshold
                if keep_mask.sum() == 0:
                    continue

                boxes  = boxes[keep_mask]
                conf   = conf[keep_mask]
                labels = labels[keep_mask]

                all_boxes.append(boxes)
                all_scores.append(conf)
                all_labels.append(labels)

            if len(all_boxes) == 0:
                # No detections for this image
                batch_results.append({
                    "boxes":  torch.zeros((0, 4), device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                })
                continue

            all_boxes  = torch.cat(all_boxes,  dim=0)   # [total_A, 4]
            all_scores = torch.cat(all_scores, dim=0)   # [total_A]
            all_labels = torch.cat(all_labels, dim=0)   # [total_A]

            # Apply NMS to remove duplicate detections
            keep = nms(all_boxes, all_scores, iou_threshold=iou_threshold)

            batch_results.append({
                "boxes":  all_boxes[keep],
                "scores": all_scores[keep],
                "labels": all_labels[keep],
            })

        return batch_results

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Quick sanity test ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Building DeepMeow detector...")
    model = DeepMeowDetector(num_classes=1, input_size=416)
    total = model.count_parameters()
    print(f"Total trainable parameters: {total:,}")

    # ── Test 1: Training forward pass ──────────────────────────────
    print("\n[Training mode test]")
    model.train()

    dummy_images = torch.randn(2, 3, 416, 416)

    # Fake targets: image 0 has 2 cats, image 1 has 1 cat
    dummy_targets = [
        {
            "boxes":  torch.tensor([[50., 30., 150., 120.],
                                    [200., 100., 300., 250.]]),
            "labels": torch.tensor([0, 0]),
        },
        {
            "boxes":  torch.tensor([[80., 60., 200., 180.]]),
            "labels": torch.tensor([0]),
        },
    ]

    loss, loss_dict = model(dummy_images, dummy_targets)
    print(f"  Total loss:  {loss_dict['total']:.4f}")
    print(f"  Box loss:    {loss_dict['box']:.4f}")
    print(f"  Object loss: {loss_dict['obj']:.4f}")
    print(f"  Class loss:  {loss_dict['cls']:.4f}")

    # ── Test 2: Inference forward pass ─────────────────────────────
    print("\n[Inference mode test]")
    results = model.predict(dummy_images, conf_threshold=0.01)

    for i, r in enumerate(results):
        print(f"  Image {i}: {r['boxes'].shape[0]} detections")

    print("\ndetector.py sanity check passed!")
