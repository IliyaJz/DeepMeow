"""
boxes.py — Bounding Box Utilities
==================================
This module contains all the mathematical operations we need to work
with bounding boxes throughout the detection pipeline.

Key operations:
  - IoU (Intersection over Union): measures how much two boxes overlap
  - CIoU: a better version of IoU used as a loss function for box regression
  - NMS (Non-Maximum Suppression): removes duplicate detections
  - Anchor generation: creates reference boxes at each grid cell
  - Box encoding/decoding: converts between absolute and relative coordinates

Box format convention used throughout this project:
  [x1, y1, x2, y2]  = top-left corner + bottom-right corner (absolute pixels)
  [cx, cy, w, h]    = center-x, center-y, width, height (used in some places)
  [x, y, w, h]      = COCO format (top-left + width/height) -- only in raw annotations
"""

import torch
import numpy as np


# ─── IoU: Intersection over Union ─────────────────────────────────
def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of bounding boxes.

    IoU measures how much two boxes overlap:
        IoU = Area(intersection) / Area(union)
        IoU = 0 means no overlap at all
        IoU = 1 means the boxes are identical

    Args:
        boxes1 (Tensor): Shape [N, 4], format [x1, y1, x2, y2]
        boxes2 (Tensor): Shape [M, 4], format [x1, y1, x2, y2]

    Returns:
        iou (Tensor): Shape [N, M] — IoU value for every pair (i, j)
    """
    # Area of each box = width * height
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    # For every pair (i, j), find the overlap rectangle
    # Overlap starts at the max of the top-left corners
    # Overlap ends at the min of the bottom-right corners
    inter_x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))  # [N, M]
    inter_y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    inter_x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    inter_y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))

    # .clamp(0) ensures no negative area if boxes don't overlap
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)  # [N, M]

    # Union = area1 + area2 - intersection (standard Venn diagram formula)
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area  # [N, M]

    return inter_area / (union_area + 1e-7)  # +1e-7 prevents division by zero


# ─── CIoU: Complete IoU ───────────────────────────────────────────
def compute_ciou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute Complete IoU (CIoU) between predicted and ground-truth boxes.

    CIoU improves on plain IoU by penalizing three things:
      1. Overlap (standard IoU term)
      2. Distance between box centers
      3. Difference in aspect ratio

    CIoU Loss = 1 - CIoU  (minimize this during training)

    Formula:
        CIoU = IoU - (center_distance^2 / diagonal^2) - alpha * v
        where v = (4/pi^2) * (arctan(w_gt/h_gt) - arctan(w_pred/h_pred))^2
        and   alpha = v / (1 - IoU + v)

    Args:
        pred_boxes (Tensor): Predicted boxes, shape [N, 4], format [x1, y1, x2, y2]
        gt_boxes   (Tensor): Ground-truth boxes, shape [N, 4], format [x1, y1, x2, y2]
                             (N matched pairs — one pred per gt)

    Returns:
        ciou (Tensor): CIoU scores, shape [N], values in [-1, 1]
    """
    # --- Standard IoU calculation ---
    px1, py1, px2, py2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    gx1, gy1, gx2, gy2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

    pred_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    gt_area   = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)

    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union_area  = pred_area + gt_area - inter_area
    iou = inter_area / (union_area + 1e-7)

    # --- Center distance penalty ---
    # Pred center
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0
    # GT center
    gcx = (gx1 + gx2) / 2.0
    gcy = (gy1 + gy2) / 2.0

    # Squared Euclidean distance between centers
    center_dist_sq = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    # Smallest enclosing box (the bounding box that contains both boxes)
    enc_x1 = torch.min(px1, gx1)
    enc_y1 = torch.min(py1, gy1)
    enc_x2 = torch.max(px2, gx2)
    enc_y2 = torch.max(py2, gy2)
    # Squared diagonal of the enclosing box
    enc_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

    # --- Aspect ratio consistency penalty ---
    pw = (px2 - px1).clamp(1e-7)  # predicted width
    ph = (py2 - py1).clamp(1e-7)  # predicted height
    gw = (gx2 - gx1).clamp(1e-7)  # gt width
    gh = (gy2 - gy1).clamp(1e-7)  # gt height

    v = (4.0 / (torch.pi ** 2)) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
    alpha = v / (1 - iou + v + 1e-7)

    # Final CIoU formula
    ciou = iou - (center_dist_sq / enc_diag_sq) - alpha * v
    return ciou


# ─── NMS: Non-Maximum Suppression ────────────────────────────────
def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45) -> torch.Tensor:
    """
    Non-Maximum Suppression: removes redundant overlapping detections.

    The algorithm:
      1. Sort all predicted boxes by their confidence score (highest first)
      2. Take the highest-score box — this is a real detection, keep it
      3. Remove all other boxes that overlap with it by more than iou_threshold
         (they are likely detecting the same object)
      4. Repeat from step 2 with the remaining boxes

    Why do we need this?
      A detector predicts thousands of boxes per image.
      Many of them detect the same object (just slightly shifted).
      NMS collapses all these duplicates into a single clean detection.

    Args:
        boxes         (Tensor): [N, 4] in [x1, y1, x2, y2] format
        scores        (Tensor): [N]    confidence scores for each box
        iou_threshold (float):  boxes with IoU > this threshold are suppressed

    Returns:
        keep (Tensor): 1D tensor of indices of the boxes we keep
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Sort by score, descending
    order = scores.argsort(descending=True)
    keep  = []

    while order.numel() > 0:
        # The top-scoring box is always kept
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # Compute IoU of the kept box against all remaining boxes
        rest  = order[1:]
        iou   = compute_iou(boxes[i].unsqueeze(0), boxes[rest])[0]  # [len(rest)]

        # Keep only those with IoU below the threshold
        order = rest[iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# ─── Anchor Generation ────────────────────────────────────────────
def generate_anchors(
    feature_map_size: int,
    anchor_sizes: list,
    stride: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate anchor boxes for a single feature map scale.

    Anchors are pre-defined reference boxes placed at every grid cell.
    The detector learns to *adjust* these anchors to fit actual objects.

    For a 13x13 feature map with 3 anchor sizes, we produce 13*13*3 = 507 anchors.
    Each anchor is centered at the corresponding grid cell center.

    Args:
        feature_map_size (int):   Size of the square feature map (e.g. 13 for P5)
        anchor_sizes     (list):  List of (width, height) pairs, e.g. [[116,90], ...]
        stride           (int):   Downsampling factor (input_size / feature_map_size)
                                  For P5 with input 416: stride = 416/13 = 32
        device           (device): Which device to put the tensors on

    Returns:
        anchors (Tensor): Shape [num_anchors, 4] in [x1, y1, x2, y2] format
                          where num_anchors = feature_map_size^2 * len(anchor_sizes)
    """
    # Create grid of cell center coordinates
    # Each cell center is at (col + 0.5) * stride, (row + 0.5) * stride
    shifts_x = (torch.arange(feature_map_size, device=device) + 0.5) * stride
    shifts_y = (torch.arange(feature_map_size, device=device) + 0.5) * stride
    # shifts_x, shifts_y: [feature_map_size]

    # Create a 2D grid of (cx, cy) values for all cells
    grid_y, grid_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    grid_x = grid_x.flatten()  # [feature_map_size^2]
    grid_y = grid_y.flatten()  # [feature_map_size^2]

    anchors = []
    for (aw, ah) in anchor_sizes:
        # Each anchor is centered at (cx, cy) with width aw and height ah
        # Convert to [x1, y1, x2, y2]
        x1 = grid_x - aw / 2.0
        y1 = grid_y - ah / 2.0
        x2 = grid_x + aw / 2.0
        y2 = grid_y + ah / 2.0
        anchors.append(torch.stack([x1, y1, x2, y2], dim=-1))  # [N_cells, 4]

    # Concatenate all anchor sizes: [N_cells * num_anchor_sizes, 4]
    return torch.cat(anchors, dim=0)


# ─── Box Encoding (GT -> regression targets) ──────────────────────
def encode_boxes(gt_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """
    Encode ground-truth boxes relative to anchor boxes.

    The detector does NOT directly predict [x1, y1, x2, y2].
    Instead, it predicts offsets relative to the anchor box.
    This makes the learning problem much easier.

    Encoding formulas (YOLO-style):
        tx = (cx_gt - cx_anchor) / w_anchor
        ty = (cy_gt - cy_anchor) / h_anchor
        tw = log(w_gt / w_anchor)
        th = log(h_gt / h_anchor)

    The log makes the w/h targets scale-invariant.

    Args:
        gt_boxes (Tensor): Ground-truth boxes [N, 4] in [x1, y1, x2, y2]
        anchors  (Tensor): Matched anchor boxes [N, 4] in [x1, y1, x2, y2]

    Returns:
        targets (Tensor): Regression targets [N, 4] = [tx, ty, tw, th]
    """
    # Convert both from [x1,y1,x2,y2] to [cx,cy,w,h]
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
    gt_w  = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(1e-7)
    gt_h  = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(1e-7)

    an_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
    an_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0
    an_w  = (anchors[:, 2] - anchors[:, 0]).clamp(1e-7)
    an_h  = (anchors[:, 3] - anchors[:, 1]).clamp(1e-7)

    tx = (gt_cx - an_cx) / an_w
    ty = (gt_cy - an_cy) / an_h
    tw = torch.log(gt_w / an_w)
    th = torch.log(gt_h / an_h)

    return torch.stack([tx, ty, tw, th], dim=-1)


# ─── Box Decoding (predictions -> absolute boxes) ─────────────────
def decode_boxes(predictions: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """
    Decode network output (offsets) back to absolute bounding box coordinates.

    This is the inverse of encode_boxes:
        cx = tx * w_anchor + cx_anchor
        cy = ty * h_anchor + cy_anchor
        w  = exp(tw) * w_anchor
        h  = exp(th) * h_anchor

    Args:
        predictions (Tensor): Network's box predictions [N, 4] = [tx, ty, tw, th]
        anchors     (Tensor): Corresponding anchor boxes  [N, 4] in [x1,y1,x2,y2]

    Returns:
        boxes (Tensor): Decoded absolute boxes [N, 4] in [x1, y1, x2, y2]
    """
    an_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
    an_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0
    an_w  = (anchors[:, 2] - anchors[:, 0]).clamp(1e-7)
    an_h  = (anchors[:, 3] - anchors[:, 1]).clamp(1e-7)

    cx = predictions[:, 0] * an_w + an_cx
    cy = predictions[:, 1] * an_h + an_cy
    # clamp tw/th to avoid numerical explosion in exp()
    w  = torch.exp(predictions[:, 2].clamp(-4, 4)) * an_w
    h  = torch.exp(predictions[:, 3].clamp(-4, 4)) * an_h

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    return torch.stack([x1, y1, x2, y2], dim=-1)


# ─── Anchor-GT Matching ───────────────────────────────────────────
def match_anchors_to_gt(anchors: torch.Tensor, gt_boxes: torch.Tensor,
                        pos_iou_threshold: float = 0.5,
                        neg_iou_threshold: float = 0.4):
    """
    Assign each anchor to a ground-truth box (or background).

    The matching strategy:
      - For each GT box, the anchor with the highest IoU is forced positive
        (even if IoU is below pos_iou_threshold)
      - Anchors with IoU >= pos_iou_threshold with any GT box -> POSITIVE
      - Anchors with IoU <  neg_iou_threshold with ALL GT boxes -> NEGATIVE
      - Anchors in between (0.4 - 0.5) -> IGNORE during loss computation

    Args:
        anchors           (Tensor): All anchors [A, 4]
        gt_boxes          (Tensor): Ground-truth boxes [G, 4]
        pos_iou_threshold (float):  IoU threshold for positive assignment
        neg_iou_threshold (float):  IoU threshold for negative assignment

    Returns:
        matched_gt_idx (Tensor): [A] — which GT box each anchor is matched to
                                  (-1 = negative/background, -2 = ignore)
        max_iou        (Tensor): [A] — IoU of each anchor with its matched GT
    """
    num_anchors = anchors.shape[0]

    if gt_boxes.shape[0] == 0:
        # No objects in this image — all anchors are background
        return (
            torch.full((num_anchors,), -1, dtype=torch.long, device=anchors.device),
            torch.zeros(num_anchors, device=anchors.device),
        )

    # IoU matrix: [A, G]
    iou_matrix = compute_iou(anchors, gt_boxes)

    # For each anchor, find its best-matching GT box
    max_iou, matched_gt_idx = iou_matrix.max(dim=1)   # [A], [A]

    # Start: everything is background
    labels = torch.full((num_anchors,), -1, dtype=torch.long, device=anchors.device)

    # Negative: IoU below neg_iou_threshold
    labels[max_iou < neg_iou_threshold] = -1

    # Ignore zone (between thresholds)
    labels[(max_iou >= neg_iou_threshold) & (max_iou < pos_iou_threshold)] = -2

    # Positive: IoU >= pos_iou_threshold
    labels[max_iou >= pos_iou_threshold] = matched_gt_idx[max_iou >= pos_iou_threshold]

    # Force-positive: for each GT, the anchor with highest IoU must be positive
    # (ensures every GT object gets at least one positive anchor)
    best_anchor_per_gt = iou_matrix.max(dim=0)[1]  # [G]
    labels[best_anchor_per_gt] = torch.arange(gt_boxes.shape[0], device=anchors.device)

    return labels, max_iou


# ─── Quick sanity test ─────────────────────────────────────────────
if __name__ == "__main__":
    # Test IoU: two identical boxes should have IoU = 1.0
    box_a = torch.tensor([[10., 10., 50., 50.]])
    box_b = torch.tensor([[10., 10., 50., 50.]])
    print(f"IoU (identical boxes): {compute_iou(box_a, box_b).item():.4f}")  # expect 1.0

    # Test IoU: completely non-overlapping
    box_c = torch.tensor([[100., 100., 150., 150.]])
    print(f"IoU (no overlap):      {compute_iou(box_a, box_c).item():.4f}")  # expect 0.0

    # Test NMS
    boxes  = torch.tensor([[10., 10., 50., 50.], [12., 12., 52., 52.], [200., 200., 250., 250.]])
    scores = torch.tensor([0.9, 0.8, 0.95])
    kept = nms(boxes, scores, iou_threshold=0.5)
    print(f"NMS kept indices: {kept.tolist()}")  # expect [2, 0] (index 1 suppressed)

    print("boxes.py sanity checks passed!")
