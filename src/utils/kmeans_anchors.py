"""
kmeans_anchors.py - K-Means Anchor Optimization
================================================
Clusters all GT boxes in the training set using k-means with 1-IoU
distance to find the 9 anchor sizes that best match the cat dataset.

Usage:
    python src/utils/kmeans_anchors.py --ann data/annotations/train.json
"""

import json
import argparse
import numpy as np
from pathlib import Path


def iou_distance(boxes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Compute 1-IoU between each box and each centroid.
    Boxes/centroids are (width, height) pairs centered at origin.
    Returns [N, K] distance matrix.
    """
    inter_w = np.minimum(boxes[:, 0:1], centroids[:, 0])
    inter_h = np.minimum(boxes[:, 1:2], centroids[:, 1])
    inter   = inter_w * inter_h
    box_area      = boxes[:, 0] * boxes[:, 1]
    centroid_area = centroids[:, 0] * centroids[:, 1]
    union = box_area[:, None] + centroid_area[None, :] - inter
    iou   = inter / (union + 1e-7)
    return 1.0 - iou


def kmeans_anchors(
    ann_file: str,
    k: int = 9,
    input_size: int = 416,
    max_iter: int = 1000,
    tol: float = 1e-5,
    seed: int = 42,
) -> list:
    """
    Run k-means clustering on GT box dimensions.

    Args:
        ann_file   (str): Path to COCO-format JSON annotation file
        k          (int): Number of clusters (default 9: 3 per scale)
        input_size (int): Model input size in pixels
        max_iter   (int): Maximum k-means iterations
        tol        (float): Convergence tolerance
        seed       (int): Random seed

    Returns:
        anchors (list): 9 [w, h] pairs sorted by area.
                        First 3=small(P3), middle 3=medium(P4), last 3=large(P5).
    """
    np.random.seed(seed)

    print(f"Loading annotations from: {ann_file}")
    with open(ann_file, "r") as f:
        coco = json.load(f)

    img_size = {img["id"]: (img["width"], img["height"])
                for img in coco["images"]}

    boxes = []
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        if w <= 2 or h <= 2:
            continue
        img_w, img_h = img_size[ann["image_id"]]
        boxes.append([w / img_w, h / img_h])

    boxes = np.array(boxes, dtype=np.float32)
    print(f"Total GT boxes: {len(boxes)}")
    print(f"Box size stats (normalised):")
    print(f"  Width  - mean: {boxes[:,0].mean():.3f}  std: {boxes[:,0].std():.3f}")
    print(f"  Height - mean: {boxes[:,1].mean():.3f}  std: {boxes[:,1].std():.3f}")

    # Initialise centroids by random sampling
    idx       = np.random.choice(len(boxes), k, replace=False)
    centroids = boxes[idx].copy()

    prev_assignments = None
    for iteration in range(max_iter):
        distances   = iou_distance(boxes, centroids)
        assignments = distances.argmin(axis=1)

        if prev_assignments is not None and np.all(assignments == prev_assignments):
            print(f"K-means converged at iteration {iteration + 1}")
            break
        prev_assignments = assignments.copy()

        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            cluster_boxes = boxes[assignments == c]
            if len(cluster_boxes) == 0:
                new_centroids[c] = boxes[np.random.randint(len(boxes))]
            else:
                new_centroids[c] = cluster_boxes.mean(axis=0)

        shift = np.abs(new_centroids - centroids).mean()
        centroids = new_centroids
        if shift < tol:
            print(f"K-means converged (shift={shift:.2e}) at iteration {iteration + 1}")
            break
    else:
        print(f"K-means did not fully converge after {max_iter} iterations")

    # Quality metric
    distances = iou_distance(boxes, centroids)
    best_iou  = 1.0 - distances.min(axis=1)
    print(f"Mean best IoU: {best_iou.mean():.4f}")

    # Scale to pixels and sort by area
    centroids_px = (centroids * input_size).round().astype(int)
    areas        = centroids_px[:, 0] * centroids_px[:, 1]
    sorted_idx   = np.argsort(areas)
    anchors      = centroids_px[sorted_idx].tolist()

    print(f"\nOptimised anchors (sorted by area @ {input_size}x{input_size}):")
    labels = ["P3 (small) "] * 3 + ["P4 (medium)"] * 3 + ["P5 (large) "] * 3
    for i, (w, h) in enumerate(anchors):
        print(f"  [{labels[i]}]  [{w:4d}, {h:4d}]  area={w*h:,}")

    print("\nCopy into detector.py ANCHOR_SIZES:")
    print(f"  small  (P3): {anchors[0:3]}")
    print(f"  medium (P4): {anchors[3:6]}")
    print(f"  large  (P5): {anchors[6:9]}")

    return anchors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Means anchor clustering for DeepMeow")
    parser.add_argument("--ann",        type=str, default="data/annotations/train.json")
    parser.add_argument("--k",          type=int, default=9)
    parser.add_argument("--input_size", type=int, default=416)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    kmeans_anchors(ann_file=args.ann, k=args.k, input_size=args.input_size, seed=args.seed)
