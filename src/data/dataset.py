"""
dataset.py — PyTorch Dataset Class
====================================
Defines HOW to load a single training sample (image + bounding boxes).

PyTorch's training loop expects a Dataset object that:
  1. Knows how many samples exist   → __len__()
  2. Can return any single sample   → __getitem__(index)

Our __getitem__ returns:
  - image:   a [3, H, W] float tensor (RGB, normalized)
  - target:  a dict with 'boxes' [N, 4] and 'labels' [N]
             N = number of cats in this image
             Boxes are in [x_min, y_min, x_max, y_max] format
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image  # Opens image files

import torch
from torch.utils.data import Dataset


class CatDataset(Dataset):
    """
    Loads cat images and bounding box annotations from a COCO-format JSON file.
    
    COCO annotation format (bbox field):
        [x_min, y_min, width, height]  ← note: NOT x_max/y_max yet
    
    We convert these to:
        [x_min, y_min, x_max, y_max]   ← needed by our loss function
    """

    def __init__(self, ann_file: str, image_dir: str, transform=None):
        """
        Args:
            ann_file  (str): Path to the COCO-format JSON annotation file.
            image_dir (str): Path to the folder containing images.
            transform (callable, optional): Augmentation pipeline to apply.
        """
        self.image_dir = Path(image_dir)
        self.transform = transform  # Augmentation pipeline (Week 1 augmentations.py)
        
        # ── Load annotation file ────────────────────────────────────
        with open(ann_file, "r") as f:
            coco = json.load(f)
        
        # Build a lookup: image_id → image metadata (file name, width, height)
        self.id_to_image = {img["id"]: img for img in coco["images"]}
        
        # Group annotations by image_id so we can quickly find all
        # bounding boxes for a given image.
        self.id_to_anns = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.id_to_anns:
                self.id_to_anns[img_id] = []
            self.id_to_anns[img_id].append(ann)
        
        # Keep only image IDs that have at least one cat annotation
        self.image_ids = [
            img_id for img_id in self.id_to_image
            if img_id in self.id_to_anns  # skip images with no cats
        ]
        
        print(f"Dataset loaded: {len(self.image_ids)} images with cat annotations.")

    def __len__(self):
        """How many images are in this dataset split."""
        return len(self.image_ids)

    def __getitem__(self, index: int):
        """
        Load and return one training sample.
        
        Returns:
            image  (Tensor): Shape [3, H, W], values in [0, 1], RGB.
            target (dict):
                "boxes"  (Tensor): Shape [N, 4], format [x1, y1, x2, y2]
                "labels" (Tensor): Shape [N],    all zeros (only one class: cat)
                "image_id" (int): For debugging / evaluation
        """
        img_id   = self.image_ids[index]
        img_info = self.id_to_image[img_id]
        
        # ── 1. Load image ──────────────────────────────────────────
        # Determine which split folder the image lives in (train or val)
        # The file_name in COCO is just e.g. "000000123456.jpg"
        img_path = self._find_image(img_info["file_name"])
        
        # Open as PIL Image and convert to RGB (some PNGs are RGBA)
        image = Image.open(img_path).convert("RGB")
        
        # ── 2. Load bounding boxes ─────────────────────────────────
        anns  = self.id_to_anns[img_id]
        boxes = []
        
        for ann in anns:
            # COCO format: [x_min, y_min, width, height]
            x, y, w, h = ann["bbox"]
            
            # Skip degenerate boxes (width or height is 0 or negative)
            if w <= 0 or h <= 0:
                continue
            
            # Convert to [x_min, y_min, x_max, y_max]
            boxes.append([x, y, x + w, y + h])
        
        # If all annotations were degenerate, return empty tensors
        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.long)
        else:
            boxes  = torch.tensor(boxes, dtype=torch.float32)  # [N, 4]
            # All labels are 0 (cat). Shape: [N]
            labels = torch.zeros(len(boxes), dtype=torch.long)
        
        target = {
            "boxes":    boxes,   # [N, 4]
            "labels":   labels,  # [N]
            "image_id": img_id,
        }
        
        # ── 3. Apply augmentations ─────────────────────────────────
        # Our augmentation pipeline (augmentations.py) takes a PIL image
        # and a target dict, and returns an augmented version of both.
        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            # Without augmentation: just convert PIL image to a tensor
            # Converts HWC uint8 [0,255] → CHW float32 [0,1]
            image = _pil_to_tensor(image)
        
        return image, target

    def _find_image(self, file_name: str) -> Path:
        """
        Search for the image file across the train/ and val/ subdirectories.
        COCO stores images in split subfolders.
        """
        for split in ["train", "val"]:
            path = self.image_dir / split / file_name
            if path.exists():
                return path
        # Fallback: try the image_dir itself
        return self.image_dir / file_name


# ─── Utility: PIL Image → Tensor ──────────────────────────────────
def _pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image (HxW, RGB, uint8) to a float tensor (3xHxW, [0,1]).
    
    Breakdown:
      np.array(...)    → HxW x3 NumPy array, values 0-255
      / 255.0          → normalize to 0.0–1.0 range
      .transpose(2,0,1)→ rearrange from HWC to CHW (PyTorch convention)
      torch.tensor     → convert NumPy array to PyTorch tensor
    """
    arr = np.array(pil_image, dtype=np.float32) / 255.0  # [H, W, 3], float
    arr = arr.transpose(2, 0, 1)                          # [3, H, W]
    return torch.tensor(arr, dtype=torch.float32)


# ─── Custom Collate Function ───────────────────────────────────────
def collate_fn(batch):
    """
    PyTorch's default collate_fn tries to stack everything into a single
    tensor. That fails here because different images have different numbers
    of bounding boxes (some images have 1 cat, others have 5).

    This custom function keeps images stacked but keeps targets as a list.
    
    Input:  list of (image, target) tuples — one per image in the batch
    Output: (stacked_images, list_of_targets)
             stacked_images: [batch_size, 3, H, W]
             list_of_targets: [{boxes: ..., labels: ...}, ...]
    """
    images  = [item[0] for item in batch]  # list of [3, H, W] tensors
    targets = [item[1] for item in batch]  # list of dicts
    
    images = torch.stack(images, dim=0)    # [B, 3, H, W]
    
    return images, targets
