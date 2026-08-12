"""
augmentations.py — Data Augmentation Pipeline
===============================================
Data augmentation randomly transforms training images so the model
sees more variety and becomes more robust.

Key challenge:  When we flip or crop an image, we must also
                transform the bounding boxes the same way.

We use the Albumentations library because it handles
box transforms automatically.

Our pipeline:
  Training:
    1. Resize + pad to square (keeps aspect ratio)
    2. Random horizontal flip (50% chance)
    3. Random brightness/contrast changes
    4. Normalize pixel values to ImageNet mean/std
  
  Validation:
    1. Resize + pad to square only (no random changes — we want consistent evaluation)
    2. Normalize pixel values
"""

import numpy as np
from PIL import Image

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── ImageNet normalization statistics ─────────────────────────────
# These are the mean and std values computed over all ImageNet images.
# We apply them so our input distribution matches what backbone weights
# were trained on (useful if we ever use pretrained weights).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transform(input_size: int = 416):
    """
    Returns the augmentation pipeline for TRAINING images.
    
    Each time this pipeline is called on an image, it applies
    random transformations — so the same image looks different
    every epoch.
    
    Args:
        input_size (int): Final image size (e.g. 416 → 416×416 pixels).
    
    Returns:
        An Albumentations Compose object that transforms (image, bboxes) together.
    """
    return A.Compose(
        [
            # ── Spatial transforms ──────────────────────────────────
            
            # Resize the longest side to input_size, then pad the short
            # side with zeros (black bars) to make it a perfect square.
            # This keeps the cat's aspect ratio intact.
            A.LongestMaxSize(max_size=input_size),
            A.PadIfNeeded(
                min_height=input_size,
                min_width=input_size,
                border_mode=0,          # 0 = constant padding (black)
                fill_value=(0, 0, 0),   # Black padding (fill_value replaces deprecated 'value')
            ),
            
            # Flip horizontally with 50% probability.
            # A cat facing left becomes a cat facing right.
            # Albumentations automatically flips the boxes too.
            A.HorizontalFlip(p=0.5),
            
            # Randomly scale and shift the image slightly.
            # scale: zoom in/out by up to 10%
            # translate_percent: shift up/down/left/right by up to 5%
            # Affine replaces the deprecated ShiftScaleRotate.
            # translate_percent shifts the image up to 5% in x and y.
            # scale randomly zooms in/out by up to 10%.
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=0,          # No rotation
                p=0.3,
                mode=0,            # constant border (black)
            ),
            
            # ── Color / appearance transforms ───────────────────────
            # These don't affect boxes, just make the model color-robust.
            
            # Randomly change brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            
            # Randomly change hue, saturation, value (HSV color space)
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.3
            ),
            
            # ── Normalization & Tensor conversion ───────────────────
            
            # Divide by 255 and apply ImageNet mean/std normalization
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            
            # Convert NumPy HWC array → PyTorch CHW tensor
            ToTensorV2(),
        ],
        
        # Tell Albumentations that our boxes are in [x_min, y_min, x_max, y_max]
        # format and that coordinates are absolute pixels (not 0-1 fractions).
        bbox_params=A.BboxParams(
            format="pascal_voc",         # [x1, y1, x2, y2] absolute
            label_fields=["labels"],     # The "labels" list will be transformed alongside boxes
            min_visibility=0.3,          # Drop boxes that become >70% hidden after crop/pad
        ),
    )


def get_val_transform(input_size: int = 416):
    """
    Returns the augmentation pipeline for VALIDATION images.
    
    No random transforms here — we want consistent, reproducible
    evaluation results every time we test the model.
    """
    return A.Compose(
        [
            # Same resize+pad as training (model needs fixed 416×416 input)
            A.LongestMaxSize(max_size=input_size),
            A.PadIfNeeded(
                min_height=input_size,
                min_width=input_size,
                border_mode=0,
                fill_value=(0, 0, 0),
            ),
            # Normalize
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            # To tensor
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.3,
        ),
    )


class AlbumentationsWrapper:
    """
    Wraps an Albumentations pipeline so it works with our Dataset class.
    
    Our Dataset passes (PIL Image, target_dict) to the transform.
    Albumentations expects NumPy arrays and flat lists.
    This wrapper handles the conversion in both directions.
    
    Usage:
        transform = AlbumentationsWrapper(get_train_transform(416))
        image_tensor, target = transform(pil_image, target_dict)
    """

    def __init__(self, alb_transform):
        """
        Args:
            alb_transform: An Albumentations Compose object.
        """
        self.transform = alb_transform

    def __call__(self, pil_image: Image.Image, target: dict):
        """
        Args:
            pil_image (PIL.Image): The raw image.
            target    (dict):      Contains 'boxes' [N,4] tensor and 'labels' [N] tensor.
        
        Returns:
            image_tensor (torch.Tensor): [3, H, W] float tensor
            target       (dict):         Updated with transformed boxes/labels
        """
        # Convert PIL image to NumPy (Albumentations works with NumPy)
        image_np = np.array(pil_image)  # [H, W, 3], uint8
        
        # Convert boxes from PyTorch tensor to a plain Python list
        # Albumentations expects: [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
        boxes  = target["boxes"].numpy().tolist()   # [[x1,y1,x2,y2], ...]
        labels = target["labels"].numpy().tolist()  # [0, 0, 0, ...]  (all cats)
        
        # ── Apply the Albumentations transform ──────────────────────
        result = self.transform(
            image=image_np,
            bboxes=boxes,
            labels=labels,
        )
        
        # ── Extract results ─────────────────────────────────────────
        image_tensor = result["image"]  # Already a [3, H, W] tensor (ToTensorV2)
        
        # Convert transformed boxes back to a PyTorch tensor
        new_boxes  = result["bboxes"]   # List of [x1,y1,x2,y2] tuples
        new_labels = result["labels"]   # List of ints
        
        if len(new_boxes) > 0:
            target["boxes"]  = torch.tensor(new_boxes,  dtype=torch.float32)
            target["labels"] = torch.tensor(new_labels, dtype=torch.long)
        else:
            # All boxes were removed (e.g. cat was fully outside the crop)
            target["boxes"]  = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,),   dtype=torch.long)
        
        return image_tensor, target


# ─── Convenience factory functions ────────────────────────────────

def build_train_transform(input_size: int = 416) -> AlbumentationsWrapper:
    """Returns a ready-to-use training augmentation wrapper."""
    return AlbumentationsWrapper(get_train_transform(input_size))


def build_val_transform(input_size: int = 416) -> AlbumentationsWrapper:
    """Returns a ready-to-use validation augmentation wrapper."""
    return AlbumentationsWrapper(get_val_transform(input_size))
