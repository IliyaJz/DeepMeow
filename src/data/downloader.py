"""
downloader.py — Data Collection Script
=======================================
Downloads cat images and their bounding box annotations
from the COCO 2017 dataset.

COCO (Common Objects in Context) is a large public dataset.
We only need the "cat" category (COCO category_id = 17).

Run this script once from the project root:
    python src/data/downloader.py
"""

import os
import json
import shutil
import urllib.request
from pathlib import Path
from tqdm import tqdm  # Shows a nice progress bar


# ─── Configuration ────────────────────────────────────────────────
# COCO 2017 annotation files (JSON files listing images + bounding boxes)
COCO_TRAIN_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Where to save everything
RAW_DIR        = Path("data/raw")          # Downloaded images go here
ANN_DIR        = Path("data/annotations")  # Our filtered annotation files go here
COCO_CAT_ID    = 17                        # COCO's ID for the "cat" category
MAX_IMAGES     = 3000                      # Limit to 3000 images to start


def download_file(url: str, dest_path: Path):
    """Download a file from a URL and save it locally, showing progress."""
    print(f"Downloading {url.split('/')[-1]} ...")
    
    def progress_hook(count, block_size, total_size):
        # Calculate and display download percentage
        pct = int(count * block_size * 100 / total_size)
        print(f"\r  Progress: {pct}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
    print()  # New line after progress


def filter_coco_for_cats(coco_ann_file: Path, split: str, max_images: int = MAX_IMAGES):
    """
    Read a full COCO annotation JSON and keep ONLY cat images.
    
    COCO JSON structure:
        {
          "images":      [ {id, file_name, width, height, ...}, ... ],
          "annotations": [ {id, image_id, category_id, bbox, ...}, ... ],
          "categories":  [ {id, name, ...}, ... ]
        }
    
    bbox format in COCO: [x_min, y_min, width, height]
    """
    print(f"\nFiltering COCO {split} annotations for cats...")
    
    with open(coco_ann_file, "r") as f:
        coco = json.load(f)
    
    # Step 1: Find all annotation IDs that belong to the cat category
    cat_ann_ids = {ann["id"] for ann in coco["annotations"] if ann["category_id"] == COCO_CAT_ID}
    
    # Step 2: Find all image IDs that contain at least one cat annotation
    cat_image_ids = {
        ann["image_id"] for ann in coco["annotations"]
        if ann["category_id"] == COCO_CAT_ID
    }
    
    # Step 3: Limit to max_images (so we don't download everything at once)
    cat_image_ids = set(list(cat_image_ids)[:max_images])
    
    # Step 4: Filter the images and annotations lists to only include cat items
    filtered_images = [img for img in coco["images"] if img["id"] in cat_image_ids]
    filtered_anns   = [ann for ann in coco["annotations"]
                       if ann["image_id"] in cat_image_ids and ann["category_id"] == COCO_CAT_ID]
    
    # Step 5: Build our new compact annotation file
    filtered_coco = {
        "images":      filtered_images,
        "annotations": filtered_anns,
        "categories":  [{"id": 1, "name": "cat"}]  # Re-label category ID to 1 (simpler)
    }
    
    # Re-map category_id from 17 → 1 in all annotations
    for ann in filtered_coco["annotations"]:
        ann["category_id"] = 1
    
    # Step 6: Save the filtered annotation file
    out_path = ANN_DIR / f"{split}.json"
    with open(out_path, "w") as f:
        json.dump(filtered_coco, f)
    
    print(f"  ✓ Saved {len(filtered_images)} images, {len(filtered_anns)} annotations → {out_path}")
    return filtered_images


def download_images(images: list, split: str):
    """
    Download actual image files from COCO's image server.
    Images are stored in split subfolders: data/raw/train/ and data/raw/val/
    """
    split_dir = RAW_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO images are hosted at this base URL
    base_url = f"http://images.cocodataset.org/{split}2017/"
    
    print(f"\nDownloading {len(images)} {split} images...")
    skipped = 0
    
    for img_info in tqdm(images, desc=f"  {split} images"):
        dest = split_dir / img_info["file_name"]
        
        # Skip if already downloaded (allows resuming after interruption)
        if dest.exists():
            skipped += 1
            continue
        
        try:
            urllib.request.urlretrieve(base_url + img_info["file_name"], dest)
        except Exception as e:
            print(f"\n  ✗ Failed to download {img_info['file_name']}: {e}")
    
    if skipped > 0:
        print(f"  Skipped {skipped} already-downloaded images.")
    print(f"  ✓ Images saved to {split_dir}")


def main():
    """
    Full download pipeline:
    1. Create directory structure
    2. Download COCO annotation ZIPs
    3. Filter to cat-only annotations
    4. Download cat images
    """
    # ── Create directories ──────────────────────────────────────────
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("data/tmp")
    tmp_dir.mkdir(exist_ok=True)
    
    # ── Step 1: Download COCO annotations ZIP ──────────────────────
    zip_path = tmp_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        download_file(COCO_TRAIN_ANN_URL, zip_path)
    else:
        print("Annotation ZIP already downloaded, skipping.")
    
    # ── Step 2: Extract the ZIP ─────────────────────────────────────
    ann_extract_dir = tmp_dir / "coco_anns"
    if not ann_extract_dir.exists():
        print("Extracting annotations ZIP...")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(ann_extract_dir)
        print("  ✓ Extracted.")
    
    # COCO extracts into an "annotations/" subfolder
    coco_ann_base = ann_extract_dir / "annotations"
    
    # ── Step 3: Filter and save cat-only annotation files ──────────
    # Train split
    train_ann_file = coco_ann_base / "instances_train2017.json"
    train_images   = filter_coco_for_cats(train_ann_file, split="train", max_images=MAX_IMAGES)
    
    # Validation split (use a smaller number for quick evaluation)
    val_ann_file   = coco_ann_base / "instances_val2017.json"
    val_images     = filter_coco_for_cats(val_ann_file, split="val", max_images=500)
    
    # ── Step 4: Download the actual image files ─────────────────────
    download_images(train_images, split="train")
    download_images(val_images,   split="val")
    
    # ── Step 5: Clean up temporary files ───────────────────────────
    print("\nCleaning up temporary files...")
    shutil.rmtree(tmp_dir)
    print("  ✓ Done!")
    
    print("\n✅ Dataset ready!")
    print(f"   Train: {len(train_images)} images  →  data/raw/train/")
    print(f"   Val:   {len(val_images)}   images  →  data/raw/val/")
    print(f"   Annotations: data/annotations/train.json & val.json")


if __name__ == "__main__":
    main()
