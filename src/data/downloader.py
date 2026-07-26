"""
downloader.py — Data Collection Script
=======================================
Downloads cat images and their bounding box annotations
from the COCO 2017 dataset.

Works both locally and on Google Colab.

Run from the repo root:
    python src/data/downloader.py

On Google Colab:
    !python src/data/downloader.py
    
    Optional: mount Google Drive first so data persists across sessions:
    from google.colab import drive
    drive.mount('/content/drive')
    Then set SAVE_TO_DRIVE = True below.
"""

import os
import json
import shutil
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


# ─── Colab / Drive Configuration ──────────────────────────────────
# Set SAVE_TO_DRIVE = True if you mounted Google Drive in Colab
# and want data to persist between sessions (recommended for teams).
SAVE_TO_DRIVE = False
DRIVE_PATH    = Path("/content/drive/MyDrive/DeepMeow/data")

# ─── Dataset Configuration ────────────────────────────────────────
COCO_TRAIN_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_CAT_ID        = 17    # COCO's integer ID for the "cat" class
MAX_TRAIN_IMAGES   = 3000  # Enough to start; increase later if needed
MAX_VAL_IMAGES     = 500


def _get_data_root() -> Path:
    """
    Return the root data directory.
    - If running in Colab with Drive mounted: use Drive path (data persists!)
    - Otherwise: use local 'data/' folder relative to the repo root
    """
    if SAVE_TO_DRIVE and DRIVE_PATH.parent.exists():
        print(f"💾 Saving data to Google Drive: {DRIVE_PATH}")
        return DRIVE_PATH
    return Path("data")


def download_file(url: str, dest_path: Path):
    """Download a file from a URL, showing a tqdm progress bar."""
    print(f"\n⬇️  Downloading: {url.split('/')[-1]}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # urllib doesn't give byte-level progress, so we do a streaming download
    response = urllib.request.urlopen(url)
    total    = int(response.headers.get("Content-Length", 0))
    
    chunk_size = 1024 * 64  # 64 KB per chunk
    downloaded = 0

    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="  Progress"
    ) as bar:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            bar.update(len(chunk))
            downloaded += len(chunk)

    print(f"  ✓ Saved to {dest_path}")


def filter_coco_for_cats(coco_ann_file: Path, split: str, max_images: int):
    """
    Read a full COCO annotation JSON (which has 80 classes) and keep
    ONLY images/annotations that contain at least one cat.

    COCO JSON structure:
        {
          "images":      [ {id, file_name, width, height, ...}, ... ],
          "annotations": [ {id, image_id, category_id, bbox, ...}, ... ],
          "categories":  [ {id, name, ...}, ... ]
        }

    bbox format in COCO: [x_min, y_min, width, height]
    We keep it as-is here; dataset.py converts to [x1, y1, x2, y2].
    """
    print(f"\n🔍 Filtering COCO '{split}' annotations for cats...")
    
    with open(coco_ann_file, "r") as f:
        coco = json.load(f)

    # Step 1: Find all image_ids that have at least one cat annotation
    cat_image_ids = {
        ann["image_id"]
        for ann in coco["annotations"]
        if ann["category_id"] == COCO_CAT_ID
    }

    # Step 2: Limit to max_images (prevents downloading everything)
    cat_image_ids = set(list(cat_image_ids)[:max_images])

    # Step 3: Filter images and annotations lists
    filtered_images = [
        img for img in coco["images"]
        if img["id"] in cat_image_ids
    ]
    filtered_anns = [
        ann for ann in coco["annotations"]
        if ann["image_id"] in cat_image_ids
        and ann["category_id"] == COCO_CAT_ID
    ]

    # Step 4: Re-map category_id from 17 → 1 (we only have 1 class now)
    for ann in filtered_anns:
        ann["category_id"] = 1

    # Step 5: Build final annotation dict and save
    result = {
        "images":      filtered_images,
        "annotations": filtered_anns,
        "categories":  [{"id": 1, "name": "cat"}],
    }

    print(f"  ✓ {len(filtered_images)} images, {len(filtered_anns)} annotations")
    return result, filtered_images


def download_images(images: list, split: str, raw_dir: Path):
    """
    Download the actual .jpg image files from COCO's server.
    Skips already-downloaded files so it's safe to re-run.
    """
    split_dir = raw_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"http://images.cocodataset.org/{split}2017/"
    
    print(f"\n📥 Downloading {len(images)} {split} images...")
    failed  = []
    skipped = 0

    for img_info in tqdm(images, desc=f"  {split}"):
        dest = split_dir / img_info["file_name"]
        
        # Skip already-downloaded images (safe to restart)
        if dest.exists():
            skipped += 1
            continue
        
        try:
            urllib.request.urlretrieve(base_url + img_info["file_name"], dest)
        except Exception as e:
            failed.append(img_info["file_name"])

    if skipped > 0:
        print(f"  ⏭️  Skipped {skipped} already-downloaded images.")
    if failed:
        print(f"  ⚠️  {len(failed)} images failed. They will be missing from training.")
    
    print(f"  ✓ Images saved to: {split_dir}")


def main():
    """
    Full pipeline:
      1. Create directory structure
      2. Download COCO 2017 annotation ZIP (~241 MB)
      3. Extract and filter to cat-only JSON files
      4. Download cat image files
      5. Clean up the temporary ZIP
    """
    data_root = _get_data_root()
    raw_dir   = data_root / "raw"
    ann_dir   = data_root / "annotations"
    tmp_dir   = data_root / "tmp"

    raw_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(" DeepMeow Dataset Downloader")
    print("=" * 55)

    # ── Step 1: Download COCO annotation ZIP ──────────────────────
    zip_path = tmp_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        download_file(COCO_TRAIN_ANN_URL, zip_path)
    else:
        print(f"\n✓ Annotation ZIP already exists, skipping download.")

    # ── Step 2: Extract ZIP ────────────────────────────────────────
    extract_dir = tmp_dir / "extracted"
    if not extract_dir.exists():
        print("\n📦 Extracting annotations ZIP...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        print("  ✓ Extracted.")
    
    coco_ann_base = extract_dir / "annotations"

    # ── Step 3: Filter and save cat-only annotation files ─────────
    for split, max_imgs in [("train", MAX_TRAIN_IMAGES), ("val", MAX_VAL_IMAGES)]:
        ann_out = ann_dir / f"{split}.json"
        
        if ann_out.exists():
            print(f"\n✓ {split}.json already exists, skipping filter step.")
            # Still need image list for downloading
            with open(ann_out) as f:
                images = json.load(f)["images"]
        else:
            coco_ann_file = coco_ann_base / f"instances_{split}2017.json"
            result, images = filter_coco_for_cats(coco_ann_file, split, max_imgs)
            
            with open(ann_out, "w") as f:
                json.dump(result, f)
            print(f"  ✓ Saved → {ann_out}")

        # ── Step 4: Download images ────────────────────────────────
        download_images(images, split, raw_dir)

    # ── Step 5: Clean up temp files ───────────────────────────────
    print("\n🧹 Cleaning up temporary files...")
    shutil.rmtree(tmp_dir)
    print("  ✓ Done!")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" ✅ Dataset ready!")
    print(f"    Train images : {raw_dir}/train/")
    print(f"    Val images   : {raw_dir}/val/")
    print(f"    Annotations  : {ann_dir}/train.json & val.json")
    print("=" * 55)


if __name__ == "__main__":
    main()
