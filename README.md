# DeepMeow: Real-Time Cat Detection & Multi-Object Tracking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IliyaJz/DeepMeow/blob/main/notebooks/DeepMeow_Colab.ipynb)

A research-focused computer vision implementation in PyTorch. The goal of this project is to build a single-shot object detector and multi-object tracker from first principles (without relying on high-level prebuilt frameworks like `ultralytics/yolov8`), making every component fully interpretable and extensible.

---

## Project Motivation & Scope

While off-the-shelf detection APIs exist, building a multi-scale object detector from scratch provides deep insights into:
- Feature extraction dynamics across receptive fields
- Anchor-based target encoding and multi-task loss design (CIoU + Focal Loss)
- Motion estimation via Kalman filtering and state association via the Hungarian algorithm

We focus on single-class detection and tracking ("cat") using annotated images from the COCO 2017 dataset for training and custom video feeds for tracking evaluation.

---

## Repository Structure

Here is how the codebase is organized so you can easily follow along with the modular pipeline:

```
DeepMeow/
├── notebooks/
│   └── DeepMeow_Colab.ipynb   # Interactive execution notebook (GPU recommended)
├── data/                      # Dataset root (git-ignored, generated at runtime)
│   ├── raw/                   # COCO cat images (train/val splits)
│   └── annotations/           # Filtered COCO JSON annotations
├── src/
│   ├── data/
│   │   ├── downloader.py      # Automated COCO filtering & streaming downloader
│   │   ├── dataset.py         # PyTorch Dataset implementation
│   │   ├── augmentations.py   # Albumentations pipeline (Spatial + BBox transforms)
│   │   └── mosaic.py          # 4-image Mosaic and Mixup data augmentation
│   ├── models/
│   │   ├── backbone.py        # Custom ResNet-style CNN feature extractor (P3, P4, P5)
│   │   ├── neck.py            # Feature Pyramid Network (FPN) — top-down feature fusion
│   │   ├── head.py            # Multi-scale anchor-based detection head
│   │   └── detector.py        # End-to-end detector (Backbone -> FPN -> Head)
│   ├── losses/
│   │   └── detection_loss.py  # CIoU box loss + Focal objectness + BCE classification
│   ├── utils/
│   │   ├── boxes.py           # IoU, CIoU, NMS, anchor generation, box encode/decode
│   │   ├── metrics.py         # COCO-style mAP evaluation (mAP@50 & mAP@50:95)
│   │   ├── ema.py             # ModelEMA (Exponential Moving Average) weight smoothing
│   │   └── kmeans_anchors.py  # 1-IoU K-Means anchor clustering for custom datasets
│   └── tracking/              # SORT & DeepSORT tracking (Week 5)
├── configs/
│   └── default.yaml           # Central hyperparameter configuration
└── requirements.txt           # Environment dependencies
```

---

## 6-Week Development Roadmap

We are following a 6-week research build schedule:

- [x] **Week 1: Data Pipeline & Custom CNN Backbone**
  - Configured project directory structure & YAML hyperparameter definitions
  - Implemented automated dataset downloader filtering COCO 2017 for cat annotations (~3,000 train / ~500 val images)
  - Built custom PyTorch `Dataset` & Albumentations bbox transform wrapper
  - Built ResNet-style CNN backbone producing multi-scale feature maps ($P_3$, $P_4$, $P_5$) — ~40.5M parameters

- [x] **Week 2: Feature Pyramid Network (FPN), Detection Head & Loss Functions**
  - Implemented FPN top-down pathway with lateral connections, fusing backbone maps into uniform 256-channel outputs
  - Designed anchor-based `MultiScaleHead` producing 6 values per anchor (4 offsets + objectness + class)
  - Implemented anchor-GT matching with IoU thresholds (positive >= 0.5, ignore 0.4–0.5, negative < 0.4)
  - Implemented Complete IoU (CIoU) box regression loss and Focal Loss for objectness imbalance
  - Assembled the full `DeepMeowDetector` with training and NMS-based inference modes

- [x] **Week 3: Training Pipeline & Advanced Augmentation**
  - Implemented AdamW optimizer with linear warmup + Cosine Annealing LR schedule
  - Integrated 4-image Mosaic and Mixup data augmentation strategies
  - Implemented COCO-style mAP evaluation (mAP@50 and mAP@50:95 with 11-point interpolation)
  - Built full training loop with gradient clipping, periodic checkpointing, and best-model saving

- [x] **Week 4: Optimization, Hyperparameter Tuning & Scaling Up**
  - Implemented K-Means anchor clustering using $1 - \text{IoU}$ distance metric to compute optimal anchor sizes for cat shapes
  - Built `ModelEMA` (Exponential Moving Average) with warm-up ramp-up decay and checkpoint state persistence
  - Updated validation to evaluate on smoothed EMA weights for enhanced stability and mAP gains
  - Built multi-session training workflow with LR schedule position recovery and cosine floor ($\eta_{\text{min}}$) for Google Colab

- [ ] **Week 5: Multi-Object Tracking Engine (SORT / DeepSORT)**
  - Implementing 8D linear Kalman Filter for bounding box trajectory state estimation
  - Implementing Hungarian algorithm assignment matrix solving
  - Integrating appearance embeddings for occlusion handling

- [ ] **Week 6: Inference Pipeline, Profiling & Portfolio Documentation**
  - Building real-time video processing pipeline with visual track histories
  - Benchmarking FPS performance and memory footprint
  - Writing final technical project report and ablation summary

---

## Current Architecture Overview

The detection pipeline flows as follows:

```
Input Image [B, 3, 416, 416]
        |
   [Backbone]   <- custom ResNet-style CNN
        |
  P3 [B, 256,  52, 52]   <- stride 8  (small objects)
  P4 [B, 512,  26, 26]   <- stride 16 (medium objects)
  P5 [B, 1024, 13, 13]   <- stride 32 (large objects)
        |
    [FPN Neck]  <- top-down lateral feature fusion
        |
  F3 [B, 256, 52, 52]
  F4 [B, 256, 26, 26]
  F5 [B, 256, 13, 13]
        |
 [Detection Head]  <- separate head per scale
        |
  pred3 [B, 52, 52, 3, 6]   <- 3 anchors, 6 values each
  pred4 [B, 26, 26, 3, 6]
  pred5 [B, 13, 13, 3, 6]
        |
  [Loss / NMS]  <- training vs inference
```

Total predictions per image: 52x52x3 + 26x26x3 + 13x13x3 = **10,647 anchors**

---

## Running the Pipeline (Google Colab)

To replicate our experiments without setting up a local GPU environment:

1. Click the **Open in Colab** badge at the top of this page.
2. Ensure GPU acceleration is enabled (`Runtime -> Change runtime type -> T4 GPU`).
3. Execute the cells in `notebooks/DeepMeow_Colab.ipynb` sequentially:
   - **Cells 1–8**: Week 1 (environment setup, dataset download, backbone test)
   - **Cells 9–15**: Week 2 (FPN, head, loss, and full detector verification)
   - **Cells 16–22**: Week 3 (mAP evaluation, mosaic augmentation, and baseline training)
   - **Cells 23–28**: Week 4 (K-means anchors, EMA verification, multi-session training, debug checklist)

If you prefer running locally:
```bash
git clone https://github.com/IliyaJz/DeepMeow.git
cd DeepMeow
pip install -r requirements.txt
python src/data/downloader.py
```
