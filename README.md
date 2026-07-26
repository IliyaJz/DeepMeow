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
│   │   └── augmentations.py   # Albumentations pipeline (Spatial + BBox transforms)
│   ├── models/
│   │   ├── backbone.py        # Custom ResNet-style CNN feature extractor (P3, P4, P5)
│   │   ├── neck.py            # Feature Pyramid Network (FPN) for multi-scale fusion
│   │   ├── head.py            # Anchor-based detection head
│   │   └── detector.py        # Complete end-to-end detector module
│   ├── tracking/              # SORT & DeepSORT tracking algorithms
│   ├── losses/                # Bounding box regression (CIoU) & Focal Loss modules
│   └── utils/                 # Bounding box operations (IoU, NMS) & mAP metrics
├── configs/
│   └── default.yaml           # Central hyperparameter configuration
└── requirements.txt           # Environment dependencies
```

---

## 6-Week Development Roadmap

We are following a 6-week research build schedule:

- [x] **Week 1: Data Pipeline & Custom CNN Backbone**
  - Configured project directory structure & YAML hyperparameter definitions
  - Implemented automated dataset downloader filtering COCO 2017 for cat annotations
  - Built custom PyTorch `Dataset` & Albumentations bbox transform wrapper
  - Built ResNet-style CNN backbone producing multi-scale feature maps ($P_3, P_4, P_5$)

- [ ] **Week 2: Feature Pyramid Network (FPN), Detection Head & Loss Functions**
  - Constructing FPN top-down pathway and lateral connections
  - Designing anchor-based detection head and regression targets
  - Implementing CIoU bounding box loss and Focal Loss for objectness

- [ ] **Week 3: Training Pipeline & Advanced Augmentation**
  - Setting up AdamW optimizer with warmup + Cosine Annealing learning rate schedule
  - Integrating Mosaic & Mixup data augmentations
  - Implementing COCO-style mAP evaluation metric

- [ ] **Week 4: Optimization, Hyperparameter Tuning & Full Training**
  - Running k-means anchor clustering on dataset ground-truth boxes
  - Implementing Exponential Moving Average (EMA) model weights
  - Completing full-scale training runs (200+ epochs)

- [ ] **Week 5: Multi-Object Tracking Engine (SORT / DeepSORT)**
  - Implementing 8D linear Kalman Filter for bounding box trajectory state estimation
  - Implementing Hungarian algorithm assignment matrix solving
  - Integrating appearance embeddings for occlusion handling

- [ ] **Week 6: Inference Pipeline, Profiling & Portfolio Documentation**
  - Building real-time video processing pipeline with visual track histories
  - Benchmarking FPS performance and memory footprint
  - Writing final technical project report and ablation summary

---

## Running the Pipeline (Google Colab)

To replicate our experiments without setting up a local GPU environment:

1. Click the **Open in Colab** badge at the top of this page.
2. Ensure GPU acceleration is enabled (`Runtime -> Change runtime type -> T4 GPU`).
3. Execute the cells in [`notebooks/DeepMeow_Colab.ipynb`](file:///Users/macbook/Codes/Antigravity/Cat%20Detector/notebooks/DeepMeow_Colab.ipynb) sequentially.

If you prefer running locally:
```bash
git clone https://github.com/IliyaJz/DeepMeow.git
cd DeepMeow
pip install -r requirements.txt
python src/data/downloader.py
```
