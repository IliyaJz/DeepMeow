# 🐱 DeepMeow: Custom Cat Detection & Tracking from Scratch

**DeepMeow** is a computer vision research project built in PyTorch to detect and track cats in video streams using custom deep learning models built from scratch.

---

## 🎯 Project Overview

This repository contains:
- **Custom Backbone & FPN**: ResNet-inspired feature extractor + Feature Pyramid Network for multi-scale visual features.
- **Single-Shot Detector**: Custom anchor-based detection head with CIoU loss and Non-Maximum Suppression (NMS).
- **Multi-Object Tracker**: Implementation of SORT (Kalman Filter + Hungarian Algorithm) and DeepSORT with appearance feature embeddings.
- **Dataset Pipeline**: Data collection, annotation parsing (COCO format), and spatial/color augmentations (Mosaic, Mixup).

---

## 📁 Repository Structure

```
DeepMeow/
├── data/                  # Datasets (ignored in git)
├── src/                   # Source code
│   ├── data/              # Dataset loading & augmentations
│   ├── models/            # Backbone, FPN, detection head
│   ├── tracking/          # Kalman Filter, SORT, DeepSORT
│   ├── losses/            # Multi-task detection & bounding box loss
│   └── utils/             # Box math (IoU, NMS), evaluation metrics (mAP)
├── notebooks/             # Exploratory analysis & visualization
├── configs/               # Hyperparameter configurations
├── checkpoints/           # Model checkpoints (ignored in git)
└── results/               # Evaluation & benchmark outputs
```

---

## 🚀 Plan & Roadmap

- **Week 1**: Data Pipeline & Custom CNN Backbone
- **Week 2**: Detection Head, Anchors & CIoU Loss
- **Week 3**: Training Pipeline & Data Augmentation (Mosaic/Mixup)
- **Week 4**: Hyperparameter Optimization & Full Training
- **Week 5**: SORT & DeepSORT Tracking Engine
- **Week 6**: Inference Optimization, Profiling & Portfolio Documentation
