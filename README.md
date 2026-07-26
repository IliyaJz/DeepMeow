# 🐱 DeepMeow: Custom Cat Detection & Tracking from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IliyaJz/DeepMeow/blob/main/notebooks/DeepMeow_Colab.ipynb)

**DeepMeow** is a computer vision research project built in PyTorch to detect and track cats in video streams using custom deep learning models built from scratch.

---

## 🚀 Quick Start (Google Colab — Recommended)

Click the **"Open in Colab"** badge above. The notebook will:
1. Clone this repository into Colab
2. Install all dependencies
3. Download the COCO cat dataset (~500 MB, takes ~1 min on Colab)
4. Verify the backbone forward pass

> **No local setup needed.** Free GPU access included.

---

## 🎯 Project Overview

This repository contains:
- **Custom Backbone & FPN**: ResNet-inspired feature extractor + Feature Pyramid Network for multi-scale visual features.
- **Single-Shot Detector**: Custom anchor-based detection head with CIoU loss and Non-Maximum Suppression (NMS).
- **Multi-Object Tracker**: Implementation of SORT (Kalman Filter + Hungarian Algorithm) and DeepSORT with appearance feature embeddings.
- **Dataset Pipeline**: Data collection from COCO 2017, annotation parsing, and spatial/color augmentations (Mosaic, Mixup).

---

## 📁 Repository Structure

```
DeepMeow/
├── notebooks/
│   └── DeepMeow_Colab.ipynb   # ← Start here on Google Colab
├── data/                      # Datasets (downloaded at runtime, not in git)
├── src/
│   ├── data/
│   │   ├── downloader.py      # Downloads COCO cat images & annotations
│   │   ├── dataset.py         # PyTorch Dataset class
│   │   └── augmentations.py   # Image + bounding box augmentations
│   ├── models/
│   │   ├── backbone.py        # Custom CNN (ConvBlock + ResidualBlock + FPN outputs)
│   │   ├── neck.py            # Feature Pyramid Network (Week 2)
│   │   ├── head.py            # Detection head (Week 2)
│   │   └── detector.py        # Full model assembly (Week 2)
│   ├── tracking/              # SORT & DeepSORT (Week 5)
│   ├── losses/                # CIoU + Focal loss (Week 2)
│   └── utils/                 # IoU, NMS, mAP metrics
├── configs/
│   └── default.yaml           # All hyperparameters
└── requirements.txt
```

---

## 🗺️ 6-Week Roadmap

| Week | Focus | Status |
|------|-------|--------|
| **1** | Data Pipeline & Custom CNN Backbone | ✅ Done |
| **2** | FPN Neck, Detection Head & Loss Functions | 🔄 In Progress |
| **3** | Full Training Pipeline & Augmentations | ⏳ Upcoming |
| **4** | Hyperparameter Optimization & Full Training | ⏳ Upcoming |
| **5** | SORT & DeepSORT Multi-Object Tracking | ⏳ Upcoming |
| **6** | Inference Optimization & Portfolio Polish | ⏳ Upcoming |

---

## 👥 Team Workflow

Each team member can work on Colab independently using the same shared codebase:
```bash
# In your Colab notebook or terminal:
git clone https://github.com/IliyaJz/DeepMeow.git
cd DeepMeow
pip install -r requirements.txt
python src/data/downloader.py
```
