# ViT-Chest-Xray

**A Comparative Study of CNN, ResNet, and Vision Transformers for Multi-label Classification of Chest X-ray Diseases**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-NIH%20ChestXray14-blue)](https://www.kaggle.com/datasets/nih-chest-xrays/data)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project implements and compares three deep learning architectures — **CNN**, **ResNet-34**, and **Vision Transformer (ViT)** — for multi-label chest X-ray disease classification on the **NIH ChestX-ray14** dataset (112,120 images, 15 classes including "No Finding").

Key contributions:
- From-scratch implementations of CNN, ResNet, and ViT in PyTorch
- Patient-level data splitting to prevent data leakage
- Transfer learning with pretrained ViT (timm `vit_base_patch16_224`)
- 8 custom loss functions for handling class imbalance
- YAML-based configuration system with inheritance
- Modular callback-driven training pipeline

---

## Results

| Model | Data Scale | Parameters | Test AUC | Test Accuracy | Epochs |
|:------|:-----------|:-----------|:---------|:--------------|:-------|
| CNN Baseline | 60 images | 95.6M | 0.5777 | — | 10 |
| ResNet-34 | 60 images | 21.3M | 0.4462 | — | 10 |
| ViT-v1 (scratch) | 60 images | 9.0M | 0.5854 | 91.33% | 10 |
| ViT-v2 (SGD + Early Stop) | 60 images | 9.0M | 0.6303 | 89.67% | 9 |
| ViT-ResNet (pretrained) | 60 images | 85.8M | 0.6694 | 87.00% | 10 |
| **ViT Final (scratch)** | **112K images** | **9.0M** | **0.7225** | **92.91%** | **10** |

### Key Findings

1. **Data scale is decisive** — Same ViT architecture improved from AUC 0.5854 (60 images) to **0.7225** (112K images), a 23.4% gain.
2. **Transfer learning excels on small data** — Pretrained ViT achieved 0.6694 AUC with only 60 training images.
3. **CNN overfits severely** — 99.98% of parameters in FC layers cause train/test AUC gap of 0.33.
4. **ViT achieves best overall performance** — Highest AUC and accuracy on the full dataset.

### Per-class AUC (ViT Full-scale)

| Disease | AUC | Disease | AUC |
|:--------|:----|:--------|:----|
| Edema | **0.8422** | Pleural Thickening | 0.6997 |
| Cardiomegaly | 0.7996 | Fibrosis | 0.6977 |
| Effusion | 0.7880 | Mass | 0.6762 |
| Consolidation | 0.7615 | Pneumonia | 0.6710 |
| Pneumothorax | 0.7540 | Infiltration | 0.6614 |
| Hernia | 0.7460 | Nodule | 0.5747 |
| Emphysema | 0.7375 | | |
| Atelectasis | 0.7170 | **Macro Average** | **0.7225** |
| No Finding | 0.7114 | | |

---

## Project Structure

```
ViT-Chest-Xray/
├── main.py                      # CLI entry point (train, evaluate, predict, verify)
├── requirements.txt             # Python dependencies
├── README.md
│
├── src/                         # Core source package
│   ├── models/
│   │   ├── cnn.py               # CNN baseline (2 conv layers, ~95.6M params)
│   │   ├── resnet.py            # ResNet-18/34/50/101 from scratch (~21.3M params)
│   │   ├── vit.py               # Vision Transformer from scratch (~9M params)
│   │   └── pretrained.py        # Pretrained model wrapper (timm/torchvision)
│   ├── data/
│   │   ├── dataset.py           # DatasetParser, ChestXrayDataset
│   │   ├── splits.py            # Patient-level train/val/test splitting
│   │   └── transforms.py        # 4 augmentation levels (none/basic/standard/advanced)
│   ├── losses/
│   │   ├── focal_loss.py        # Focal Loss
│   │   ├── weighted_loss.py     # Weighted BCE Loss
│   │   ├── asymmetric_loss.py   # Asymmetric Loss (ICCV 2021)
│   │   ├── dice_loss.py         # Dice / Dice-BCE Loss
│   │   ├── combined_loss.py     # Multi-component combined loss
│   │   ├── smoothing_loss.py    # Label Smoothing BCE
│   │   ├── distillation_loss.py # Knowledge Distillation Loss
│   │   └── utils.py             # Class weight computation (4 methods)
│   └── utils/
│       ├── training.py          # Trainer class with training/eval loops
│       ├── evaluation.py        # AUC, ROC curves, confusion matrices
│       ├── callbacks.py         # EarlyStopping, ModelCheckpoint, MetricsHistory
│       ├── config_loader.py     # YAML config with inheritance & env vars
│       └── reproducibility.py   # Seed management (Python/NumPy/PyTorch/CUDA)
│
├── configs/                     # YAML configuration files
│   ├── base.yaml                # Base config (shared settings)
│   ├── cnn.yaml                 # CNN-specific config
│   ├── resnet.yaml              # ResNet-specific config
│   └── vit_small.yaml           # ViT-Small config
│
├── scripts/
│   ├── train.py                 # Full CLI training script
│   └── demo.py                  # Demo inference script
│
├── notebooks/
│   ├── analysis/                # Data exploration & download
│   │   ├── data_download.ipynb  # Download NIH dataset via Kaggle API
│   │   └── data.ipynb           # EDA, preprocessing, DataLoaders
│   └── experiments/             # Model training experiments
│       ├── cnn.ipynb            # CNN training
│       ├── resnet.ipynb         # ResNet-34 training
│       ├── ViT-v1.ipynb         # ViT v1 (Adam, from scratch)
│       ├── ViT-v2.ipynb         # ViT v2 (SGD + Early Stopping)
│       ├── ViT-ResNet.ipynb     # Pretrained ViT (timm)
│       ├── Final_ViT_ChestXray.ipynb  # Full-scale ViT experiment (112K images)
│       └── 01-06_*.ipynb        # Improvement experiments (augmentation, losses, etc.)
│
├── models/checkpoints/          # Trained model weights (Git LFS)
│   ├── cnn_model.pth
│   ├── resnet_model.pth
│   ├── vit_best.pth             # Best ViT (full dataset)
│   ├── vit_pretrained_best.pth
│   ├── vit_v1_best.pth
│   └── vit_v2_best.pth
│
├── assets/image/                # Figures for reports
├── data/                        # Dataset directory (not tracked)
│   ├── raw/                     # Raw CSV metadata
│   └── processed/               # Processed images (~42GB)
│
├── docs/                        # LaTeX reports
│   ├── report_vi.tex            # Vietnamese report
│   ├── report_en.tex            # English report
│   └── Proposal/                # Project proposal
│
├── tests/                       # Unit tests
│   ├── test_models.py           # Model architecture tests
│   └── test_refactor.py         # Refactoring validation tests
│
├── Project/                     # Legacy notebooks (archived)
└── results/                     # Experiment outputs
```

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/<your-username>/ViT-Chest-Xray.git
cd ViT-Chest-Xray
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python main.py verify
```

This checks all imports, GPU availability, and module integrity.

### 3. Download Dataset

Run the notebook `notebooks/analysis/data_download.ipynb` to download the NIH ChestX-ray14 dataset via Kaggle API. The dataset (~42GB) will be saved to `data/processed/`.

### 4. Train a Model

**Using CLI:**
```bash
# Train ViT with YAML config
python main.py train --config configs/vit_small.yaml

# Train CNN
python main.py train --config configs/cnn.yaml

# Train ResNet
python main.py train --config configs/resnet.yaml
```

**Using Notebooks:**
Open any notebook in `notebooks/experiments/` for interactive training with visualization.

### 5. Evaluate

```bash
python main.py evaluate --checkpoint models/checkpoints/vit_best.pth --data test
```

### 6. List Available Models

```bash
python main.py models --list
python main.py models --test  # Run forward pass tests
```

---

## Dataset

**NIH ChestX-ray14** — one of the largest publicly available chest X-ray datasets.

| Attribute | Value |
|:----------|:------|
| Total images | 112,120 |
| Unique patients | 30,805 |
| Classes | 14 diseases + No Finding = 15 |
| Resolution | 1024 × 1024 (resized to 224 × 224) |
| Format | PNG (grayscale → RGB) |
| Class imbalance | 269× ratio (No Finding 53.84% vs Hernia 0.20%) |

**15 Disease Classes:** Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax, No Finding

**Data Split (Full-scale):**
- Train: 78,614 images (21,563 patients)
- Validation: 11,212 images (3,081 patients)
- Test: 22,294 images (6,161 patients)

Patient-level splitting ensures no data leakage between sets.

---

## Model Architectures

### CNN Baseline
- 2 convolutional layers (32 → 64 channels), MaxPool, FC 512, Dropout 0.5
- **95.6M parameters** (99.98% in FC layers — severe overfitting)

### ResNet-34
- From-scratch implementation with BasicBlock and skip connections
- Stages: [3, 4, 6, 3] blocks, adaptive average pooling
- **21.3M parameters**, supports ResNet-18/50/101 variants

### Vision Transformer (ViT)
- Patch size 32×32, embedding dim 64, 8 encoder layers, 4 attention heads
- [CLS] token + learnable positional embeddings
- **9.0M parameters** — lightest model, best performance
- Configurable: Small (384d/12L/6H), Base (768d/12L/12H), Large (1024d/24L/16H)

### Pretrained ViT (Transfer Learning)
- `vit_base_patch16_224` from timm library, pretrained on ImageNet
- **85.8M parameters**, best AUC on small data (0.6694)

---

## Loss Functions

| Loss | Description | Best For |
|:-----|:------------|:---------|
| `BCEWithLogitsLoss` | Standard binary cross-entropy | Baseline |
| `FocalLoss` | Down-weights easy examples (α=0.25, γ=2.0) | Class imbalance |
| `WeightedBCELoss` | Per-class weighted BCE | Known class frequencies |
| `AsymmetricLoss` | Different γ for positives/negatives | Multi-label imbalance |
| `DiceLoss` | Overlap-based loss | Segmentation-inspired |
| `CombinedLoss` | Multi-component weighted combination | Advanced training |
| `LabelSmoothingBCE` | Soft labels to prevent overconfidence | Regularization |
| `DistillationLoss` | Knowledge distillation from teacher | Model compression |

Class weight computation supports 4 strategies: `balanced`, `inverse_sqrt`, `effective`, `pos_neg_ratio`.

---

## Configuration System

YAML-based configuration with inheritance:

```yaml
# configs/vit_small.yaml
_base_: base.yaml

model:
  name: vit_small
  embed_dim: 384
  depth: 12
  num_heads: 6

training:
  optimizer: adamw
  lr: 0.0001
  loss: combined
  augmentation: advanced
```

Features:
- Config inheritance via `_base_` key
- Environment variable interpolation: `${ENV_VAR:default}`
- CLI overrides: `--override training.lr=0.001`
- Dotted attribute access: `config.training.lr`

---

## Training Configuration

| Parameter | Value |
|:----------|:------|
| Image size | 224 × 224 |
| Batch size | 32 (16 for pretrained ViT) |
| Classes | 15 |
| Loss | BCEWithLogitsLoss (baseline) |
| Learning rate | 1 × 10⁻⁴ |
| Optimizer | AdamW (SGD for ViT-v2) |
| Epochs | 10 |
| GPU | NVIDIA GeForce RTX 3060 Laptop (6GB) |
| CUDA | 12.6 |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test model architectures
python -m pytest tests/test_models.py -v

# Quick import verification
python main.py verify
```

Tests cover:
- CNN/ResNet/ViT model creation and forward pass
- Gradient computation
- ResNet variant parametrization (18/34/50/101)
- ViT patch embedding dimensions

---

## Notebooks Guide

### Analysis
| Notebook | Description |
|:---------|:------------|
| `data_download.ipynb` | Download NIH ChestX-ray14 via Kaggle API |
| `data.ipynb` | Exploratory data analysis, preprocessing |

### Experiments
| Notebook | Description |
|:---------|:------------|
| `cnn.ipynb` | CNN baseline training & evaluation |
| `resnet.ipynb` | ResNet-34 from scratch |
| `ViT-v1.ipynb` | ViT with Adam optimizer |
| `ViT-v2.ipynb` | ViT with SGD + Early Stopping |
| `ViT-ResNet.ipynb` | Pretrained ViT (timm) transfer learning |
| `Final_ViT_ChestXray.ipynb` | Full-scale ViT on 112K images |

### Improvement Experiments
| Notebook | Topic |
|:---------|:------|
| `01_setup_and_config.ipynb` | Environment & configuration setup |
| `01_transfer_learning.ipynb` | Transfer learning strategies |
| `02_class_imbalance.ipynb` | Handling class imbalance |
| `02_data_augmentation.ipynb` | Advanced augmentation techniques |
| `03_loss_functions.ipynb` | Custom loss function experiments |
| `03_comprehensive_improvements.ipynb` | Full pipeline improvements |
| `04_model_architectures.ipynb` | Architecture ablation studies |
| `05_data_loading.ipynb` | Optimized data pipeline |
| `06_training_infrastructure.ipynb` | Training optimizations |

---

## Documentation

LaTeX reports are available in both Vietnamese and English:

- **Vietnamese**: `docs/report_vi.tex` — compile with `pdflatex`
- **English**: `docs/report_en.tex`
- **Project Proposal**: `docs/Proposal/`

To compile:
```bash
cd docs
pdflatex report_vi.tex
pdflatex report_vi.tex  # Run twice for TOC/references
```

---

## References

1. Jain, A. et al. (2024). *A Comparative Study of CNN, ResNet, and Vision Transformers for Multi-Classification of Chest Diseases*. [arXiv:2406.00237](https://arxiv.org/abs/2406.00237)
2. Wang, X. et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks*. CVPR.
3. Dosovitskiy, A. et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR.
4. He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
5. Rajpurkar, P. et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays*. [arXiv:1711.05225](https://arxiv.org/abs/1711.05225)
6. Lin, T.-Y. et al. (2017). *Focal Loss for Dense Object Detection*. ICCV.
7. Ridnik, T. et al. (2021). *Asymmetric Loss For Multi-Label Classification*. ICCV.
8. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.

---

## Disclaimer

> **Academic Research & Review**
>
> This repository is a **research review and academic study** of the original work:
> - **Original Repository:** [github.com/Aviral-03/ViT-Chest-Xray](https://github.com/Aviral-03/ViT-Chest-Xray)
> - **Original Paper:** [arXiv:2406.00237](https://arxiv.org/abs/2406.00237) — *A Comparative Study of CNN, ResNet, and Vision Transformers for Multi-Classification of Chest Diseases*
> - **Original Authors:** Ananya Jain, Aviral Bhardwaj, Kaushik Murali, Isha Surani (University of Toronto)
>
> This work is conducted for academic purposes as part of the **Master of Software Engineering program at FPT Graduate School (FSB)**. All credit for the original research goes to the original authors.

---

*Last Updated: February 9, 2026*
