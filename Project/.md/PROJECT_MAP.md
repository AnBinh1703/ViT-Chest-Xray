# 🗺️ PROJECT MAP - ViT Chest X-ray Classification

**Generated:** Auto-generated from codebase analysis  
**Reference Paper:** [arXiv:2406.00237](https://arxiv.org/abs/2406.00237)  
**Original Repo:** [Aviral-03/ViT-Chest-Xray](https://github.com/Aviral-03/ViT-Chest-Xray)

---

## 📂 Complete File Inventory

| # | Folder | File | Purpose | Input | Output | Issues | Suggested Fix |
|---|--------|------|---------|-------|--------|--------|---------------|
| 1 | `/Project` | `config.py` | Centralized configuration | None | Config constants | Hardcoded paths (Windows) | Use pathlib, relative paths |
| 2 | `/Project` | `data.ipynb` | Data loading & parsing | CSV + images | DataLoaders | Image-level split (leakage risk) | Implement patient-level split |
| 3 | `/Project` | `data_download.ipynb` | Dataset download | Kaggle API | Raw images (42GB) | No progress indicator | Add tqdm progress bars |
| 4 | `/Project` | `cnn.ipynb` | Baseline CNN model | DataLoaders | cnn_model.pth | Huge flatten layer (~95M params) | Add GlobalAvgPool |
| 5 | `/Project` | `resnet.ipynb` | ResNet-34 from scratch | DataLoaders | resnet_model.pth | No pretrained weights | Consider torchvision.models |
| 6 | `/Project` | `ViT-v1.ipynb` | ViT scratch v1 | DataLoaders | vit_v1_best.pth | No learning rate scheduler | Add scheduler |
| 7 | `/Project` | `ViT-v2.ipynb` | ViT scratch v2 + optimizations | DataLoaders | vit_v2_best.pth | Early stopping may trigger too soon | Tune patience |
| 8 | `/Project` | `ViT-ResNet.ipynb` | Pretrained ViT (timm) | DataLoaders | vit_pretrained_best.pth | Heavy memory usage (~86M params) | Gradient checkpointing |
| 9 | `/Project/input` | `Data_Entry_2017_v2020.csv` | NIH labels | None | Labels DataFrame | Large file | Index by Image Index |
| 10 | `/Project/files` | `*.pth` | Model checkpoints | Trained models | Inference | No version tracking | Add model metadata |
| 11 | `/Project/improve` | `config.py` | Improvement config | None | Extended config | Duplicates Project/config.py | Consolidate configs |
| 12 | `/Project/improve` | `focal_loss.py` | Focal loss function | Predictions, targets | Loss value | Fixed gamma=2 | Parameterize gamma |
| 13 | `/Project/improve` | `weighted_loss.py` | Weighted BCE loss | Predictions, targets | Loss value | Manual weight calculation | Auto-compute from data |
| 14 | `/Project/improve` | `asymmetric_loss.py` | Asymmetric loss | Predictions, targets | Loss value | Not well documented | Add docstrings |
| 15 | `/Project/improve` | `01_setup_and_config.ipynb` | Setup guide | None | Environment setup | Incomplete | Finish setup steps |
| 16 | `/Project/improve` | `02_data_augmentation.ipynb` | Advanced augmentation | DataLoaders | Augmented data | HorizontalFlip concern | Add medical imaging note |
| 17 | `/Project/improve` | `03_comprehensive_improvements.ipynb` | All improvements combined | Multiple | Comprehensive results | Overwhelming complexity | Split into modules |
| 18 | `/Report/LaTeX` | `main.tex` | Full LaTeX report | None | PDF report | Vietnamese only | Add English version |
| 19 | `/Report/LaTeX/chapters` | `*.tex` | Report chapters | None | Section content | Incomplete experiments section | Add more results |

---

## 🏗️ Architecture Overview

```
ViT-Chest-Xray/
├── Project/                          # Main implementation
│   ├── config.py                     # 📋 Configuration (paths, hyperparams)
│   ├── data.ipynb                    # 📊 Data pipeline
│   ├── data_download.ipynb           # ⬇️ Download NIH dataset
│   ├── cnn.ipynb                     # 🧠 CNN baseline (~95M params)
│   ├── resnet.ipynb                  # 🧠 ResNet-34 (~21M params)
│   ├── ViT-v1.ipynb                  # 🧠 ViT scratch v1 (~3M params)
│   ├── ViT-v2.ipynb                  # 🧠 ViT scratch v2 + scheduler
│   ├── ViT-ResNet.ipynb              # 🧠 Pretrained ViT (~86M params)
│   ├── Final_ViT_ChestXray.ipynb     # ✨ CONSOLIDATED FINAL NOTEBOOK
│   ├── files/                        # 💾 Model checkpoints
│   ├── input/                        # 📁 Dataset (CSV + images)
│   └── improve/                      # 🔧 Improvement experiments
│       ├── *.py                      # Loss functions, utilities
│       └── *.ipynb                   # Improvement notebooks
├── Report/                           # 📄 LaTeX reports
│   └── LaTeX/                        # Main report structure
└── results/                          # 📈 Experiment outputs
```

---

## 🔢 Key Configuration Parameters

| Parameter | Value | Source | Notes |
|-----------|-------|--------|-------|
| `IMAGE_SIZE` | 224 | config.py | Standard ViT input size |
| `BATCH_SIZE` | 32 | config.py | Limited by GPU memory |
| `NUM_EPOCHS` | 10 | config.py | Early stopping may reduce |
| `LEARNING_RATE` | 1e-4 | config.py | Adam/AdamW default |
| `WEIGHT_DECAY` | 1e-6 | config.py | L2 regularization |
| `PATCH_SIZE` | 32 | config.py | 224/32 = 7x7 = 49 patches |
| `PROJECTION_DIM` | 64 | config.py | Embedding dimension |
| `NUM_HEADS` | 4 | config.py | Multi-head attention |
| `TRANSFORMER_LAYERS` | 8 | config.py | Encoder depth |
| `NUM_CLASSES` | 15 | config.py | 14 diseases + No Finding |

---

## 📊 Dataset Summary

| Attribute | Value | Source |
|-----------|-------|--------|
| Total Images | 112,120 | Paper/README |
| Total Patients | 30,805 | Paper |
| Classes | 15 (multi-label) | config.py |
| Image Size | 1024x1024 (original) | NIH dataset |
| Processed Size | 224x224 | config.py |
| Storage | ~42GB | README |
| Split Ratio | 60/20/20 | data.ipynb |
| Split Type | Image-level ⚠️ | data.ipynb (should be patient-level) |

---

## 🧠 Model Comparison

| Model | Parameters | Architecture | Training Time* | Best AUC** |
|-------|------------|--------------|----------------|------------|
| CNN Baseline | ~95M | 2 Conv + Dense | Fast | ~0.65 |
| ResNet-34 | ~21M | 34-layer residual | Medium | ~0.72 |
| ViT-v1 (scratch) | ~3M | 8-layer, 4-head | Medium | ~0.68 |
| ViT-v2 (scratch) | ~3M | + scheduler, early stop | Medium | ~0.70 |
| ViT-Pretrained | ~86M | vit_base_patch16_224 | Slow | ~0.78 |

*Relative comparison on similar hardware  
**Approximate values from notebook outputs (may vary with hyperparameters)

---

## ⚠️ Critical Issues Identified

### 1. Data Leakage Risk (HIGH)
- **File:** `data.ipynb`
- **Issue:** Image-level split instead of patient-level split
- **Impact:** Same patient's images may appear in train and test sets
- **Fix:** Implement patient-level split using Patient ID

### 2. Horizontal Flip Augmentation (MEDIUM)
- **File:** `data.ipynb`, `improve/02_data_augmentation.ipynb`
- **Issue:** Medical images have anatomical orientation (heart on left)
- **Impact:** Flipped images may confuse model
- **Fix:** Remove or reduce flip probability, add documentation

### 3. Hardcoded Paths (LOW)
- **File:** `config.py`
- **Issue:** Windows-specific paths like `D:\MSE\...`
- **Impact:** Breaks on other systems
- **Fix:** Use pathlib with relative paths from project root

### 4. AUC NaN Handling (FIXED)
- **File:** `ViT-ResNet.ipynb`
- **Issue:** roc_auc_score fails when class has only one label value
- **Status:** Fixed with valid_classes check

---

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
opencv-python>=4.8.0
pillow>=9.0.0
tqdm>=4.65.0
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (requires Kaggle API)
# Run data_download.ipynb

# 3. Run final notebook
# Open Project/Final_ViT_ChestXray.ipynb
```

---

*Last updated: Auto-generated*
