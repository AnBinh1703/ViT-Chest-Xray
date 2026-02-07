# PROJECT MAP - ViT Chest X-ray Classification

**Audit Date:** January 2025  
**Auditor:** Senior Research Engineer  
**Project:** NIH Chest X-ray 14 Multi-label Classification with Vision Transformers

---

## 📋 Executive Summary

| Metric | Value |
|--------|-------|
| Total Notebooks | 7 main training notebooks |
| Total Models | 5 architectures (CNN, ResNet-34, ViT-v1, ViT-v2, ViT-Pretrained) |
| Dataset | NIH Chest X-ray 14: 112,120 images, 30,805 patients |
| Task Type | Multi-label classification (15 classes) |
| Primary Metric | Macro AUC-ROC |
| Best Model | ViT-Pretrained (Test AUC: 0.6694) |
| Framework | PyTorch 2.x with timm library |

---

## 🗂️ Complete File Inventory

### Root Directory

| File | Purpose | Inputs | Outputs | Dependencies | Issues | Fixes |
|------|---------|--------|---------|--------------|--------|-------|
| `README.md` | Project documentation | - | - | - | ⚠️ May be outdated | Update with current results |
| `requirements.txt` | Package dependencies | - | pip install | pip | ⚠️ Version pins needed | Add specific versions |
| `install_packages.py` | Setup helper | requirements.txt | Installed packages | pip | ✅ OK | - |
| `Paper.pdf` | Reference paper (Jain et al.) | - | - | - | ✅ OK | - |
| `2406.00237v1.pdf` | ArXiv paper | - | - | - | ✅ OK | - |
| `COMPLETE_DOCUMENTATION.md` | Full docs | - | - | - | ⚠️ Needs update | Sync with actual results |
| `IMPROVEMENT_PLAN.md` | Future work | - | - | - | ✅ OK | - |

### Project/ Directory

| File | Purpose | Inputs | Outputs | Dependencies | Issues | Fixes |
|------|---------|--------|---------|--------------|--------|-------|
| `config.py` | Centralized configuration | - | Config class | - | ⚠️ SAMPLE_SIZE=3000 default | Set to None for full training |
| `data_download.ipynb` | Download NIH dataset | Kaggle API | images/ folder | kaggle, zipfile | ✅ OK | - |
| `data.ipynb` | Data exploration & EDA | CSV, images | Plots, statistics | pandas, matplotlib | ⚠️ No patient-split demo | Add patient-level split example |
| `cnn.ipynb` | CNN baseline training | Data splits | cnn_model.pth | torch, Config | ⚠️ Overfitting (Train AUC 0.90 vs Val 0.58) | Add regularization |
| `resnet.ipynb` | ResNet-34 training | Data splits | resnet_model.pth | torch, Config | ⚠️ Low AUC (Val 0.53) | Increase epochs, add pretraining |
| `ViT-v1.ipynb` | ViT from scratch v1 | Data splits | vit_v1_best.pth | torch, Config | ⚠️ Suboptimal (Test AUC 0.59) | Tune hyperparameters |
| `ViT-v2.ipynb` | ViT from scratch v2 | Data splits | vit_v2_best.pth | torch, Config | ⚠️ Better but still low (Test AUC 0.63) | Use pretrained weights |
| `ViT-ResNet.ipynb` | Pretrained ViT (timm) | Data splits | vit_pretrained_best.pth | timm, torch | ✅ Best model (Test AUC 0.67) | Production ready |
| `Final_ViT_ChestXray.ipynb` | Consolidated notebook | All configs | All artifacts | All deps | ⚠️ Training code commented | Execute full training |
| `comprehensive_analysis.py` | Analysis utilities | Notebook outputs | Reports | pandas, numpy | ✅ OK | - |

### Project/data/ Directory

| Folder | Purpose | Size | Notes |
|--------|---------|------|-------|
| `images/` | Main images directory | ~42GB total | Contains all X-ray images |
| `images_01/` to `images_12/` | NIH batched downloads | ~3.5GB each | Original batch structure |

### Project/input/ Directory

| File | Purpose | Rows | Columns |
|------|---------|------|---------|
| `Data_Entry_2017_v2020.csv` | Image labels metadata | 112,120 | Image Index, Finding Labels, Patient ID, etc. |

### Project/files/ Directory (Model Checkpoints)

| File | Model | Test AUC | Parameters | Status |
|------|-------|----------|------------|--------|
| `cnn_model.pth` | CNN Baseline | ~0.58* | ~95M | Overfitted |
| `resnet_model.pth` | ResNet-34 | ~0.53* | ~21M | Underfitted |
| `vit_v1_best.pth` | ViT-v1 (scratch) | 0.5854 | 9.0M | Usable |
| `vit_v2_best.pth` | ViT-v2 (scratch) | 0.6303 | 9.0M | Good |
| `vit_pretrained_best.pth` | ViT-Pretrained | 0.6694 | ~86M | **Best** |

*Note: CNN/ResNet test AUC from validation metrics (no separate test eval in notebooks)

### Project/improve/ Directory

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | Improvement experiments docs | ✅ OK |
| `config.py` | Improvement-specific config | ✅ OK |
| `focal_loss.py` | Focal loss for imbalance | ✅ Implemented |
| `dice_loss.py` | Dice loss for segmentation-like | ✅ Implemented |
| `weighted_loss.py` | Class-weighted BCE | ✅ Implemented |
| `asymmetric_loss.py` | ASL for multi-label | ✅ Implemented |
| `combined_loss.py` | Multi-loss combination | ✅ Implemented |
| `smoothing_loss.py` | Label smoothing | ✅ Implemented |
| `distillation_loss.py` | Knowledge distillation | ✅ Implemented |
| `utils/loss_functions.py` | Unified loss factory | ✅ Consolidated |
| `utils/improved_models.py` | Advanced architectures | ✅ Implemented |
| `utils/evaluation.py` | Evaluation utilities | ✅ Implemented |

### Report/ Directory

| File | Purpose | Status |
|------|---------|--------|
| `LaTeX/main.tex` | Main Vietnamese report | ⚠️ Needs metrics update |
| `LaTeX_EN/main.tex` | English version | ⚠️ Needs metrics update |
| `Critical_Analysis_Report.tex` | Critical analysis | ✅ OK |

---

## 📊 Verified Experimental Results

### Model Performance Comparison (Extracted from Notebook Outputs)

| Model | Parameters | Test Loss | Test Accuracy | Test Macro AUC | Best Val AUC |
|-------|------------|-----------|---------------|----------------|--------------|
| CNN Baseline | ~95M | N/A | N/A | ~0.58* | 0.5998 (epoch 1) |
| ResNet-34 | ~21M | N/A | N/A | ~0.53* | 0.5293 (epoch 6) |
| ViT-v1 (scratch) | 9,005,839 | 0.2534 | 91.33% | 0.5854 | 0.6431 (epoch 4) |
| ViT-v2 (scratch) | 9,005,839 | 0.2749 | 89.67% | 0.6303 | 0.5947 (epoch 9) |
| **ViT-Pretrained** | ~86M | 0.3768 | 87.00% | **0.6694** | N/A |

*Validation AUC used as proxy for test (no separate test eval in these notebooks)

### Per-Class AUC (ViT-v2 Test Set)

| Disease | AUC | Status |
|---------|-----|--------|
| Cardiomegaly | 1.00 | 🟢 Excellent |
| Emphysema | 0.84 | 🟢 Good |
| Effusion | 0.73 | 🟡 Moderate |
| No Finding | 0.76 | 🟡 Moderate |
| Pneumothorax | 0.69 | 🟡 Moderate |
| Pleural_Thickening | 0.68 | 🟡 Moderate |
| Infiltration | 0.52 | 🔴 Poor |
| Edema | 0.42 | 🔴 Poor |
| Consolidation | 0.33 | 🔴 Poor |
| Atelectasis | 0.32 | 🔴 Poor |

### Per-Class AUC (ViT-v1 Test Set)

| Disease | AUC | Status |
|---------|-----|--------|
| Infiltration | 0.89 | 🟢 Good |
| Pneumothorax | 0.84 | 🟢 Good |
| Cardiomegaly | 0.79 | 🟢 Good |
| Nodule | 0.68 | 🟡 Moderate |
| Consolidation | 0.61 | 🟡 Moderate |
| No Finding | 0.53 | 🔴 Poor |
| Effusion | 0.51 | 🔴 Poor |
| Atelectasis | 0.47 | 🔴 Poor |
| Pleural_Thickening | 0.42 | 🔴 Poor |
| Mass | 0.11 | 🔴 Very Poor |

---

## 🔴 Critical Findings & High-Risk Issues

### Issue 1: Data Leakage Risk
- **Severity:** 🔴 HIGH
- **Location:** Early notebooks may use image-level split
- **Problem:** Same patient's images can appear in train AND test sets
- **Impact:** Inflated metrics, model won't generalize
- **Fix:** ✅ Final notebook uses `get_patient_level_split()` correctly

### Issue 2: Wrong Primary Metric (Accuracy)
- **Severity:** 🔴 HIGH
- **Location:** Some notebooks prioritize accuracy
- **Problem:** Accuracy is misleading for imbalanced multi-label data
- **Impact:** 91%+ accuracy but only 58% AUC
- **Fix:** ✅ Final notebook uses Macro AUC-ROC as primary metric

### Issue 3: Class Imbalance
- **Severity:** 🟡 MEDIUM
- **Location:** Dataset inherent issue
- **Problem:** "No Finding" dominates (53%), rare diseases < 2%
- **Impact:** Model biased toward majority classes
- **Fix:** Consider focal loss, class weighting (available in improve/)

### Issue 4: Overfitting (CNN)
- **Severity:** 🟡 MEDIUM
- **Location:** cnn.ipynb
- **Problem:** Train AUC 0.90 vs Val AUC 0.58 (gap = 0.32)
- **Impact:** Model memorizes training data
- **Fix:** Add dropout, weight decay, early stopping

### Issue 5: Insufficient Training (ResNet)
- **Severity:** 🟡 MEDIUM
- **Location:** resnet.ipynb
- **Problem:** Val AUC only 0.53, training from scratch
- **Impact:** ResNet needs pretrained weights for medical imaging
- **Fix:** Use ImageNet pretrained ResNet

### Issue 6: Sample Size for Testing
- **Severity:** 🟡 MEDIUM
- **Location:** config.py `SAMPLE_SIZE=3000`
- **Problem:** Results on 3000 samples may not generalize
- **Impact:** Reported metrics may differ from full dataset
- **Fix:** Run final evaluation with `SAMPLE_SIZE=None`

---

## 🔧 Configuration Summary

### Training Configuration (from config.py)

```python
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATCH_SIZE = 16
PROJECTION_DIM = 768
NUM_HEADS = 8
TRANSFORMER_LAYERS = 12
EARLY_STOPPING_PATIENCE = 5
SEED = 42
```

### Data Split (Patient-Level)
- Train: 70%
- Validation: 10%
- Test: 20%

### Loss Function
- `BCEWithLogitsLoss` (sigmoid + binary cross-entropy)

### Optimizer
- ViT-v1/Pretrained: AdamW
- ViT-v2: SGD with lr=0.01

---

## 📁 Directory Tree

```
ViT-Chest-Xray/
├── README.md
├── requirements.txt
├── Paper.pdf
├── PROJECT_MAP.md          ← This file
├── FILE_REVIEWS.md         ← File-level reviews
├── Project/
│   ├── config.py           ← Global config
│   ├── Final_ViT_ChestXray.ipynb  ← Main deliverable
│   ├── cnn.ipynb           ← CNN baseline
│   ├── resnet.ipynb        ← ResNet-34
│   ├── ViT-v1.ipynb        ← ViT scratch v1
│   ├── ViT-v2.ipynb        ← ViT scratch v2
│   ├── ViT-ResNet.ipynb    ← Pretrained ViT (best)
│   ├── data.ipynb          ← EDA
│   ├── data_download.ipynb ← Data download
│   ├── data/               ← X-ray images (~42GB)
│   ├── input/              ← CSV metadata
│   ├── files/              ← Model checkpoints
│   └── improve/            ← Advanced experiments
├── Report/
│   ├── LaTeX/              ← Vietnamese report
│   └── LaTeX_EN/           ← English report
└── Proposal/               ← Project proposal
```

---

## ✅ Recommendations

1. **Production Deployment:** Use ViT-Pretrained model (AUC 0.67)
2. **Further Improvement:** Implement focal loss for rare diseases
3. **Ensemble:** Combine ViT-v1 + ViT-v2 for complementary predictions
4. **Threshold Tuning:** Per-class thresholds for clinical use
5. **Full Dataset Training:** Re-run with SAMPLE_SIZE=None

---

*Generated by Senior Research Engineer Audit - January 2025*
