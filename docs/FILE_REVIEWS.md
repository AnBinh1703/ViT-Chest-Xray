# FILE REVIEWS - ViT Chest X-ray Classification

**Audit Date:** January 2025  
**Auditor:** Senior Research Engineer  
**Scoring Scale:** A (Excellent) → G (Critical Issues)

---

## Scoring Rubric

| Grade | Description |
|-------|-------------|
| **A** | Production-ready, follows best practices, no issues |
| **B** | Minor improvements needed, functionally correct |
| **C** | Moderate issues, works but needs refactoring |
| **D** | Significant issues, partially working |
| **E** | Major problems, requires substantial fixes |
| **F** | Critical bugs, does not achieve intended purpose |
| **G** | Fundamentally broken, needs complete rewrite |

---

## 1. data.ipynb

**Location:** [Project/data.ipynb](Project/data.ipynb)  
**Purpose:** Data exploration and preprocessing  
**Grade:** **C+**

### Strengths ✅
- Comprehensive EDA with visualizations
- Class distribution analysis
- Sample image display
- Multi-label encoding demonstration

### Issues ⚠️
1. **No patient-level split demonstration** - Critical for medical imaging
2. **Limited statistical analysis** - Missing correlation between diseases
3. **No data quality checks** - Missing image validation
4. **Hardcoded paths** - Should use config.py

### Extracted Metrics
- Total images: 112,120
- Unique patients: 30,805
- Classes: 15 (including "No Finding")
- Dominant class: "No Finding" (~53%)

### Recommendations
```python
# Add patient-level split demonstration
parser = DatasetParser(data_root, labels_csv, labels_list)
train_df, val_df, test_df = parser.get_patient_level_split(
    test_size=0.2, val_size=0.125, random_state=42
)
print(f"Patient overlap check: {set(train_df['Patient ID']) & set(test_df['Patient ID'])}")
```

---

## 2. cnn.ipynb

**Location:** [Project/cnn.ipynb](Project/cnn.ipynb)  
**Purpose:** CNN baseline model training  
**Grade:** **D+**

### Strengths ✅
- Simple architecture for baseline comparison
- Uses BCEWithLogitsLoss (correct for multi-label)
- Training loop implemented correctly

### Issues ⚠️
1. **🔴 Severe overfitting** - Train AUC 0.90 vs Val AUC 0.58 (gap = 0.32)
2. **No regularization** - Missing dropout in conv layers
3. **No early stopping** - Continues training despite overfitting
4. **Best Val AUC at epoch 1** - Model degrades after first epoch

### Extracted Metrics (from notebook output)
| Epoch | Train Loss | Train AUC | Val Loss | Val AUC |
|-------|------------|-----------|----------|---------|
| 1 | 0.2451 | 0.5855 | 0.2299 | **0.5998** |
| 10 | 0.1219 | 0.8961 | 0.2614 | 0.5847 |

### Architecture
```python
CNN: Conv2d(3,32) → ReLU → MaxPool → Conv2d(32,64) → ReLU → MaxPool → FC(512) → FC(15)
Parameters: ~95M (mostly from FC layer)
```

### Recommendations
1. Add dropout after conv layers
2. Implement early stopping (patience=3)
3. Use data augmentation
4. Consider smaller FC layer to reduce parameters

---

## 3. resnet.ipynb

**Location:** [Project/resnet.ipynb](Project/resnet.ipynb)  
**Purpose:** ResNet-34 training from scratch  
**Grade:** **D**

### Strengths ✅
- Proper ResNet-34 implementation with BasicBlocks
- Correct residual connections
- Uses BCEWithLogitsLoss

### Issues ⚠️
1. **🔴 Training from scratch** - No pretrained weights (critical mistake)
2. **Low performance** - Val AUC only 0.53 (barely above random)
3. **Overfitting observed** - Train AUC 0.78 vs Val AUC 0.52
4. **No learning rate scheduler** - Could improve convergence

### Extracted Metrics (from notebook output)
| Epoch | Train Loss | Train AUC | Val Loss | Val AUC |
|-------|------------|-----------|----------|---------|
| 1 | 0.2556 | 0.5350 | 0.2375 | 0.5069 |
| 6 | 0.2203 | 0.6689 | 0.2273 | **0.5293** |
| 10 | 0.1973 | 0.7768 | 0.2333 | 0.5235 |

### Architecture
```python
ResNet34: Conv7x7 → BN → ReLU → MaxPool → 
          Layer1(3 blocks) → Layer2(4 blocks) → Layer3(6 blocks) → Layer4(3 blocks) →
          AdaptiveAvgPool → FC(15)
Parameters: ~21.3M
```

### Recommendations
```python
# Use pretrained ResNet-34
import torchvision.models as models
resnet = models.resnet34(pretrained=True)
resnet.fc = nn.Linear(512, 15)  # Modify output layer
```

---

## 4. ViT-v1.ipynb

**Location:** [Project/ViT-v1.ipynb](Project/ViT-v1.ipynb)  
**Purpose:** Vision Transformer from scratch (version 1)  
**Grade:** **C**

### Strengths ✅
- Complete ViT implementation from scratch
- Proper multi-head self-attention
- Positional embeddings included
- BCEWithLogitsLoss for multi-label

### Issues ⚠️
1. **Moderate performance** - Test AUC 0.5854
2. **High variance per class** - AUC ranges from 0.11 (Mass) to 0.89 (Infiltration)
3. **No learning rate scheduler** - Training could be improved
4. **Val AUC peaks early** - Best at epoch 4, then degrades

### Extracted Metrics (from notebook output)
| Metric | Value |
|--------|-------|
| Parameters | 9,005,839 |
| Test Loss | 0.2534 |
| Test Accuracy | 91.33% |
| Test Macro AUC | 0.5854 |
| Best Val AUC | 0.6431 (epoch 4) |

### Per-Class AUC (Test Set)
| Disease | AUC | Assessment |
|---------|-----|------------|
| Infiltration | 0.89 | 🟢 Excellent |
| Pneumothorax | 0.84 | 🟢 Good |
| Cardiomegaly | 0.79 | 🟢 Good |
| Nodule | 0.68 | 🟡 Moderate |
| Consolidation | 0.61 | 🟡 Moderate |
| No Finding | 0.53 | 🔴 Poor |
| Effusion | 0.51 | 🔴 Poor |
| Atelectasis | 0.47 | 🔴 Poor |
| Pleural_Thickening | 0.42 | 🔴 Poor |
| Mass | 0.11 | 🔴 Very Poor |

### Architecture
```python
ViT: PatchEmbed(16x16) → [ClassToken + PosEmbed] → 
     TransformerEncoder(12 layers, 8 heads, dim=768) →
     LayerNorm → FC(15)
```

### Recommendations
1. Implement ReduceLROnPlateau scheduler
2. Use focal loss for rare classes
3. Add gradient clipping
4. Consider smaller patch size (8x8) for fine-grained features

---

## 5. ViT-v2.ipynb

**Location:** [Project/ViT-v2.ipynb](Project/ViT-v2.ipynb)  
**Purpose:** Vision Transformer from scratch (version 2, improved)  
**Grade:** **B-**

### Strengths ✅
- Improved training with SGD optimizer
- ReduceLROnPlateau scheduler
- Early stopping implemented (patience=5)
- Better macro AUC than v1

### Issues ⚠️
1. **Still suboptimal** - Test AUC 0.63 vs pretrained 0.67
2. **Class imbalance persists** - Some diseases have AUC < 0.50
3. **SGD with lr=0.01** - Aggressive for ViT (usually lr=1e-4)
4. **Early stopped at epoch 9** - Could train longer

### Extracted Metrics (from notebook output)
| Metric | Value |
|--------|-------|
| Parameters | 9,005,839 |
| Test Loss | 0.2749 |
| Test Accuracy | 89.67% |
| Test Macro AUC | **0.6303** |
| Best Val AUC | 0.5947 (epoch 9) |

### Per-Class AUC (Test Set)
| Disease | AUC | Assessment |
|---------|-----|------------|
| Cardiomegaly | 1.00 | 🟢 Perfect |
| Emphysema | 0.84 | 🟢 Good |
| No Finding | 0.76 | 🟡 Moderate |
| Effusion | 0.73 | 🟡 Moderate |
| Pneumothorax | 0.69 | 🟡 Moderate |
| Pleural_Thickening | 0.68 | 🟡 Moderate |
| Infiltration | 0.52 | 🔴 Poor |
| Edema | 0.42 | 🔴 Poor |
| Consolidation | 0.33 | 🔴 Poor |
| Atelectasis | 0.32 | 🔴 Poor |

### Training Configuration
```python
optimizer = SGD(lr=0.01, momentum=0.9)
scheduler = ReduceLROnPlateau(mode='min', factor=0.1, patience=3)
early_stopping_patience = 5
```

### Recommendations
1. Try AdamW with lr=1e-4 (standard for ViT)
2. Increase warmup epochs
3. Use mixup/cutmix augmentation
4. Consider label smoothing

---

## 6. ViT-ResNet.ipynb

**Location:** [Project/ViT-ResNet.ipynb](Project/ViT-ResNet.ipynb)  
**Purpose:** Pretrained Vision Transformer using timm library  
**Grade:** **A-**

### Strengths ✅
- Uses pretrained vit_base_patch16_224 from timm
- Best performing model (Test AUC 0.6694)
- Proper transfer learning approach
- ImageNet pretraining transfers well to medical imaging

### Issues ⚠️
1. **Higher loss than scratch models** - Test loss 0.3768 vs 0.25-0.27
2. **Lower accuracy** - 87% vs 89-91% (but AUC is higher!)
3. **Missing per-class AUC breakdown** - Need detailed analysis
4. **Large model size** - ~86M parameters

### Extracted Metrics (from notebook output)
| Metric | Value |
|--------|-------|
| Parameters | ~86M |
| Test Loss | 0.3768 |
| Test Accuracy | 87.00% |
| Test Macro AUC | **0.6694** ✅ Best |

### Architecture
```python
# timm pretrained ViT
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=15)
```

### Recommendations
1. Fine-tune only last few layers initially
2. Implement gradual unfreezing
3. Use smaller learning rate for pretrained layers
4. Add per-class AUC logging

---

## 7. config.py

**Location:** [Project/config.py](Project/config.py)  
**Purpose:** Centralized configuration management  
**Grade:** **B**

### Strengths ✅
- Centralized hyperparameters
- Path management with pathlib
- Class variable approach for easy access
- Serialization to JSON supported

### Issues ⚠️
1. **SAMPLE_SIZE=3000 by default** - Should be None for production
2. **No validation of paths** - Could fail silently
3. **Hardcoded seed** - Should be configurable
4. **Missing some hyperparameters** - warmup, gradient clipping

### Configuration Values
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
SAMPLE_SIZE = 3000  # ⚠️ Should be None
```

### Recommendations
```python
# Add validation
@classmethod
def validate(cls):
    assert cls.LABELS_CSV.exists(), f"Labels CSV not found: {cls.LABELS_CSV}"
    assert cls.IMAGES_DIR.exists(), f"Images dir not found: {cls.IMAGES_DIR}"
    
# Add configurable seed
SEED = int(os.getenv('RANDOM_SEED', 42))
```

---

## 8. improve/utils/loss_functions.py

**Location:** [Project/improve/utils/loss_functions.py](Project/improve/utils/loss_functions.py)  
**Purpose:** Advanced loss functions for multi-label classification  
**Grade:** **A-**

### Strengths ✅
- Comprehensive collection of loss functions
- Focal loss for class imbalance
- Dice loss for overlap optimization
- Asymmetric loss for multi-label
- Factory pattern for easy selection
- Well-documented with formulas

### Issues ⚠️
1. **Not integrated into main notebooks** - Experiments only
2. **Missing ablation study** - Which loss is best for chest X-ray?
3. **Some losses not tested** - Knowledge distillation untested
4. **Hyperparameters not tuned** - Focal gamma, alpha defaults

### Available Loss Functions
| Loss | Class | Purpose |
|------|-------|---------|
| FocalLoss | `FocalLoss` | Handle class imbalance |
| DiceLoss | `DiceLoss` | Overlap-based optimization |
| WeightedBCE | `WeightedBCELoss` | Per-class weights |
| AsymmetricLoss | `AsymmetricLoss` | Multi-label specific |
| LabelSmoothing | `LabelSmoothingLoss` | Regularization |
| CombinedLoss | `CombinedLoss` | Multi-loss ensemble |

### Usage Example
```python
from improve.utils.loss_functions import get_loss_function

criterion = get_loss_function(
    loss_type='focal',
    gamma=2.0,
    alpha=0.25
)
```

### Recommendations
1. Run ablation study on all loss functions
2. Tune focal loss gamma for chest X-ray
3. Test combined BCE + Dice for rare classes
4. Integrate best loss into Final notebook

---

## Summary Table

| File | Grade | Key Issue | Fix Priority |
|------|-------|-----------|--------------|
| data.ipynb | C+ | No patient-split demo | 🟡 Medium |
| cnn.ipynb | D+ | Severe overfitting | 🔴 High |
| resnet.ipynb | D | No pretrained weights | 🔴 High |
| ViT-v1.ipynb | C | Suboptimal AUC | 🟡 Medium |
| ViT-v2.ipynb | B- | Class imbalance | 🟡 Medium |
| ViT-ResNet.ipynb | A- | Missing per-class details | 🟢 Low |
| config.py | B | SAMPLE_SIZE default | 🟡 Medium |
| loss_functions.py | A- | Not integrated | 🟡 Medium |

---

## Recommendations Priority

### 🔴 High Priority
1. Fix overfitting in CNN (add regularization)
2. Use pretrained ResNet weights
3. Set SAMPLE_SIZE=None for production

### 🟡 Medium Priority
4. Integrate focal loss into training
5. Add patient-level split to EDA
6. Tune ViT learning rate

### 🟢 Low Priority
7. Add per-class AUC to pretrained ViT
8. Run loss function ablation study
9. Update documentation

---

*Generated by Senior Research Engineer Audit - January 2025*
