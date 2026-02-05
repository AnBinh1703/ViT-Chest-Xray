# 📝 DETAILED FILE REVIEWS - ViT Chest X-ray Classification

**Reviewer:** Lead Research Engineer + Code Reviewer  
**Date:** Auto-generated from codebase analysis

---

## Review Template

For each file:
- **A. Purpose**: What the file does
- **B. Input**: What data/dependencies it requires  
- **C. Output**: What it produces
- **D. Key Implementation Details**: Technical specifics
- **E. Results/Performance**: Metrics and outputs
- **F. Issues & Risks**: Problems identified
- **G. Suggested Fixes**: Recommendations
- **Score**: 0-10 (10 = production-ready)

---

## 1. `config.py`

### A. Purpose
Centralized configuration file storing all project paths, hyperparameters, and dataset labels for the chest X-ray classification project.

### B. Input
- None (static configuration)

### C. Output
- `PROJECT_ROOT`: Root directory path
- `DATA_ROOT`: Data folder path
- `IMAGES_DIR`: Images folder path
- `LABELS_CSV`: Labels CSV path
- `LABELS`: List of 15 disease labels
- Hyperparameters: `IMAGE_SIZE=224`, `BATCH_SIZE=32`, `NUM_EPOCHS=10`, `LR=1e-4`, etc.

### D. Key Implementation Details
```python
# Labels list (15 classes)
LABELS = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Nodule',
          'Pneumothorax', 'Atelectasis', 'Pleural_Thickening', 'Mass',
          'Edema', 'Consolidation', 'Infiltration', 'Fibrosis', 
          'Pneumonia', 'No Finding']

# Key hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
```

### E. Results/Performance
- N/A (configuration file)

### F. Issues & Risks
1. **Hardcoded Windows paths**: `D:\MSE\10.Deep Learning\...`
2. **No environment variable support**: Paths not configurable
3. **Duplicate configs**: `improve/config.py` also exists

### G. Suggested Fixes
1. Use `pathlib.Path` with relative paths
2. Add `os.environ.get()` for overrides
3. Consolidate into single config module

### Score: 6/10
Basic functionality works but not portable across systems.

---

## 2. `data_download.ipynb`

### A. Purpose
Download the NIH Chest X-ray dataset from Kaggle using kagglehub API.

### B. Input
- Kaggle API credentials (pre-configured)
- Dataset identifier: `nih-chest-xrays/data`

### C. Output
- Downloaded images (~42GB, 112,120 images)
- `Data_Entry_2017_v2020.csv` labels file

### D. Key Implementation Details
```python
import kagglehub
# Download dataset
path = kagglehub.dataset_download("nih-chest-xrays/data")
```

### E. Results/Performance
- Downloads full NIH Chest X-ray 14 dataset
- Requires significant storage (~50GB with extraction)

### F. Issues & Risks
1. **No progress indicator**: Downloads without progress feedback
2. **No checksum verification**: Data integrity not verified
3. **Hardcoded dataset path**: Not configurable
4. **No partial download support**: Must restart if interrupted

### G. Suggested Fixes
1. Add tqdm progress bars for download monitoring
2. Implement checksum verification after download
3. Add resume capability for interrupted downloads
4. Parameterize dataset selection

### Score: 5/10
Basic download works but lacks robustness features.

---

## 3. `data.ipynb`

### A. Purpose
Data loading, preprocessing, and DataLoader creation for multi-label chest X-ray classification.

### B. Input
- `Data_Entry_2017_v2020.csv`: Labels file
- Raw images from `/data/images_*/images/` folders

### C. Output
- `DatasetParser` class for label parsing
- `ChestXrayDataset` PyTorch Dataset class
- `train_loader`, `val_loader`, `test_loader` DataLoaders

### D. Key Implementation Details
```python
# DatasetParser class
class DatasetParser:
    def __init__(self, data_root, labels_csv, labels_list):
        # Parse multi-labels from "Finding Labels" column
        # Labels separated by '|' (e.g., "Effusion|Infiltration")
        
# ChestXrayDataset class
class ChestXrayDataset(Dataset):
    def __getitem__(self, idx):
        # Load image with cv2
        # Convert BGR to RGB
        # Apply transforms
        # Return one-hot encoded labels

# Train/Val/Test split: 60/20/20
# ⚠️ Image-level split (not patient-level)
```

### E. Results/Performance
- Successfully loads and processes dataset
- Applies augmentation: HorizontalFlip, Rotation, ColorJitter
- Creates balanced batches with multi-label support

### F. Issues & Risks
1. **⚠️ CRITICAL: Image-level split causes data leakage**
   - Same patient's images may appear in train AND test sets
   - Inflates test metrics artificially
2. **Horizontal flip may affect diagnosis**: Medical images have anatomical orientation
3. **No stratified sampling**: Class imbalance not handled in split
4. **Missing images not gracefully handled**: May crash on missing files

### G. Suggested Fixes
1. **Implement patient-level split**:
```python
# Extract Patient ID from filename (format: 00000001_000.png)
df['Patient ID'] = df['Image Index'].apply(lambda x: x.split('_')[0])
# Split by Patient ID, not Image Index
train_patients, test_patients = train_test_split(unique_patients, ...)
```
2. Reduce horizontal flip probability to 0.1 or remove
3. Add stratified sampling for validation set
4. Add try-catch for missing images

### Score: 5/10
Critical data leakage issue significantly impacts reliability of results.

---

## 4. `cnn.ipynb`

### A. Purpose
Baseline CNN model for multi-label chest X-ray classification.

### B. Input
- DataLoaders from `data.ipynb`
- config hyperparameters

### C. Output
- `cnn_model.pth` checkpoint (~380MB due to large dense layer)
- Training history (loss, accuracy, AUC)

### D. Key Implementation Details
```python
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=15):
        # 2 convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # ⚠️ ISSUE: Huge flatten layer
        # 224 -> 222 -> 111 -> 109 -> 54
        # Flatten: 64 * 54 * 54 = 186,624 features
        # Dense: 186,624 * 512 = 95.5M parameters!
        
        self.fc1 = nn.Linear(64 * 54 * 54, 512)
        self.fc2 = nn.Linear(512, num_classes)

# Loss: BCEWithLogitsLoss (multi-label)
# Optimizer: AdamW(lr=1e-4, weight_decay=1e-6)
```

### E. Results/Performance
- Parameters: ~95M (mostly in fc1 layer)
- Typical AUC: ~0.65 (baseline)
- Training time: Fast (simple architecture)

### F. Issues & Risks
1. **Excessive parameters**: 95M params for simple CNN is inefficient
2. **No global pooling**: Uses flatten instead of adaptive pooling
3. **No batch normalization**: Training may be unstable
4. **No dropout between conv layers**: May overfit

### G. Suggested Fixes
```python
# Replace flatten with Global Average Pooling
self.features = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1))  # Global Avg Pool
)
# Now only 64 features -> 64 * 512 = 32K params
```

### Score: 4/10
Working baseline but inefficient architecture with excessive parameters.

---

## 5. `resnet.ipynb`

### A. Purpose
ResNet-34 implementation from scratch for chest X-ray classification.

### B. Input
- DataLoaders from `data.ipynb`
- config hyperparameters

### C. Output
- `resnet_model.pth` checkpoint (~85MB)
- Training history

### D. Key Implementation Details
```python
class BasicBlock(nn.Module):
    """ResNet Basic Block with skip connection"""
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity  # Skip connection
        return self.relu(out)

class ResNet(nn.Module):
    """ResNet-34: [3, 4, 6, 3] blocks"""
    def __init__(self):
        self.layer1 = self._make_layer(64, 3)   # 3 blocks
        self.layer2 = self._make_layer(128, 4)  # 4 blocks
        self.layer3 = self._make_layer(256, 6)  # 6 blocks
        self.layer4 = self._make_layer(512, 3)  # 3 blocks
        self.fc = nn.Linear(512, num_classes)
```

### E. Results/Performance
- Parameters: ~21M (standard ResNet-34)
- Typical AUC: ~0.72
- Training time: Medium

### F. Issues & Risks
1. **From scratch training**: No pretrained ImageNet weights
2. **No learning rate warmup**: May have unstable initial training
3. **Fixed architecture**: No easy modification for experiments

### G. Suggested Fixes
1. Use `torchvision.models.resnet34(pretrained=True)` for better initialization
2. Add learning rate warmup scheduler
3. Consider ResNet-18 for faster iteration

### Score: 7/10
Solid implementation but missing pretrained weights significantly limits performance.

---

## 6. `ViT-v1.ipynb`

### A. Purpose
Vision Transformer implementation from scratch (version 1).

### B. Input
- DataLoaders from `data.ipynb`
- config hyperparameters

### C. Output
- `vit_v1_best.pth` checkpoint (~12MB)
- Training history

### D. Key Implementation Details
```python
class PatchEmbedding(nn.Module):
    """Split image into patches and embed"""
    def __init__(self, img_size=224, patch_size=32, embed_dim=64):
        # 224 / 32 = 7 -> 7 * 7 = 49 patches
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

class TransformerEncoderBlock(nn.Module):
    """Standard Transformer encoder with multi-head attention"""
    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Self-attention + residual
        x = x + self.mlp(self.ln2(x))   # MLP + residual
        return x

class VisionTransformer(nn.Module):
    """Full ViT model"""
    def __init__(self, depth=8, num_heads=4, embed_dim=64):
        self.patch_embed = PatchEmbedding(...)
        self.pos_embed = nn.Parameter(...)  # Learnable position embeddings
        self.blocks = nn.ModuleList([TransformerEncoderBlock(...) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Linear(embed_dim * num_patches, 2048),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.GELU(), 
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
```

### E. Results/Performance
- Parameters: ~3M (lightweight)
- Typical AUC: ~0.68
- Training time: Medium

### F. Issues & Risks
1. **No learning rate scheduler**: Fixed LR throughout training
2. **No early stopping**: May overtrain
3. **Flatten all patches**: Loses spatial structure vs. CLS token
4. **Small embedding dimension**: 64 may be too small for complex patterns

### G. Suggested Fixes
1. Add `ReduceLROnPlateau` or `CosineAnnealingLR` scheduler
2. Implement early stopping callback
3. Consider using CLS token instead of flattening all patches
4. Increase embed_dim to 128 or 256

### Score: 6/10
Good implementation but lacks training optimization features.

---

## 7. `ViT-v2.ipynb`

### A. Purpose
Improved Vision Transformer with learning rate scheduler and early stopping.

### B. Input
- DataLoaders from `data.ipynb`
- config hyperparameters

### C. Output
- `vit_v2_best.pth` checkpoint
- Training history with scheduler info

### D. Key Implementation Details
```python
# Same ViT architecture as v1, but with:

# 1. Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=3
)

# 2. Early stopping
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    torch.save(model.state_dict(), save_path)
else:
    patience_counter += 1
    if patience_counter >= patience:
        break

# 3. Multiple optimizer options (commented)
# optimizer = optim.SGD(params, lr=0.01, momentum=0.9)
# optimizer = optim.Adam(params, lr=1e-4)
# optimizer = optim.AdamW(params, lr=1e-4, weight_decay=1e-5)
```

### E. Results/Performance
- Parameters: ~3M (same as v1)
- Typical AUC: ~0.70 (improved over v1)
- Training time: Medium (may stop early)

### F. Issues & Risks
1. **Early stopping by loss, not AUC**: May stop when AUC is still improving
2. **Default SGD optimizer**: Adam/AdamW often better for transformers
3. **Fixed patience=5**: May be too aggressive for some runs

### G. Suggested Fixes
1. Monitor validation AUC instead of loss for early stopping
2. Default to AdamW optimizer
3. Make patience configurable

### Score: 7/10
Good improvements over v1, minor optimization opportunities.

---

## 8. `ViT-ResNet.ipynb`

### A. Purpose
Pretrained Vision Transformer using timm library for transfer learning.

### B. Input
- DataLoaders from `data.ipynb`
- timm pretrained model

### C. Output
- `vit_pretrained_best.pth` checkpoint (~340MB)
- Training history

### D. Key Implementation Details
```python
import timm

# Load pretrained ViT-Base with 16x16 patches
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Replace classification head for our task
model.head = nn.Linear(model.head.in_features, NUM_CLASSES)

# AUC calculation with NaN handling
def compute_auc(targets, outputs):
    valid_classes = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) > 1:  # Need both 0 and 1
            valid_classes.append(i)
    
    if len(valid_classes) > 0:
        return roc_auc_score(
            targets[:, valid_classes],
            outputs[:, valid_classes],
            average='macro'
        )
    return 0.0
```

### E. Results/Performance
- Parameters: ~86M (ViT-Base)
- Typical AUC: ~0.78 (best among all models)
- Training time: Slow (large model + pretrained features)

### F. Issues & Risks
1. **High memory usage**: 86M params requires significant GPU memory
2. **Slow training**: Each epoch takes longer
3. **Potential overfitting**: Large model on limited data

### G. Suggested Fixes
1. Use gradient checkpointing for memory efficiency
2. Consider ViT-Small or ViT-Tiny for faster iteration
3. Add more regularization (dropout, label smoothing)
4. Implement mixed precision training (fp16)

### Score: 8/10
Best performing model with solid implementation. Main concerns are resource usage.

---

## 📊 Summary Scores

| File | Score | Status |
|------|-------|--------|
| `config.py` | 6/10 | ⚠️ Needs portability fixes |
| `data_download.ipynb` | 5/10 | ⚠️ Basic, lacks robustness |
| `data.ipynb` | 5/10 | 🔴 Critical: Data leakage risk |
| `cnn.ipynb` | 4/10 | ⚠️ Inefficient architecture |
| `resnet.ipynb` | 7/10 | ✅ Solid, needs pretrained weights |
| `ViT-v1.ipynb` | 6/10 | ⚠️ Needs training optimizations |
| `ViT-v2.ipynb` | 7/10 | ✅ Good improvements |
| `ViT-ResNet.ipynb` | 8/10 | ✅ Best model, resource-heavy |

**Average Score: 6.0/10**

---

## 🎯 Priority Actions

### Critical (Must Fix)
1. **Implement patient-level split** in `data.ipynb`

### High Priority
2. Replace flatten with GlobalAvgPool in `cnn.ipynb`
3. Use pathlib for cross-platform paths in `config.py`

### Medium Priority
4. Add early stopping by AUC in `ViT-v2.ipynb`
5. Consider pretrained ResNet in `resnet.ipynb`
6. Add progress indicators in `data_download.ipynb`

### Low Priority
7. Add gradient checkpointing for memory efficiency
8. Consolidate duplicate configs
9. Add comprehensive documentation

---

*Generated from codebase analysis*
