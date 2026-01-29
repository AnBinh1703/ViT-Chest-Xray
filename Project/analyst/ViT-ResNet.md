# Phân Tích Chi Tiết: Vision Transformer Pre-trained (PyTorch)

## 📋 Tổng Quan

File `ViT-ResNet.ipynb` (mặc dù tên có "ResNet" nhưng thực tế là **ViT Pre-trained**) sử dụng:
- **PyTorch** thay vì TensorFlow
- **timm library** với pretrained ViT
- **Transfer Learning** từ ImageNet

Đây là approach **production-ready** và **SOTA** cho medical imaging.

---

## 🔄 So sánh với ViT-v1/v2

| Feature | ViT-v1/v2 | ViT-ResNet |
|---------|-----------|------------|
| Framework | TensorFlow | **PyTorch** |
| Pretrained | ❌ | ✅ ImageNet |
| Library | Custom | **timm** |
| Model | ViT-Tiny (~3M) | **ViT-Base** (~86M) |
| Dataset class | Keras Generator | **Custom Dataset** |

---

## 🔧 Cell 1: Imports

```python
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import timm

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
```

### Key Libraries:

| Library | Purpose |
|---------|---------|
| `torch` | PyTorch deep learning framework |
| `timm` | PyTorch Image Models - pretrained models |
| `torchvision.transforms` | Image augmentation |
| `DataLoader` | Efficient batch loading |

### `timm` Library:

```
┌─────────────────────────────────────────────────────────────────┐
│                      TIMM LIBRARY                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PyTorch Image Models (timm):                                  │
│  - 800+ pretrained models                                      │
│  - ViT, DeiT, Swin, ConvNeXt, EfficientNet, etc.              │
│  - Easy API: timm.create_model('model_name', pretrained=True)  │
│                                                                 │
│  Available ViT models:                                          │
│  - vit_tiny_patch16_224                                        │
│  - vit_small_patch16_224                                       │
│  - vit_base_patch16_224  ← Used here                          │
│  - vit_large_patch16_224                                       │
│  - vit_huge_patch14_clip_224                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Cell 2: Custom Dataset Class

```python
class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label
```

### PyTorch Dataset Pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                   PYTORCH DATASET CLASS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  class MyDataset(Dataset):                                      │
│      │                                                          │
│      ├── __init__(): Initialize with data paths, labels        │
│      │                                                          │
│      ├── __len__(): Return dataset size                        │
│      │              Used by DataLoader                          │
│      │                                                          │
│      └── __getitem__(idx): Return single (image, label) pair   │
│                            Called during iteration              │
│                                                                 │
│  DataLoader(dataset) → Iterator over batches                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Flow:

```python
dataset = ChestXRayDataset(paths, labels, transform)
loader = DataLoader(dataset, batch_size=16)

for images, labels in loader:
    # images: (16, 3, 224, 224)
    # labels: (16, 15)
    model(images)
```

---

## 🔧 Cell 3-4: Data Loading

```python
data_path = '/Users/ananyajain/Desktop/.../sample'
images_dir = 'sample/images'
labels_csv = 'sample_labels.csv'

# Load and process labels
labels_df = pd.read_csv(os.path.join(data_path, labels_csv))
new_labels_df['labels'] = labels_df['Finding Labels'].apply(lambda val: val.split('|'))

# Multi-label binarization
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(new_labels_df['labels'])
labels = np.array(labels, dtype=float)
```

### MultiLabelBinarizer:

```python
mlb = MultiLabelBinarizer()

# Input
labels_list = [
    ['Cardiomegaly', 'Effusion'],
    ['No Finding'],
    ['Pneumonia', 'Infiltration', 'Effusion']
]

# Output (one-hot encoded)
mlb.fit_transform(labels_list)
# array([[1, 0, 1, 0, ...],   # Cardiomegaly + Effusion
#        [0, 0, 0, 1, ...],   # No Finding
#        [0, 1, 1, 0, ...]])  # Pneumonia + Infiltration + Effusion
```

---

## 🔧 Cell 5: Image Transforms

```python
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])
```

### Transform Pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                   TRANSFORM PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NumPy Array (H, W, 3)                                         │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                           │
│  │  ToPILImage()   │  Convert to PIL Image                     │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  Resize(224)    │  Resize to 224×224                        │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ RandomHorizFlip │  50% chance horizontal flip               │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ RandomRotation  │  Rotate ±20 degrees                       │
│  │     (20)        │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  ToTensor()     │  PIL → Tensor, scale to [0,1]             │
│  │                 │  (H,W,C) → (C,H,W)                        │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  Normalize()    │  ImageNet normalization                   │
│  │  μ=[.485,.456,  │  x = (x - mean) / std                     │
│  │     .406]       │                                           │
│  │  σ=[.229,.224,  │                                           │
│  │     .225]       │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  Tensor (3, 224, 224) - Ready for model                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ImageNet Normalization:

```
mean = [0.485, 0.456, 0.406]  # RGB means
std  = [0.229, 0.224, 0.225]  # RGB stds

# These values are computed from ImageNet dataset
# Required when using pretrained ImageNet models!
```

---

## 🔧 Cell 6: Data Split & Loaders

```python
train_paths, val_test_paths, train_labels, val_test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    val_test_paths, val_test_labels, test_size=0.5, random_state=42)

train_dataset = ChestXRayDataset(train_paths, train_labels, transform)
val_dataset = ChestXRayDataset(val_paths, val_labels, transform)
test_dataset = ChestXRayDataset(test_paths, test_labels, transform)

batch_size = 16
loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

### Split Ratio:

```
Total Data
    │
    ├── 80% train
    │
    └── 20% val+test
            │
            ├── 50% → 10% val
            │
            └── 50% → 10% test

Final: 80% train, 10% val, 10% test
```

### ⚠️ Note:
- `batch_size=16` (thay vì 32) do ViT-Base lớn hơn, cần nhiều GPU memory

---

## 🔧 Cell 7: Device Configuration

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, 
      f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
```

### PyTorch Device Management:

```python
# Move model to GPU
model.to(device)

# Move data to GPU (in training loop)
inputs, labels = inputs.to(device), labels.to(device)
```

---

## 🔧 Cell 8: Pretrained ViT Model

```python
model = timm.create_model('vit_base_patch16_224', pretrained=True)
num_classes = 15
model.head = nn.Linear(model.head.in_features, num_classes)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-4)
```

### timm Model Creation:

```python
timm.create_model(
    'vit_base_patch16_224',  # Model name
    pretrained=True          # Use ImageNet weights
)
```

### Model Name Breakdown:

```
vit_base_patch16_224
 │    │      │    │
 │    │      │    └── Input size: 224×224
 │    │      └── Patch size: 16×16
 │    └── Size variant: base (86M params)
 └── Architecture: Vision Transformer
```

### ViT-Base Architecture:

| Component | Value |
|-----------|-------|
| Patch size | 16×16 |
| Num patches | 14×14 = 196 |
| Hidden dim | 768 |
| Num heads | 12 |
| Num layers | 12 |
| MLP ratio | 4 |
| Parameters | ~86M |

### Transfer Learning - Replace Classification Head:

```python
# Original head (ImageNet: 1000 classes)
model.head = nn.Linear(768, 1000)

# Replace with our head (Chest X-ray: 15 classes)
model.head = nn.Linear(model.head.in_features, 15)
```

```
┌─────────────────────────────────────────────────────────────────┐
│                   TRANSFER LEARNING                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────┐                 │
│  │           PRETRAINED ViT-Base             │                 │
│  │  ┌─────────────────────────────────────┐  │                 │
│  │  │      Patch Embedding                │  │ ← Frozen or    │
│  │  │      Position Embedding             │  │   Fine-tuned   │
│  │  │      Transformer Blocks ×12         │  │                 │
│  │  │      (768-dim representations)      │  │                 │
│  │  └─────────────────────────────────────┘  │                 │
│  └───────────────────────────────────────────┘                 │
│                         │                                       │
│                         ▼                                       │
│  ┌───────────────────────────────────────────┐                 │
│  │    NEW CLASSIFICATION HEAD                │ ← Trained       │
│  │    Linear(768 → 15)                       │   from scratch  │
│  │    + Sigmoid (via BCEWithLogitsLoss)      │                 │
│  └───────────────────────────────────────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Loss Function: BCEWithLogitsLoss

```python
criterion = nn.BCEWithLogitsLoss()
```

$$\text{BCEWithLogitsLoss} = -\frac{1}{N}\sum_{i}[y_i \log(\sigma(x_i)) + (1-y_i)\log(1-\sigma(x_i))]$$

**Tại sao dùng BCEWithLogitsLoss thay vì BCELoss?**
- Kết hợp Sigmoid + BCE trong một operation
- Numerically stable hơn
- Không cần sigmoid activation trong model

---

## 🔧 Cell 9: Training Function

```python
def train_model(model, criterion, optimizer, loader_train, loader_val, num_epochs=10):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in loader_train:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.sigmoid() > 0.5

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels.byte()).sum().item()
            total_samples += labels.numel()

        epoch_loss = running_loss / len(loader_train.dataset)
        epoch_acc = running_corrects / total_samples * 100
        
        val_loss, val_acc = validate_model(model, loader_val, criterion)
        
    return train_losses, val_losses, train_accuracies, val_accuracies
```

### PyTorch Training Loop:

```
┌─────────────────────────────────────────────────────────────────┐
│                   PYTORCH TRAINING LOOP                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  for epoch in range(num_epochs):                               │
│      │                                                          │
│      ├── model.train()          # Enable dropout, batchnorm    │
│      │                                                          │
│      └── for batch in loader:                                  │
│              │                                                  │
│              ├── inputs, labels = batch.to(device)             │
│              │                                                  │
│              ├── optimizer.zero_grad()   # Clear gradients     │
│              │                                                  │
│              ├── outputs = model(inputs) # Forward pass        │
│              │                                                  │
│              ├── loss = criterion(outputs, labels)             │
│              │                                                  │
│              ├── loss.backward()         # Compute gradients   │
│              │                                                  │
│              └── optimizer.step()        # Update weights      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Accuracy Calculation:

```python
preds = outputs.sigmoid() > 0.5  # Threshold at 0.5
running_corrects += (preds == labels.byte()).sum().item()
total_samples += labels.numel()  # Total number of labels (batch × 15)
```

**Note**: Đây là **element-wise accuracy** (mỗi label được đánh giá riêng)

---

## 🔧 Cell 10: Validation Function

```python
def validate_model(model, loader_val, criterion, threshold=0.5):
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader_val:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predicted = outputs.sigmoid() > threshold
            # ...
```

### model.eval() vs model.train():

| Mode | Dropout | BatchNorm |
|------|---------|-----------|
| `model.train()` | Active | Use batch stats |
| `model.eval()` | Disabled | Use running stats |

### torch.no_grad():

```python
with torch.no_grad():
    # Disable gradient computation
    # Saves memory, faster inference
    outputs = model(inputs)
```

---

## 🔧 Cell 11-12: Training & Visualization

```python
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, criterion, optimizer, loader_train, loader_val, num_epochs=10
)

# 4-panel visualization
plt.subplot(2, 2, 1)  # Train loss
plt.subplot(2, 2, 2)  # Val loss
plt.subplot(2, 2, 3)  # Train accuracy
plt.subplot(2, 2, 4)  # Val accuracy
```

---

## 📊 Đánh Giá Tổng Thể

### ✅ Điểm mạnh:
1. ✅ **Pretrained ViT-Base** - SOTA model
2. ✅ **PyTorch + timm** - Industry standard
3. ✅ **Proper ImageNet normalization**
4. ✅ **Custom Dataset class** - Flexible
5. ✅ **BCEWithLogitsLoss** - Numerically stable
6. ✅ **Correct training loop** - zero_grad, backward, step

### ❌ Điểm yếu:

| Vấn đề | Mức độ | Giải pháp |
|--------|--------|-----------|
| Hardcoded path | 🔴 Critical | Use config/env vars |
| No learning rate scheduler | 🟠 High | Add CosineAnnealingLR |
| No early stopping | 🟠 High | Add early stopping logic |
| No model saving | 🟠 High | torch.save() best model |
| Same transform for val | 🟡 Medium | No augmentation for val |
| batch_size=16 | 🟡 Medium | Gradient accumulation |

---

## 💡 Improved Version:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# Separate transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Better optimizer setup
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# Training with scheduler
for epoch in range(num_epochs):
    # ... training code ...
    scheduler.step()
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
```

---

## 📊 Expected Performance

| Metric | ViT-v1 (scratch) | ViT-Pretrained |
|--------|------------------|----------------|
| Train Accuracy | 95% | 92% |
| Val Accuracy | 75% | **88%** |
| Test AUC | 0.65 | **0.82-0.85** |
| Training Time | 10 epochs | 10 epochs |
| Convergence | Slow | **Fast** |

---

## 📚 Why Pretrained is Better?

```
┌─────────────────────────────────────────────────────────────────┐
│              PRETRAINED vs FROM SCRATCH                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FROM SCRATCH:                                                  │
│  - Random initialization                                        │
│  - Learn everything from chest X-rays only                     │
│  - Need LOTS of data (>100K images)                            │
│  - Risk of overfitting on small datasets                       │
│                                                                 │
│  PRETRAINED (ImageNet):                                         │
│  - Already knows low-level features (edges, textures)          │
│  - Already knows mid-level features (shapes, patterns)         │
│  - Only needs to learn chest X-ray specific patterns           │
│  - Works well even with small datasets (~1000 images)          │
│                                                                 │
│  Medical Imaging Bonus:                                         │
│  - ImageNet contains some medical-like images                  │
│  - Transfer learning consistently outperforms scratch          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📚 References

1. timm library: https://github.com/huggingface/pytorch-image-models
2. "An Image is Worth 16x16 Words", Dosovitskiy et al., 2020
3. PyTorch Transfer Learning Tutorial
4. "Big Transfer (BiT): General Visual Representation Learning", Kolesnikov et al., 2020
