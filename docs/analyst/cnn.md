# Phân Tích Chi Tiết: CNN Model cho Chest X-ray Classification

## 📋 Tổng Quan

File `cnn.ipynb` triển khai một mô hình **Convolutional Neural Network (CNN)** đơn giản để phân loại bệnh lý từ hình ảnh X-quang ngực. Đây là bài toán **multi-label classification** với 15 lớp bệnh lý khác nhau.

---

## 🔧 Cell 1: Import Libraries

```python
import glob, os, random, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
warnings.filterwarnings('ignore')
```

### Phân tích chi tiết:

| Thư viện | Mục đích |
|----------|----------|
| `glob, os` | Xử lý file system, tìm kiếm file |
| `numpy, pandas` | Xử lý dữ liệu số và bảng |
| `matplotlib, cv2` | Hiển thị và xử lý ảnh |
| `tensorflow, torch` | Framework deep learning (TF là chính, torch cho device check) |
| `sklearn` | Metrics đánh giá và chia dữ liệu |
| `ImageDataGenerator` | Data augmentation cho training |
| `Callbacks` | Điều khiển quá trình training |

### ⚠️ Nhận xét:
- **Import InceptionResNetV2 nhưng không sử dụng** → code thừa
- **Import cả TensorFlow và PyTorch** → không cần thiết, chỉ dùng TF
- `MultiLabelBinarizer` được import nhưng không sử dụng

---

## 🔧 Cell 2: Load Data

```python
%run data.ipynb
```

### Phân tích:
- Sử dụng magic command `%run` để chạy notebook `data.ipynb`
- Tải các biến từ data.ipynb: `train_generator`, `validation_generator`, `parser`
- **Ưu điểm**: Tái sử dụng code, modular
- **Nhược điểm**: Khó debug, phụ thuộc vào file khác

---

## 🔧 Cell 3: Device Configuration

```python
gpus = tf.config.list_physical_devices('GPU')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    if gpus:
        device = '/GPU:0' 
    else:
        device = '/CPU:0' 
print("Using device:", device)
```

### Phân tích logic:

```
┌─────────────────────────────┐
│   torch.cuda.available?    │
├──────────┬──────────────────┤
│   YES    │       NO         │
│ cuda     │  TF GPU exists?  │
│          ├────────┬─────────┤
│          │  YES   │   NO    │
│          │ /GPU:0 │ /CPU:0  │
└──────────┴────────┴─────────┘
```

### ⚠️ Vấn đề:
1. **Inconsistent device types**: `torch.device` vs TensorFlow string format
2. **Biến `device` không được sử dụng** trong model training
3. TensorFlow tự động sử dụng GPU nếu có, không cần explicit placement

### 💡 Gợi ý cải thiện:
```python
# Chỉ cần cho TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Using GPU: {gpus[0].name}")
else:
    print("Using CPU")
```

---

## 🔧 Cell 4: Hyperparameters

```python
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-6
num_epochs = 10
num_classes = 15
```

### Phân tích chi tiết:

| Parameter | Giá trị | Đánh giá |
|-----------|---------|----------|
| `batch_size` | 32 | ✅ Phù hợp cho hầu hết GPU |
| `learning_rate` | 1e-4 | ✅ Tốt cho Adam optimizer |
| `weight_decay` | 1e-6 | ⚠️ Khá nhỏ, có thể tăng lên 1e-4 |
| `num_epochs` | 10 | ⚠️ Có thể ít cho medical imaging |
| `num_classes` | 15 | ✅ Đúng với dataset NIH Chest X-ray |

### 15 Classes trong Dataset:
1. Cardiomegaly
2. Emphysema
3. Effusion
4. Hernia
5. Nodule
6. Pneumothorax
7. Atelectasis
8. Pleural_Thickening
9. Mass
10. Edema
11. Consolidation
12. Infiltration
13. Fibrosis
14. Pneumonia
15. No Finding

---

## 🔧 Cell 5: CNN Architecture

```python
def create_cnn_classifier():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', name="last_conv_layer"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='sigmoid')
    ])
    return model
```

### Kiến trúc mạng:

```
┌────────────────────────────────────────────────────────┐
│                    INPUT LAYER                         │
│                   (224, 224, 3)                        │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│              Conv2D(32, 3×3, ReLU)                     │
│         Output: (222, 222, 32)                         │
│         Params: (3×3×3+1)×32 = 896                     │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│              MaxPooling2D(2×2)                         │
│         Output: (111, 111, 32)                         │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│              Conv2D(64, 3×3, ReLU)                     │
│         Output: (109, 109, 64)                         │
│         Params: (3×3×32+1)×64 = 18,496                 │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│              MaxPooling2D(2×2)                         │
│         Output: (54, 54, 64)                           │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│                   Flatten()                            │
│         Output: (186,624)                              │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│              Dense(512, ReLU)                          │
│         Params: 186,624×512+512 = 95,552,000          │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│              Dense(15, Sigmoid)                        │
│         Params: 512×15+15 = 7,695                      │
└─────────────────────────┬──────────────────────────────┘
                          │
                    OUTPUT (15)
```

### Tổng số Parameters:
- **Total**: ~95.6 million parameters
- **Vấn đề lớn**: 99%+ params nằm ở Dense layer đầu tiên

### ⚠️ Nhược điểm nghiêm trọng:

1. **Quá đơn giản**: Chỉ 2 conv layers không đủ để extract features phức tạp từ medical images

2. **Bottleneck tại Flatten**: Flatten từ (54,54,64) tạo vector khổng lồ 186,624 chiều

3. **Thiếu các kỹ thuật regularization**:
   - Không có Dropout
   - Không có BatchNormalization
   - Dễ overfitting

4. **Không có Padding**: Sử dụng default `padding='valid'` làm giảm kích thước feature map

5. **Activation cuối là Sigmoid**: ✅ Đúng cho multi-label classification

### 💡 Gợi ý cải thiện:

```python
def create_improved_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', input_shape=(224, 224, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        GlobalAveragePooling2D(),  # Thay vì Flatten
        
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')
    ])
    return model
```

---

## 🔧 Cell 6: Training Function

```python
def run_experiment(model):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ]
    )
    history = model.fit(
        train_generator,
        epochs=num_epochs,
        validation_data=validation_generator,
        callbacks=[ModelCheckpoint(...)]
    )
    return history
```

### Phân tích chi tiết:

#### Optimizer: AdamW
- **Adam with Weight Decay**: Kết hợp Adam với L2 regularization
- **Learning rate**: 1e-4 (phù hợp)
- **Weight decay**: 1e-6 (khá nhỏ)

#### Loss Function: Binary Cross-Entropy
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{C}[y_{ij}\log(\hat{y}_{ij}) + (1-y_{ij})\log(1-\hat{y}_{ij})]$$

- ✅ **Đúng cho multi-label classification**
- Mỗi class được xử lý độc lập

#### Metrics:
| Metric | Mô tả |
|--------|-------|
| `BinaryAccuracy` | % predictions đúng cho mỗi label |
| `AUC` | Area Under ROC Curve |

#### Callback: ModelCheckpoint
- Lưu model khi `val_loss` giảm
- ✅ Best practice

### ⚠️ Thiếu sót:
1. **Không có EarlyStopping** → có thể train quá lâu
2. **Không có ReduceLROnPlateau** → không điều chỉnh learning rate
3. **Không có TensorBoard** → khó visualize training

### 💡 Gợi ý callbacks:
```python
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3),
    ModelCheckpoint('best_model.keras', save_best_only=True),
    TensorBoard(log_dir='./logs')
]
```

---

## 🔧 Cell 7: Execute Training

```python
cnn_classifier = create_cnn_classifier()
history_cnn = run_experiment(cnn_classifier)
```

### Flow:
1. Tạo model CNN
2. Compile với optimizer, loss, metrics
3. Fit trên training data
4. Trả về history để visualize

---

## 🔧 Cell 8: Visualization

```python
def plot_combined_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Loss
    plt.subplot(1, 2, 1)  
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    
    # Plot 2: Accuracy
    plt.subplot(1, 2, 2) 
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
```

### Output dự kiến:

```
┌──────────────────────┬──────────────────────┐
│    LOSS CURVES       │  ACCURACY CURVES     │
│                      │                      │
│  ╲                   │              ___     │
│   ╲___train          │         ___/  val    │
│      ╲___            │    ___/              │
│          ╲val        │  /  train            │
│                      │                      │
│  Epochs →            │  Epochs →            │
└──────────────────────┴──────────────────────┘
```

### Điều cần quan sát:
1. **Overfitting**: Train loss giảm, val loss tăng
2. **Underfitting**: Cả hai loss cao
3. **Good fit**: Cả hai giảm và hội tụ

---

## 📊 Đánh Giá Tổng Thể

### ✅ Điểm mạnh:
1. Code structure rõ ràng, dễ đọc
2. Sử dụng đúng loss function cho multi-label
3. Có model checkpointing
4. Visualization training curves

### ❌ Điểm yếu:

| Vấn đề | Mức độ | Giải pháp |
|--------|--------|-----------|
| Model quá đơn giản | 🔴 Critical | Thêm layers, dùng pretrained |
| Thiếu regularization | 🟠 High | Thêm Dropout, BatchNorm |
| Bottleneck ở Flatten | 🟠 High | Dùng GlobalAveragePooling |
| Thiếu callbacks | 🟡 Medium | Thêm EarlyStopping, ReduceLR |
| Import thừa | 🟢 Low | Cleanup imports |

### 💡 Khuyến nghị:

1. **Sử dụng Transfer Learning**:
```python
base_model = tf.keras.applications.DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
```

2. **Thêm Class Weights** cho imbalanced data:
```python
class_weights = compute_class_weight('balanced', ...)
```

3. **Metrics phù hợp hơn cho Medical Imaging**:
   - Sensitivity/Specificity per class
   - ROC-AUC per class
   - F1-score

---

## 📈 Expected Performance

Với architecture hiện tại, dự kiến:
- **AUC**: 0.55-0.65 (không tốt)
- **Binary Accuracy**: 0.85-0.90 (misleading do class imbalance)

Với improvements:
- **AUC**: 0.75-0.85
- Cần pretrained model để đạt SOTA (~0.85-0.90)

---

## 📚 References

1. NIH Chest X-ray Dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC
2. CheXNet Paper: Rajpurkar et al., 2017
3. TensorFlow Documentation: https://www.tensorflow.org/api_docs
