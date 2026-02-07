# Phân Tích Chi Tiết: ResNet Model

## 📋 Tổng Quan

File `resnet.ipynb` triển khai mô hình **Residual Network (ResNet)** từ đầu để phân loại bệnh lý từ X-quang ngực. ResNet giải quyết vấn đề **vanishing gradient** trong deep networks thông qua **skip connections**.

---

## 🔧 Cell 1-3: Setup (Giống CNN)

```python
# Import libraries
# %run data.ipynb
# Device configuration
```

*(Tương tự như trong cnn.ipynb)*

---

## 🔧 Cell 4: Hyperparameters

```python
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-6
num_epochs = 10
input_shape = (224, 224, 3) 
num_classes = 15
```

| Parameter | Giá trị | Đánh giá |
|-----------|---------|----------|
| `batch_size` | 32 | ✅ Standard |
| `learning_rate` | 1e-4 | ✅ Phù hợp |
| `weight_decay` | 1e-6 | ⚠️ Có thể tăng |
| `num_epochs` | 10 | ⚠️ Ít cho ResNet |
| `input_shape` | (224,224,3) | ✅ Standard ImageNet size |

---

## 🔧 Cell 5: Residual Block

```python
def block(x, filters, strides=1):
    identity = x
    x = Conv2D(filters, 3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    if strides != 1 or identity.shape[-1] != filters:
        identity = Conv2D(filters, 1, strides=strides, padding='same')(identity)
        identity = BatchNormalization()(identity)
    
    x += identity
    x = Activation('relu')(x)
    return x
```

### Kiến trúc Residual Block:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RESIDUAL BLOCK                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    Input (x)                                                    │
│       │                                                         │
│       ├────────────────────────────┐                           │
│       │                            │                            │
│       ▼                            │ (Identity/Shortcut)        │
│  ┌─────────────┐                   │                            │
│  │ Conv2D 3×3  │                   │                            │
│  │ (filters)   │                   │                            │
│  └──────┬──────┘                   │                            │
│         │                          │                            │
│         ▼                          │                            │
│  ┌─────────────┐                   │                            │
│  │ BatchNorm   │                   │                            │
│  └──────┬──────┘                   │                            │
│         │                          │                            │
│         ▼                          │                            │
│  ┌─────────────┐                   │                            │
│  │    ReLU     │                   │                            │
│  └──────┬──────┘                   │                            │
│         │                          │                            │
│         ▼                          │                            │
│  ┌─────────────┐                   │                            │
│  │ Conv2D 3×3  │                   │  ┌─────────────────┐       │
│  │ (filters)   │                   │  │ 1×1 Conv        │       │
│  └──────┬──────┘                   │  │ (if dimension   │       │
│         │                          │  │  mismatch)      │       │
│         ▼                          │  └────────┬────────┘       │
│  ┌─────────────┐                   │           │                │
│  │ BatchNorm   │                   │           │                │
│  └──────┬──────┘                   │           │                │
│         │                          │           │                │
│         └──────────────┬───────────┴───────────┘                │
│                        │                                        │
│                        ▼                                        │
│                   ┌─────────┐                                   │
│                   │   ADD   │  ← Skip Connection                │
│                   └────┬────┘                                   │
│                        │                                        │
│                        ▼                                        │
│                   ┌─────────┐                                   │
│                   │  ReLU   │                                   │
│                   └────┬────┘                                   │
│                        │                                        │
│                    Output                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Công thức toán học:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

Trong đó:
- $\mathbf{x}$: input
- $\mathcal{F}(\mathbf{x}, \{W_i\})$: residual mapping (2 conv layers)
- $\mathbf{y}$: output

### Identity Shortcut Projection:

```python
if strides != 1 or identity.shape[-1] != filters:
    identity = Conv2D(filters, 1, strides=strides, padding='same')(identity)
    identity = BatchNormalization()(identity)
```

**Khi nào cần projection?**
1. `strides != 1`: Khi downsample (giảm spatial dimension)
2. `identity.shape[-1] != filters`: Khi số channels khác nhau

---

## 🔧 Cell 6: Full ResNet Architecture

```python
def create_resnet():
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # Stage 1: 64 filters
    x = block(x, 64)
    x = block(x, 64)
    x = block(x, 64)

    # Stage 2: 128 filters  
    x = block(x, 128, 2)  # Downsample
    x = block(x, 128)
    x = block(x, 128)
    x = block(x, 128)

    # Stage 3: 256 filters
    x = block(x, 256, 2)  # Downsample
    x = block(x, 256)
    x = block(x, 256)
    x = block(x, 256)
    x = block(x, 256)
    x = block(x, 256)

    # Stage 4: 512 filters
    x = block(x, 512, 2)  # Downsample
    x = block(x, 512)
    x = block(x, 512)
    
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model
```

### Kiến trúc đầy đủ:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ResNet-34 Style                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: (224, 224, 3)                                          │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ STEM: Conv 7×7, 64 filters, stride 2                    │   │
│  │       BatchNorm → ReLU → MaxPool 3×3, stride 2          │   │
│  │       Output: (56, 56, 64)                              │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │ STAGE 1: 3 blocks × 64 filters                          │   │
│  │          Output: (56, 56, 64)                           │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │ STAGE 2: 4 blocks × 128 filters (first block stride=2)  │   │
│  │          Output: (28, 28, 128)                          │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │ STAGE 3: 6 blocks × 256 filters (first block stride=2)  │   │
│  │          Output: (14, 14, 256)                          │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │ STAGE 4: 3 blocks × 512 filters (first block stride=2)  │   │
│  │          Output: (7, 7, 512)                            │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │ GlobalAveragePooling2D                                   │   │
│  │          Output: (512,)                                 │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │ Dense(15, sigmoid)                                       │   │
│  │          Output: (15,) - Multi-label probabilities      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Số blocks theo stage:

| Stage | Filters | Blocks | Total Layers |
|-------|---------|--------|--------------|
| 1 | 64 | 3 | 6 conv |
| 2 | 128 | 4 | 8 conv |
| 3 | 256 | 6 | 12 conv |
| 4 | 512 | 3 | 6 conv |
| **Total** | - | **16 blocks** | **32 conv + 2 = 34 layers** |

→ Đây là kiến trúc **ResNet-34** style!

### So sánh với ResNet gốc:

| Variant | Blocks (per stage) | Total Layers |
|---------|-------------------|--------------|
| ResNet-18 | [2, 2, 2, 2] | 18 |
| **This model** | **[3, 4, 6, 3]** | **~34** |
| ResNet-34 | [3, 4, 6, 3] | 34 |
| ResNet-50 | [3, 4, 6, 3] + Bottleneck | 50 |

### Tính số Parameters:

```
STEM:
  Conv 7×7×3×64 + bias = 9,472
  
STAGE 1 (64 filters, 3 blocks):
  Each block: 2 × (3×3×64×64) = 73,728
  Total: 3 × 73,728 = 221,184
  
STAGE 2 (128 filters, 4 blocks):
  First block with projection: ~200K
  Other blocks: 3 × (2 × 3×3×128×128) = 884,736
  
... (tương tự cho stage 3, 4)

Dense: 512 × 15 = 7,680

TOTAL: ~21.3 million parameters
```

---

## 🔧 So sánh ResNet với CNN đơn giản

| Aspect | Simple CNN | ResNet |
|--------|-----------|--------|
| **Depth** | 2 conv layers | 34 layers |
| **Parameters** | ~95M (bottleneck ở Dense) | ~21M |
| **Skip connections** | ❌ | ✅ |
| **BatchNorm** | ❌ | ✅ |
| **Gradient flow** | Vanishing | Healthy |
| **Expected AUC** | 0.55-0.65 | 0.70-0.80 |

---

## 🔧 Training Function

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

### ⚠️ Missing imports:

Code sử dụng nhưng không import:
- `GlobalAveragePooling2D`
- `Dense`
- `Model`
- `keras`

### 💡 Fix:
```python
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, 
    MaxPooling2D, GlobalAveragePooling2D, Dense, Add
)
from tensorflow.keras.models import Model
from tensorflow import keras
```

---

## 📊 Vanishing Gradient Problem & Solution

### Vấn đề với Deep Networks:

```
Layer 1 → Layer 2 → ... → Layer 30
   │         │              │
   ▼         ▼              ▼
∂L/∂W₁    ∂L/∂W₂    ...   ∂L/∂W₃₀

Gradient chain rule:
∂L/∂W₁ = ∂L/∂W₃₀ × ∂W₃₀/∂W₂₉ × ... × ∂W₂/∂W₁

Nếu mỗi term < 1 → gradient tiến về 0
Nếu mỗi term > 1 → gradient explodes
```

### ResNet Solution:

```python
x += identity  # Skip connection

# Gradient của skip connection:
∂L/∂x = ∂L/∂y × (∂F/∂x + 1)  # Luôn có +1!
```

**Kết quả**: Gradient luôn có ít nhất là 1, không bao giờ vanish!

---

## 📊 Đánh Giá Tổng Thể

### ✅ Điểm mạnh:
1. ✅ Kiến trúc ResNet đúng chuẩn
2. ✅ Skip connections cho better gradient flow
3. ✅ BatchNormalization ở mọi layer
4. ✅ GlobalAveragePooling thay vì Flatten
5. ✅ Sigmoid activation cho multi-label

### ❌ Điểm yếu:

| Vấn đề | Mức độ | Giải pháp |
|--------|--------|-----------|
| Missing imports | 🔴 Critical | Add proper imports |
| Không dùng pretrained weights | 🟠 High | Use ImageNet weights |
| Thiếu Dropout | 🟡 Medium | Add after GAP |
| weight_decay quá nhỏ | 🟡 Medium | Increase to 1e-4 |

### 💡 Improved Version:

```python
def create_resnet_improved():
    # Use pretrained ResNet50
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    # Fine-tune last few layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model
```

---

## 📚 ResNet Family

| Model | Params | Top-1 Acc (ImageNet) |
|-------|--------|---------------------|
| ResNet-18 | 11.7M | 69.8% |
| ResNet-34 | 21.8M | 73.3% |
| ResNet-50 | 25.6M | 76.1% |
| ResNet-101 | 44.5M | 77.4% |
| ResNet-152 | 60.2M | 78.3% |

---

## 📚 References

1. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
2. He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016
3. TensorFlow ResNet Documentation
