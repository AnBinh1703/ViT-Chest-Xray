# Phân Tích Chi Tiết: Vision Transformer V1

## 📋 Tổng Quan

File `ViT-v1.ipynb` triển khai **Vision Transformer (ViT)** từ đầu - một kiến trúc đột phá áp dụng Transformer (vốn dùng cho NLP) vào Computer Vision. ViT chia ảnh thành các patches và xử lý chúng như sequence tokens.

---

## 🎯 Vision Transformer Concept

### Ý tưởng chính:

```
┌─────────────────────────────────────────────────────────────────┐
│                    VISION TRANSFORMER                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  "An image is worth 16x16 words"                               │
│                                                                 │
│  Thay vì convolution, ViT:                                     │
│  1. Chia ảnh thành patches (16×16 hoặc 32×32)                  │
│  2. Flatten mỗi patch thành vector                             │
│  3. Áp dụng Transformer encoder                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Cell 1-4: Setup

```python
# Standard imports + data loading + device config
%run data.ipynb
```

---

## 🔧 Cell 5: MLP Block

```python
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x
```

### GELU Activation:

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}[1 + \text{erf}(x/\sqrt{2})]$$

```
     GELU vs ReLU
     │
  1  │        ___GELU___
     │      /
     │     /    ___ReLU
     │    /   /
  0  ├───/───/──────────
     │  /
     │ /
 -1  │/
     └────────────────────
       -2  -1   0   1   2
```

**Tại sao GELU?**
- Smooth approximation của ReLU
- Non-zero gradient cho negative inputs
- Được dùng trong BERT, GPT, ViT

---

## 🔧 Cell 6: Patches Layer

```python
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
```

### Patch Extraction Visualization:

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE (224 × 224 × 3)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐                  │
│   │  1  │  2  │  3  │  4  │  5  │  6  │  7  │                  │
│   ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤                  │
│   │  8  │  9  │ 10  │ 11  │ 12  │ 13  │ 14  │                  │
│   ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤                  │
│   │ 15  │ 16  │ 17  │ 18  │ 19  │ 20  │ 21  │   patch_size=32  │
│   ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤   → 7×7 = 49     │
│   │ 22  │ 23  │ 24  │ 25  │ 26  │ 27  │ 28  │     patches      │
│   ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤                  │
│   │ 29  │ 30  │ 31  │ 32  │ 33  │ 34  │ 35  │                  │
│   ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤                  │
│   │ 36  │ 37  │ 38  │ 39  │ 40  │ 41  │ 42  │                  │
│   ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤                  │
│   │ 43  │ 44  │ 45  │ 46  │ 47  │ 48  │ 49  │                  │
│   └─────┴─────┴─────┴─────┴─────┴─────┴─────┘                  │
│                                                                 │
│   Mỗi patch: 32 × 32 × 3 = 3,072 dimensions                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Tính toán:

```python
image_size = 224
patch_size = 32
num_patches = (224 // 32) ** 2 = 7 × 7 = 49 patches
patch_dims = 32 × 32 × 3 = 3,072
```

---

## 🔧 Cell 7: Patch Visualization

```python
for image_batch, label_batch in train_generator:
    image = image_batch[0] 
    break 

patches = Patches(patch_size)(tf.expand_dims(image, 0))

# Visualize patches in grid
n = int(np.sqrt(num_patches))  # 7
for i in range(n * n):
    patch_img = patches_numpy[0, i].reshape(patch_size, patch_size, 3)
    plt.subplot(n, n, i + 1)
    plt.imshow(patch_img)
```

### Output:

```
Original Image → 7×7 Grid of Patches
┌─────────────┐    ┌───┬───┬───┬───┬───┬───┬───┐
│             │    │   │   │   │   │   │   │   │
│    Chest    │ →  ├───┼───┼───┼───┼───┼───┼───┤
│    X-ray    │    │   │   │   │   │   │   │   │
│             │    ├───┼───┼───┼───┼───┼───┼───┤
└─────────────┘    │   │   │ ⬛│ ⬛│   │   │   │ ← Lung regions
                   └───┴───┴───┴───┴───┴───┴───┘
```

---

## 🔧 Cell 8: Patch Encoder

```python
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded
```

### Patch Encoding Process:

```
┌─────────────────────────────────────────────────────────────────┐
│                     PATCH ENCODER                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Flattened patches (batch, 49, 3072)                    │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐       │
│  │           Linear Projection (Dense)                  │       │
│  │           3072 → projection_dim (64)                │       │
│  └──────────────────────────┬──────────────────────────┘       │
│                             │                                   │
│                             ▼                                   │
│         Projected patches: (batch, 49, 64)                      │
│                             │                                   │
│                             │     ┌────────────────────┐       │
│                             │     │ Position Embedding │       │
│                             │     │ (49, 64)           │       │
│                             │     └─────────┬──────────┘       │
│                             │               │                   │
│                             └───────┬───────┘                   │
│                                     │                           │
│                                     ▼                           │
│                              ┌─────────┐                        │
│                              │   ADD   │                        │
│                              └────┬────┘                        │
│                                   │                             │
│                                   ▼                             │
│              Output: (batch, 49, 64) with positional info       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Position Embedding:

```
Patch   Position    Embedding Vector (64-dim)
  1        0        [0.12, -0.34, 0.56, ...]
  2        1        [0.23, -0.45, 0.67, ...]
  3        2        [0.34, -0.56, 0.78, ...]
  ...
  49       48       [0.89, -0.12, 0.34, ...]
```

**Tại sao cần Position Embedding?**
- Transformer không có khái niệm về thứ tự
- Cần encode vị trí của mỗi patch trong ảnh

---

## 🔧 Cell 9: Hyperparameters

```python
input_shape = (224, 224, 3)  
patch_size = 32             
num_patches = (224 // 32) ** 2  = 49
projection_dim = 64          
num_heads = 4                
transformer_units = [128, 64]  # projection_dim * 2, projection_dim
transformer_layers = 8       
mlp_head_units = [2048, 1024]  
num_classes = 15
```

### Giải thích:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `patch_size` | 32 | Kích thước mỗi patch |
| `num_patches` | 49 | Số patches = sequence length |
| `projection_dim` | 64 | Embedding dimension |
| `num_heads` | 4 | Multi-head attention heads |
| `transformer_layers` | 8 | Số Transformer blocks |
| `mlp_head_units` | [2048, 1024] | Classification head |

### So sánh với ViT gốc:

| Config | ViT-Base | This Model |
|--------|----------|------------|
| patch_size | 16 | 32 |
| num_patches | 196 | 49 |
| projection_dim | 768 | 64 |
| num_heads | 12 | 4 |
| transformer_layers | 12 | 8 |
| Parameters | 86M | ~3M |

→ Đây là **ViT-Tiny** version!

---

## 🔧 Cell 10: ViT Classifier

```python
def create_vit_classifier():
    inputs = Input(shape=input_shape)
    
    # Patch extraction + encoding
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Transformer blocks
    for _ in range(transformer_layers):
        # Layer Normalization 1
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-Head Self-Attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        
        # Skip connection 1
        x2 = Add()([attention_output, encoded_patches])
        
        # Layer Normalization 2
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        
        # Skip connection 2
        encoded_patches = Add()([x3, x2])

    # Classification head
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = Dense(num_classes, activation='sigmoid')(features)
    
    model = Model(inputs=inputs, outputs=logits)
    return model
```

### Kiến trúc đầy đủ:

```
┌─────────────────────────────────────────────────────────────────┐
│                    VISION TRANSFORMER                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: (224, 224, 3)                                          │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ PATCH EMBEDDING                                          │   │
│  │   Patches(32) → (49, 3072)                              │   │
│  │   PatchEncoder → (49, 64)                               │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│         ┌───────────────────┴───────────────────┐              │
│         │      TRANSFORMER ENCODER × 8          │              │
│         │  ┌─────────────────────────────────┐  │              │
│         │  │ LayerNorm                       │  │              │
│         │  │         │                       │  │              │
│         │  │         ▼                       │  │              │
│         │  │ Multi-Head Self-Attention       │  │              │
│         │  │ (4 heads, key_dim=64)           │  │              │
│         │  │         │                       │  │              │
│         │  │         ▼                       │  │              │
│         │  │    Add (Skip Connection)        │  │              │
│         │  │         │                       │  │              │
│         │  │         ▼                       │  │              │
│         │  │ LayerNorm                       │  │              │
│         │  │         │                       │  │              │
│         │  │         ▼                       │  │              │
│         │  │ MLP [128 → 64]                  │  │              │
│         │  │         │                       │  │              │
│         │  │         ▼                       │  │              │
│         │  │    Add (Skip Connection)        │  │              │
│         │  └─────────────────────────────────┘  │              │
│         └───────────────────┬───────────────────┘              │
│                             │                                   │
│         ┌───────────────────▼───────────────────┐              │
│         │      CLASSIFICATION HEAD               │              │
│         │  LayerNorm → Flatten → Dropout(0.5)   │              │
│         │  MLP [2048 → 1024] → Dense(15)        │              │
│         └───────────────────┬───────────────────┘              │
│                             │                                   │
│                             ▼                                   │
│                    OUTPUT: (15,) probabilities                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Head Self-Attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

```
┌─────────────────────────────────────────────────────────────────┐
│               MULTI-HEAD SELF-ATTENTION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input X: (49, 64)                                             │
│      │                                                          │
│      ├──────────┬──────────┬──────────┐                        │
│      │          │          │          │                        │
│      ▼          ▼          ▼          ▼                        │
│   Head 1     Head 2     Head 3     Head 4                      │
│   (49,16)    (49,16)    (49,16)    (49,16)                    │
│      │          │          │          │                        │
│      └──────────┴──────────┴──────────┘                        │
│                     │                                           │
│                     ▼                                           │
│              Concatenate → (49, 64)                            │
│                     │                                           │
│                     ▼                                           │
│              Linear projection                                  │
│                     │                                           │
│                     ▼                                           │
│              Output: (49, 64)                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Cell 11-14: Training & Visualization

```python
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-6
num_epochs = 10

vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)
plot_combined_history(history)
```

---

## 🔧 Cell 15: Evaluation với ROC Curves

```python
# Predict on test set
predictions = vit_classifier.predict(test_generator)

# Evaluate
loss, test_accuracy, test_auc = vit_classifier.evaluate(test_generator)

# Plot ROC curves for each class
def plot_roc_curves(y_true, y_pred, num_classes, class_labels):
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # ... plot
```

### ROC Curve Interpretation:

```
       True Positive Rate (Sensitivity)
  1.0  │         ___________
       │        /           
       │       /  AUC = 0.85 (Good)
       │      /              
  0.5  │     /    
       │    /     
       │   /      AUC = 0.5 (Random)
       │  /      /
  0.0  │─/──────/─────────────
       0.0     0.5         1.0
            False Positive Rate
```

---

## 📊 Đánh Giá Tổng Thể

### ✅ Điểm mạnh:
1. ✅ Triển khai ViT đầy đủ từ đầu
2. ✅ Patches extraction đúng chuẩn
3. ✅ Position embedding
4. ✅ Multi-head attention
5. ✅ Pre-norm architecture (LayerNorm trước attention)
6. ✅ ROC curve evaluation

### ❌ Điểm yếu:

| Vấn đề | Mức độ | Giải pháp |
|--------|--------|-----------|
| Thiếu CLS token | 🟠 High | Add learnable CLS token |
| Model quá nhỏ | 🟠 High | Increase dimensions |
| Dùng Flatten thay vì CLS | 🟡 Medium | Use CLS for classification |
| Không có pretrained weights | 🟠 High | Use ViT pretrained on ImageNet |
| Missing `ops` import | 🔴 Critical | Use `tf` hoặc `keras.ops` |

### ⚠️ Bug: `ops` not defined

```python
positions = ops.expand_dims(...)  # ❌ ops không được import
```

**Fix:**
```python
from tensorflow import keras
positions = keras.ops.expand_dims(...)  # Keras 3
# hoặc
positions = tf.expand_dims(...)  # TensorFlow
```

---

## 💡 Improved ViT với CLS Token:

```python
class ViTWithCLS(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.cls_token = self.add_weight(
            shape=(1, 1, projection_dim),
            initializer='zeros',
            trainable=True,
            name='cls_token'
        )
        
    def call(self, patches):
        batch_size = tf.shape(patches)[0]
        cls_tokens = tf.broadcast_to(
            self.cls_token, 
            [batch_size, 1, self.projection_dim]
        )
        return tf.concat([cls_tokens, patches], axis=1)
```

---

## 📚 ViT vs CNN Comparison

| Aspect | CNN | ViT |
|--------|-----|-----|
| **Inductive bias** | Strong (locality, translation equivariance) | Weak |
| **Data efficiency** | Better with small data | Needs large data |
| **Scalability** | Limited | Excellent |
| **Interpretability** | Feature maps | Attention maps |
| **Training cost** | Lower | Higher |

---

## 📚 References

1. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021
2. Vaswani et al., "Attention is All You Need", NeurIPS 2017
3. Original ViT implementation: https://github.com/google-research/vision_transformer
