# Phân Tích Chi Tiết: Vision Transformer V2 (Improved)

## 📋 Tổng Quan

File `ViT-v2.ipynb` là **phiên bản cải tiến** của ViT-v1 với các kỹ thuật regularization và training nâng cao hơn, bao gồm:
- L2 Regularization
- Early Stopping
- Learning Rate Scheduling
- Multiple Optimizer Options

---

## 🔄 So sánh V1 vs V2

| Feature | ViT-v1 | ViT-v2 |
|---------|--------|--------|
| L2 Regularization | ❌ | ✅ |
| Early Stopping | ❌ | ✅ |
| LR Scheduler | ❌ | ✅ |
| Multiple Optimizers | ❌ | ✅ |
| Dropout in MLP | 0.1 | 0.1 + L2 |
| Weight Decay | 1e-6 | 1e-5 |

---

## 🔧 Cell 5: Improved MLP with L2 Regularization

```python
def mlp(x, hidden_units, dropout_rate, regularizer_rate=0.01):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu, 
                  kernel_regularizer=l2(regularizer_rate))(x)
        x = Dropout(dropout_rate)(x)
    return x
```

### L2 Regularization:

$$\mathcal{L}_{total} = \mathcal{L}_{BCE} + \lambda \sum_{i} w_i^2$$

```
┌─────────────────────────────────────────────────────────────────┐
│                    L2 REGULARIZATION EFFECT                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Không có L2:                  Có L2 (λ=0.01):                 │
│                                                                 │
│  ┌─────────────────┐          ┌─────────────────┐              │
│  │ Large weights   │          │ Small weights   │              │
│  │ w = [5, -8, 12] │    →     │ w = [0.5, -0.8] │              │
│  │ Overfitting!    │          │ Generalize!     │              │
│  └─────────────────┘          └─────────────────┘              │
│                                                                 │
│  Effect: Penalize large weights → simpler model                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ⚠️ Vấn đề:
- `l2` chưa được import!

**Fix:**
```python
from tensorflow.keras.regularizers import l2
```

---

## 🔧 Cell 9: Hyperparameters

```python
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-5  # Tăng từ 1e-6
num_epochs = 10
```

| Parameter | V1 | V2 | Lý do |
|-----------|-----|-----|-------|
| `weight_decay` | 1e-6 | 1e-5 | Stronger regularization |

---

## 🔧 Cell 10: Advanced Training Configuration

### Early Stopping:

```python
early_stopping_callback = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=False,  # ⚠️ Nên là True!
)
```

### Early Stopping Visualization:

```
     Validation Accuracy
  │
  │                    ___
  │              _____|   |
  │         ____|         |
  │    ____|              |
  │___|                   |
  │                       └─ Stop here! (patience=3)
  └───────────────────────────────────
    1  2  3  4  5  6  7  8  9  10  Epochs
                        │
                        └─ Best model at epoch 7
                           but returns epoch 10 weights!
```

### ⚠️ Bug: `restore_best_weights=False`

**Vấn đề**: Không restore weights tốt nhất sau khi stop!

**Fix:**
```python
early_stopping_callback = EarlyStopping(
    monitor="val_loss",  # Thay vì val_accuracy
    patience=5,          # Tăng patience
    restore_best_weights=True,  # ✅ Quan trọng!
)
```

---

## 🔧 Multiple Optimizers:

```python
optimizers = {
    "adam": keras.optimizers.Adam(learning_rate=learning_rate),
    "adamw": keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
    "sgd": keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, weight_decay=weight_decay),
    "sgd_momentum": keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
    "adagrad": keras.optimizers.Adagrad(learning_rate=learning_rate),
    "rmsprop": keras.optimizers.RMSprop(learning_rate=learning_rate)
}

optimizer = optimizers["sgd"]  # Chọn SGD
```

### So sánh Optimizers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZER COMPARISON                         │
├─────────────────┬───────────────────────────────────────────────┤
│   Optimizer     │   Characteristics                             │
├─────────────────┼───────────────────────────────────────────────┤
│   Adam          │ Adaptive LR, fast convergence                 │
│                 │ Good default choice                           │
├─────────────────┼───────────────────────────────────────────────┤
│   AdamW         │ Adam + decoupled weight decay                 │
│                 │ Better for Transformers                       │
├─────────────────┼───────────────────────────────────────────────┤
│   SGD           │ Simple, good generalization                   │
│                 │ Slower convergence                            │
├─────────────────┼───────────────────────────────────────────────┤
│   SGD+Momentum  │ Accelerated SGD                               │
│                 │ Reduces oscillation                           │
├─────────────────┼───────────────────────────────────────────────┤
│   SGD+Nesterov  │ Look-ahead momentum                           │
│                 │ Better for convex optimization                │
├─────────────────┼───────────────────────────────────────────────┤
│   RMSprop       │ Adaptive, good for RNNs                       │
│                 │ Less used for vision                          │
└─────────────────┴───────────────────────────────────────────────┘
```

### ⚠️ Nhận xét:
- **Chọn SGD** thay vì AdamW là **không tối ưu** cho ViT
- ViT papers thường dùng **AdamW** với warmup

---

## 🔧 Learning Rate Scheduler:

```python
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
```

### LR Scheduling Visualization:

```
  Learning Rate
  │
  1e-4 ──────────────────┐
       │                 │
       │                 │ patience=5
       │                 │
  1e-5 │                 └─────────────────┐
       │                                   │
       │                                   │
  1e-6 │                                   └────── min_lr
       │
       └──────────────────────────────────────────
         0     5    10    15    20    25    Epochs
                    │           │
              val_loss     val_loss
              stagnates    stagnates
              (reduce!)    (reduce!)
```

### Parameters:

| Param | Value | Meaning |
|-------|-------|---------|
| `monitor` | 'val_loss' | Metric to track |
| `factor` | 0.1 | LR *= 0.1 when triggered |
| `patience` | 5 | Epochs to wait before reducing |
| `min_lr` | 1e-6 | Minimum learning rate |

---

## 🔧 Cell 11: ViT Classifier with Regularization

```python
def create_vit_classifier():
    # ... (same as V1)
    for _ in range(transformer_layers):
        # ...
        x3 = mlp(x3, transformer_units, dropout_rate=0.1, regularizer_rate=0.01)
        # ...
    
    features = mlp(representation, mlp_head_units, dropout_rate=0.5, regularizer_rate=0.01)
    # ...
```

### Regularization Points:

```
┌─────────────────────────────────────────────────────────────────┐
│                  REGULARIZATION IN VIT-V2                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TRANSFORMER BLOCK:                                             │
│  ┌──────────────────────────────────────┐                      │
│  │  MultiHeadAttention(dropout=0.1)     │ ← Dropout            │
│  │           │                          │                      │
│  │  MLP (L2=0.01, dropout=0.1)          │ ← L2 + Dropout       │
│  └──────────────────────────────────────┘                      │
│                                                                 │
│  CLASSIFICATION HEAD:                                           │
│  ┌──────────────────────────────────────┐                      │
│  │  Flatten                             │                      │
│  │  Dropout(0.5)                        │ ← Strong Dropout     │
│  │  MLP (L2=0.01, dropout=0.5)          │ ← L2 + Dropout       │
│  │  Dense (sigmoid)                     │                      │
│  └──────────────────────────────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Cell 12: Training Function

```python
def run_experiment(model):
    model.compile(
        optimizer=optimizer,  # SGD
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
        callbacks=[
            ModelCheckpoint(...),
            early_stopping_callback,
            lr_scheduler
        ],
    )
    return history
```

### Callback Execution Order:

```
Each Epoch:
  1. Train on batches
  2. Validate
  3. ModelCheckpoint: Save if val_loss improved
  4. ReduceLROnPlateau: Check if should reduce LR
  5. EarlyStopping: Check if should stop
```

---

## 🔧 Cell 13-17: Evaluation

```python
# Training
vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)

# Visualization
plot_combined_history(history)

# Test evaluation
predictions = vit_classifier.predict(test_generator)
loss, test_accuracy, test_auc = vit_classifier.evaluate(test_generator)

# ROC Curves
plot_roc_curves(y_true, predictions, num_classes, class_labels)
```

---

## 📊 So sánh Training Behavior

### V1 (No regularization):

```
Epoch   Train Loss   Val Loss   Status
─────────────────────────────────────────
  1       0.50        0.48      OK
  2       0.35        0.45      OK
  3       0.20        0.50      Overfitting starts
  4       0.10        0.60      Overfitting
  5       0.05        0.75      Severe overfitting
```

### V2 (With regularization):

```
Epoch   Train Loss   Val Loss   LR       Status
──────────────────────────────────────────────────
  1       0.50        0.48     1e-4      OK
  2       0.40        0.44     1e-4      OK
  3       0.35        0.42     1e-4      OK
  4       0.32        0.41     1e-4      OK
  5       0.30        0.40     1e-4      OK
  6       0.28        0.40     1e-5      LR reduced
  7       0.27        0.39     1e-5      OK
  8       0.26        0.39     1e-5      OK
  9       0.26        0.39     1e-5      Early stop
```

---

## 📊 Đánh Giá Tổng Thể

### ✅ Điểm mạnh so với V1:
1. ✅ L2 Regularization chống overfitting
2. ✅ Early Stopping tiết kiệm thời gian
3. ✅ LR Scheduler adaptive learning
4. ✅ Multiple optimizer options
5. ✅ ROC curve visualization

### ❌ Điểm yếu:

| Vấn đề | Mức độ | Giải pháp |
|--------|--------|-----------|
| `l2` not imported | 🔴 Critical | Add import |
| `restore_best_weights=False` | 🔴 Critical | Set to True |
| SGD thay vì AdamW | 🟠 High | Use AdamW for ViT |
| `ops` not defined | 🔴 Critical | Use tf/keras.ops |
| Missing warmup | 🟡 Medium | Add LR warmup |

---

## 💡 Optimal Training Configuration:

```python
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Warmup + Cosine decay
def lr_schedule(epoch, lr):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    return lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True
    ),
    keras.callbacks.LearningRateScheduler(lr_schedule)
]

# Use AdamW for ViT
optimizer = keras.optimizers.AdamW(
    learning_rate=1e-4,
    weight_decay=0.01,
    beta_1=0.9,
    beta_2=0.999
)
```

---

## 📊 Expected Performance Improvement

| Metric | V1 | V2 (expected) |
|--------|-----|---------------|
| Train Accuracy | 95% | 88% (less overfit) |
| Val Accuracy | 75% | 82% |
| Test AUC | 0.65 | 0.72 |
| Generalization Gap | 20% | 6% |

---

## 📚 Best Practices for ViT Training

1. **Optimizer**: AdamW với weight decay 0.01-0.1
2. **LR Schedule**: Warmup + Cosine decay
3. **Regularization**: Dropout + Label smoothing + MixUp
4. **Data Augmentation**: RandAugment, AutoAugment
5. **Pretrained**: Khởi tạo từ ImageNet-21k weights

---

## 📚 References

1. "Training data-efficient image transformers" (DeiT), Touvron et al., 2021
2. "How to train your ViT?", Steiner et al., 2021
3. "An Image is Worth 16x16 Words", Dosovitskiy et al., 2020
