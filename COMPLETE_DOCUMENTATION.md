# 🎓 TÀI LIỆU HOÀN CHỈNH: ViT-Chest-Xray Deep Learning Project

## Tác giả: AI Expert Analysis System
## Dự án: NIH Chest X-ray14 Classification using Vision Transformer
## Cập nhật: January 2025

---

# 📑 MỤC LỤC TỔNG HỢP

## PHẦN A: TỔNG QUAN DỰ ÁN
- [A1. Dataset NIH Chest X-ray14](#a1-dataset-nih-chest-x-ray14)
- [A2. Kiến trúc Models](#a2-kiến-trúc-models)
- [A3. So sánh Performance](#a3-so-sánh-performance)
- [A4. Bugs & Recommendations](#a4-bugs--recommendations)

## PHẦN B: LÝ THUYẾT DEEP LEARNING
- [B1. Neural Network Cơ bản](#b1-neural-network-cơ-bản)
- [B2. Convolution Chi tiết](#b2-convolution-chi-tiết)
- [B3. Pooling Layers](#b3-pooling-layers)
- [B4. Activation Functions](#b4-activation-functions)
- [B5. Loss Functions](#b5-loss-functions)
- [B6. Optimizer Chi tiết](#b6-optimizer-chi-tiết)
- [B7. Transformer & Attention](#b7-transformer--attention)
- [B8. Transfer Learning](#b8-transfer-learning)
- [B9. Evaluation Metrics](#b9-evaluation-metrics)

## PHẦN C: GIẢI THÍCH CODE
- [C1. Data Pipeline](#c1-data-pipeline)
- [C2. CNN Model](#c2-cnn-model)
- [C3. ResNet Model](#c3-resnet-model)
- [C4. ViT Models](#c4-vit-models)

---

# ═══════════════════════════════════════════════════════════════
# PHẦN A: TỔNG QUAN DỰ ÁN
# ═══════════════════════════════════════════════════════════════

# A1. Dataset NIH Chest X-ray14

## Thông tin cơ bản

| Thuộc tính | Giá trị |
|------------|---------|
| **Tên** | ChestX-ray14 (NIH Clinical Center) |
| **Năm phát hành** | 2017 |
| **Tổng số ảnh** | 112,120 |
| **Số bệnh nhân** | 30,805 |
| **Số bệnh lý** | 14 + "No Finding" = 15 classes |
| **Dung lượng** | ~42 GB |
| **Format** | PNG (grayscale → RGB) |
| **Resolution gốc** | ~2000×2000 pixels |

## 14 Bệnh lý (Pathologies)

```
┌─────────────────────────────────────────────────────────────────┐
│                  14 PATHOLOGICAL CONDITIONS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CARDIAC:                    PULMONARY:                        │
│  └── Cardiomegaly (Tim to)   ├── Emphysema (Khí phế thũng)    │
│                              ├── Pneumothorax (Tràn khí)       │
│  PLEURAL:                    ├── Pneumonia (Viêm phổi)         │
│  ├── Effusion (Tràn dịch)    ├── Consolidation (Đông đặc)     │
│  └── Pleural Thickening      ├── Infiltration (Thâm nhiễm)    │
│                              └── Atelectasis (Xẹp phổi)        │
│  MASSES:                                                        │
│  ├── Mass (Khối u lớn)       OTHERS:                           │
│  └── Nodule (Nốt nhỏ)        ├── Fibrosis (Xơ hóa)            │
│                              ├── Edema (Phù phổi)              │
│  DIAPHRAGM:                  └── No Finding (Bình thường)      │
│  └── Hernia (Thoát vị)                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Đặc điểm Multi-label

```python
# Một ảnh có thể có nhiều bệnh cùng lúc
"Cardiomegaly|Emphysema"     # 2 bệnh
"Hernia|Infiltration"        # 2 bệnh

# Phân bố: 1 label (~75%), 2 labels (~20%), 3+ labels (~5%)
```

## Data Quality Issues

| Vấn đề | Mô tả | Giải pháp |
|--------|-------|-----------|
| Label Noise | NLP extracted, ~90% accuracy | Focal Loss, Label smoothing |
| Class Imbalance | Ratio 300:1 (No Finding vs Hernia) | Weighted sampling, Focal Loss |
| Patient Overlap | Same patient in train/test | Split by Patient ID |
| View Position Bias | PA vs AP quality differs | Stratified sampling |

---

# A2. Kiến trúc Models

## So sánh tổng quan

| Model | Params | Framework | Pretrained | Expected AUC |
|-------|--------|-----------|------------|--------------|
| CNN | 95.6M | TensorFlow | ❌ | 0.55-0.65 |
| ResNet-34 | 21.3M | TensorFlow | ❌ | 0.70-0.78 |
| ViT-v1 | ~3M | TensorFlow | ❌ | 0.60-0.68 |
| ViT-v2 | ~3M | TensorFlow | ❌ | 0.68-0.75 |
| **ViT-Pretrained** | **86M** | **PyTorch** | **✅** | **0.82-0.88** |

## CNN Architecture

```
Input (224, 224, 3)
       │
Conv2D(32, 3×3, relu)     →  (222, 222, 32)    params: 896
MaxPool2D(2×2)            →  (111, 111, 32)
Conv2D(64, 3×3, relu)     →  (109, 109, 64)    params: 18,496
MaxPool2D(2×2)            →  (54, 54, 64)
Flatten()                 →  (186,624)
Dense(512, relu)          →  (512)              params: 95,552,000 ← 99%!
Dense(15, sigmoid)        →  (15)

TOTAL: ~95.6M parameters
🔴 ISSUE: 99% params in Dense layer → severe overfitting
```

## ResNet-34 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  STEM: Conv 7×7, 64, stride 2 → BN → ReLU → MaxPool            │
│  STAGE 1: 3 × ResBlock(64)   → (56, 56, 64)                    │
│  STAGE 2: 4 × ResBlock(128)  → (28, 28, 128)                   │
│  STAGE 3: 6 × ResBlock(256)  → (14, 14, 256)                   │
│  STAGE 4: 3 × ResBlock(512)  → (7, 7, 512)                     │
│  GlobalAveragePooling → Dense(15, sigmoid)                     │
│  TOTAL: ~21.3M parameters                                      │
└─────────────────────────────────────────────────────────────────┘

Skip Connection: y = F(x) + x
→ Gradient always has identity term → no vanishing!
```

## Vision Transformer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  1. PATCH EMBEDDING:                                           │
│     - Split into patches (16×16 or 32×32)                      │
│     - Linear projection to embedding dim                       │
│     - Add position embeddings                                   │
│                                                                 │
│  2. TRANSFORMER ENCODER × N:                                   │
│     ┌─────────────────────────────────┐                        │
│     │  LayerNorm → Multi-Head Attention → + Skip               │
│     │  LayerNorm → MLP (FFN) → + Skip                          │
│     └─────────────────────────────────┘                        │
│                                                                 │
│  3. CLASSIFICATION: [CLS] token → MLP Head → 15 classes        │
└─────────────────────────────────────────────────────────────────┘

ViT-v1: 49 patches (32×32), 8 layers, 4 heads, ~3M params
ViT-Pretrained: 196 patches (16×16), 12 layers, 12 heads, ~86M params
```

---

# A3. So sánh Performance

## Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│  RECEPTIVE FIELD:                                               │
│  ├── CNN:    Local (3×3) → grows slowly with depth             │
│  ├── ResNet: Local → larger due to depth                       │
│  └── ViT:    GLOBAL from layer 1! (attention to all patches)   │
│                                                                 │
│  DATA EFFICIENCY:                                               │
│  ├── CNN/ResNet: High (strong inductive bias)                  │
│  └── ViT:        Low (needs lots of data or pretraining)       │
│                                                                 │
│  SCALABILITY:                                                   │
│  ├── CNN:    Limited (stacking convs)                          │
│  └── ViT:    Excellent (just add more layers/heads)            │
└─────────────────────────────────────────────────────────────────┘
```

## Expected Results

| Model | Train Acc | Val Acc | AUC | Notes |
|-------|-----------|---------|-----|-------|
| CNN | ~95% | ~70% | 0.55-0.65 | Severe overfitting |
| ResNet | ~88% | ~78% | 0.70-0.78 | Good balance |
| ViT-v1 | ~92% | ~72% | 0.60-0.68 | Too small |
| ViT-v2 | ~85% | ~78% | 0.68-0.75 | Better regularization |
| **ViT-Pretrained** | **~90%** | **~86%** | **0.82-0.88** | **BEST** |

---

# A4. Bugs & Recommendations

## Critical Bugs

| File | Bug | Fix |
|------|-----|-----|
| `data.ipynb` | `idxs = random` (wrong) | `idxs = random.sample(idxs, num_images)` |
| `data_download.ipynb` | Wrong dest_path | `os.path.join(sub_dir, file)` |
| `ViT-v1.ipynb` | `ops` not defined | Use `tf.expand_dims` |
| `ViT-v2.ipynb` | `l2` not imported | `from keras.regularizers import l2` |
| `ViT-v2.ipynb` | `restore_best_weights=False` | Set to `True` |

## Recommendations

### Data Pipeline
```python
# Fix: Split by Patient ID to avoid data leakage
patient_ids = df['Patient ID'].unique()
train_patients, test_patients = train_test_split(patient_ids, test_size=0.2)
train_df = df[df['Patient ID'].isin(train_patients)]
test_df = df[df['Patient ID'].isin(test_patients)]
```

### Model Selection
```python
# Recommended: Use pretrained ViT
import timm
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=15)
```

### Loss Function for Imbalance
```python
# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
```

---

# ═══════════════════════════════════════════════════════════════
# PHẦN B: LÝ THUYẾT DEEP LEARNING
# ═══════════════════════════════════════════════════════════════

# B1. Neural Network Cơ bản

## Neuron Sinh học vs Nhân tạo

```
NEURON SINH HỌC:
┌─────────────────────────────────────────────────────────────────┐
│  Dendrites (nhận tín hiệu) → Cell Body (tổng hợp) →            │
│  Axon (truyền) → Synapses (kết nối neuron khác)                │
│                                                                 │
│  Nếu tổng tín hiệu > ngưỡng → phát xung điện                   │
└─────────────────────────────────────────────────────────────────┘

NEURON NHÂN TẠO:
┌─────────────────────────────────────────────────────────────────┐
│  x₁ ──w₁──┐                                                     │
│  x₂ ──w₂──┼──→ Σ(wᵢxᵢ) + b ──→ f(z) ──→ output               │
│  x₃ ──w₃──┘                                                     │
│                                                                 │
│  z = w₁x₁ + w₂x₂ + w₃x₃ + b                                    │
│  a = f(z)  ← activation function                               │
└─────────────────────────────────────────────────────────────────┘
```

## Weights và Bias - Ví dụ Spam Classification

```
INPUT FEATURES (email):
x₁ = Có từ "FREE"?        (1 = có, 0 = không)
x₂ = Có từ "WINNER"?
x₃ = Có tên người nhận?

SAU KHI TRAIN:
w₁ = +2.5   ← "FREE" → spam
w₂ = +1.8   ← "WINNER" → spam
w₃ = -1.5   ← Có tên → NOT spam

EMAIL: "FREE WINNER offer" (không có tên)
x = [1, 1, 0]
z = 2.5×1 + 1.8×1 + (-1.5)×0 + 0.5 = 4.8
ŷ = sigmoid(4.8) = 0.992 → 99.2% SPAM!
```

## Weight Initialization - Xavier

### Vấn đề

```
❌ All zeros: Symmetry problem - all neurons learn same thing
❌ Too large: Exploding activations → vanishing gradients
❌ Too small: Vanishing activations → no learning
```

### Chứng minh Xavier Initialization

```
GIẢ THIẾT:
- Input x ~ N(0, 1)
- Weights w ~ N(0, σ²)

MỤC TIÊU: Var[z] = Var[x] = 1

TÍNH TOÁN:
z = Σᵢ wᵢxᵢ với n inputs

Var[z] = Σᵢ Var[wᵢxᵢ]
       = n × E[wᵢ²] × E[xᵢ²]
       = n × σ² × 1
       = n × σ²

ĐỂ Var[z] = 1:
n × σ² = 1
σ = 1/√n  ← XAVIER INITIALIZATION!

CODE:
W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
```

### He Initialization (cho ReLU)

```
ReLU "giết" 50% activations (z < 0 → 0)
→ Variance giảm 50%
→ Cần σ = √(2/n) để compensate

CODE:
W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
```

---

# B2. Convolution Chi tiết

## Ví dụ tính tay: 6×6 input, 3×3 kernel

```
INPUT (6×6):                    KERNEL (3×3) - Edge Detector:
┌────┬────┬────┬────┬────┬────┐ ┌────┬────┬────┐
│  1 │  2 │  3 │  0 │  1 │  2 │ │ -1 │ -1 │ -1 │
├────┼────┼────┼────┼────┼────┤ ├────┼────┼────┤
│  0 │  1 │  2 │  3 │  0 │  1 │ │ -1 │  8 │ -1 │
├────┼────┼────┼────┼────┼────┤ ├────┼────┼────┤
│  1 │  2 │  1 │  0 │  1 │  2 │ │ -1 │ -1 │ -1 │
├────┼────┼────┼────┼────┼────┤ └────┴────┴────┘
│  2 │  1 │  0 │  1 │  2 │  0 │
├────┼────┼────┼────┼────┼────┤
│  1 │  0 │  2 │  1 │  1 │  3 │
├────┼────┼────┼────┼────┼────┤
│  0 │  1 │  1 │  2 │  0 │  1 │
└────┴────┴────┴────┴────┴────┘
```

### Step 1: Position (0,0)

```
Vùng 3×3 tại (0,0):    × Kernel:         = Element-wise:
┌────┬────┬────┐       ┌────┬────┬────┐   ┌─────┬─────┬─────┐
│  1 │  2 │  3 │   ×   │ -1 │ -1 │ -1 │ = │  -1 │  -2 │  -3 │
├────┼────┼────┤       ├────┼────┼────┤   ├─────┼─────┼─────┤
│  0 │  1 │  2 │   ×   │ -1 │  8 │ -1 │ = │   0 │   8 │  -2 │
├────┼────┼────┤       ├────┼────┼────┤   ├─────┼─────┼─────┤
│  1 │  2 │  1 │   ×   │ -1 │ -1 │ -1 │ = │  -1 │  -2 │  -1 │
└────┴────┴────┘       └────┴────┴────┘   └─────┴─────┴─────┘

SUM = -1 -2 -3 + 0 +8 -2 -1 -2 -1 = -4

Output[0,0] = -4
```

### Output Size Formula

$$\text{Output} = \frac{N - K + 2P}{S} + 1$$

- N = Input size, K = Kernel size, P = Padding, S = Stride

```
Ví dụ: 6×6 input, 3×3 kernel, no padding, stride 1
Output = (6 - 3 + 0)/1 + 1 = 4×4
```

## Padding và Stride

```
PADDING='same' (giữ kích thước):
Input 6×6 + pad 1 = 8×8
Conv 3×3 → Output 6×6 ✓

STRIDE=2 (giảm kích thước):
Kernel nhảy 2 pixel mỗi bước
→ Output size giảm ~50%
```

## Multiple Channels và Filters

```
INPUT RGB (6×6×3):
- 3 channels (R, G, B)

KERNEL (3×3×3):
- Một slice cho mỗi channel
- Total params: 3×3×3 + 1 = 28

32 FILTERS:
- 32 kernels khác nhau
- Output: 4×4×32 feature maps
- Total params: 32 × 28 = 896
```

## Receptive Field

```
CÔNG THỨC:
RF_n = RF_{n-1} + (K_n - 1) × Π S_i

VÍ DỤ: 3 Conv layers 3×3, stride 1:
RF₁ = 3
RF₂ = 3 + (3-1) × 1 = 5
RF₃ = 5 + (3-1) × 1 = 7

ViT: Mỗi patch "nhìn" 16×16 hoặc 32×32 pixels ngay từ đầu!
+ Attention → Global RF ngay lập tức!
```

---

# B3. Pooling Layers

## Max Pooling Step-by-Step

```
INPUT (4×4):                OUTPUT (2×2) với MaxPool 2×2, stride 2:
┌────┬────┬────┬────┐       ┌────┬────┐
│  1 │  3 │  2 │  1 │       │  9 │  4 │
├────┼────┼────┼────┤  →    ├────┼────┤
│  2 │  9 │  1 │  4 │       │  4 │  8 │
├────┼────┼────┼────┤       └────┴────┘
│  3 │  2 │  8 │  3 │
├────┼────┼────┼────┤       Vùng [1,3,2,9] → max = 9
│  1 │  4 │  2 │  6 │       Vùng [2,1,1,4] → max = 4
└────┴────┴────┴────┘       Vùng [3,2,1,4] → max = 4
                            Vùng [8,3,2,6] → max = 8
```

### Tại sao Max Pooling?

1. **Translation Invariance**: Feature dịch trong vùng 2×2, max vẫn giữ
2. **Dimensionality Reduction**: 4×4 → 2×2 (giảm 4 lần)
3. **Giữ feature mạnh nhất**: Loại bỏ noise

## Global Average Pooling vs Flatten

```
FLATTEN:
Feature map 7×7×512 → Flatten → 25,088 neurons
Dense(256) → 25,088 × 256 = 6,422,528 params! 😱

GLOBAL AVERAGE POOLING:
Feature map 7×7×512 → GAP → 512 neurons
Dense(256) → 512 × 256 = 131,072 params ✓

→ Giảm 50 lần số parameters!
```

---

# B4. Activation Functions

## Sigmoid

```
σ(z) = 1/(1 + e^(-z))

RANGE: (0, 1) - Tốt cho probability output

ĐẠO HÀM:
σ'(z) = σ(z) × (1 - σ(z))

CHỨNG MINH:
σ(z) = (1 + e^(-z))^(-1)
σ'(z) = -1 × (1 + e^(-z))^(-2) × (-e^(-z))
      = e^(-z) / (1 + e^(-z))²
      = [1/(1+e^(-z))] × [e^(-z)/(1+e^(-z))]
      = σ(z) × (1 - σ(z)) ✓
```

### Vanishing Gradient Problem

```
σ'(z) = σ(z) × (1 - σ(z))

Maximum tại z=0: σ'(0) = 0.5 × 0.5 = 0.25

QUA 10 LAYERS:
∂L/∂w₁ = ∂L/∂a₁₀ × σ'(z₁₀) × ... × σ'(z₁)
       ≤ C × 0.25¹⁰
       = C × 0.000001

→ Gradient gần như = 0!
→ Layers đầu không học được!
```

## ReLU

```
ReLU(z) = max(0, z)

ĐẠO HÀM:
ReLU'(z) = 1 nếu z > 0
         = 0 nếu z ≤ 0

ƯU ĐIỂM:
- Không vanishing gradient (z > 0)
- Tính toán nhanh
- Sparse activation (z < 0 → 0)

NHƯỢC ĐIỂM - Dead Neurons:
Nếu z luôn < 0 → gradient = 0 → neuron "chết"
```

## Softmax vs Sigmoid

```
SOFTMAX (Multi-class, mutually exclusive):
softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)
→ Tổng = 1, chỉ 1 class đúng
→ Output: Dog=0.7, Cat=0.2, Bird=0.1

SIGMOID (Multi-label, independent):
σ(zᵢ) = 1/(1 + e^(-zᵢ)) cho mỗi class
→ Mỗi class independent
→ Output: Cardiomegaly=0.8, Effusion=0.9, Pneumonia=0.3
→ Có thể nhiều bệnh cùng lúc!

CHEST X-RAY → SIGMOID (multi-label)
```

---

# B5. Loss Functions

## Binary Cross-Entropy (BCE)

$$L = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

### Ví dụ tính toán

```
Case 1: y=1 (có bệnh), ŷ=0.9
L = -[1×log(0.9) + 0×log(0.1)] = -log(0.9) = 0.105 ← Nhỏ, tốt!

Case 2: y=1 (có bệnh), ŷ=0.1  
L = -[1×log(0.1) + 0×log(0.9)] = -log(0.1) = 2.303 ← Lớn, tệ!

Case 3: y=0 (không bệnh), ŷ=0.1
L = -[0×log(0.1) + 1×log(0.9)] = -log(0.9) = 0.105 ← Nhỏ, tốt!
```

### BCE từ Maximum Likelihood

```
CHỨNG MINH:
P(y|x) = ŷ^y × (1-ŷ)^(1-y)  (Bernoulli distribution)

Maximum Likelihood:
max P(Y|X) = max Π P(yᵢ|xᵢ)

Log-likelihood:
log P(Y|X) = Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

Negative log-likelihood (minimize):
L = -Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

→ BCE = Negative Log-Likelihood of Bernoulli!
```

### Gradient của BCE + Sigmoid

```
TÍNH ∂L/∂z:

L = -[y log(σ) + (1-y) log(1-σ)]

∂L/∂σ = -y/σ + (1-y)/(1-σ)

∂σ/∂z = σ(1-σ)

∂L/∂z = ∂L/∂σ × ∂σ/∂z
      = [-y/σ + (1-y)/(1-σ)] × σ(1-σ)
      = -y(1-σ) + (1-y)σ
      = -y + yσ + σ - yσ
      = σ - y
      = ŷ - y ← ĐẸP!

→ Gradient đơn giản: prediction - target
```

## Focal Loss (cho Class Imbalance)

```
BCE: L = -log(pₜ)
Focal: L = -αₜ(1-pₜ)^γ log(pₜ)

γ = 2 (focusing parameter)
α = 0.25 (class weight)

VÍ DỤ:
Easy example: pₜ = 0.9
(1 - 0.9)² = 0.01 → weight giảm 100 lần!

Hard example: pₜ = 0.1
(1 - 0.1)² = 0.81 → weight gần như giữ nguyên

→ Focus vào hard examples!
→ Down-weight easy examples!
```

---

# B6. Optimizer Chi tiết

## SGD và vấn đề Oscillation

```
w_{t+1} = w_t - α∇L(w_t)

VẤN ĐỀ: Narrow Valleys
- Loss dốc theo w₂ (gradient lớn) → nhảy lớn → vượt quá → oscillate!
- Loss thoải theo w₁ (gradient nhỏ) → nhảy nhỏ → tiến chậm
```

## Momentum - "Quả bóng lăn"

```
v_{t+1} = βv_t + ∇L(w_t)
w_{t+1} = w_t - αv_{t+1}

INTUITION:
Quả bóng lăn xuống đồi:
1. Tích lũy vận tốc (momentum)
2. Quán tính giúp vượt qua local minimum
3. Oscillation bị triệt tiêu (gradient đổi dấu → v trung bình ≈ 0)

VÍ DỤ (β=0.9, α=0.1):
Step 1: gradient = [2, 1]
v = 0.9×[0,0] + [2,1] = [2, 1]

Step 2: gradient = [2, -1]  ← y oscillate!
v = 0.9×[2,1] + [2,-1] = [3.8, -0.1]

Hướng x (consistent): tích lũy!
Hướng y (oscillate): triệt tiêu!
```

## Adam - Adaptive Moment Estimation

```python
# ADAM = MOMENTUM + RMSPROP + BIAS CORRECTION

m = β₁×m + (1-β₁)×g      # First moment (mean)
v = β₂×v + (1-β₂)×g²     # Second moment (variance)
m̂ = m / (1 - β₁^t)       # Bias correction
v̂ = v / (1 - β₂^t)       # Bias correction
w = w - α × m̂ / (√v̂ + ε)

Default: β₁=0.9, β₂=0.999, α=0.001

TẠI SAO HIỆU QUẢ:
1. Adaptive LR cho từng parameter
2. Momentum giúp vượt local minima
3. Works out of the box
```

### Bias Correction

```
VẤN ĐỀ:
Ban đầu m = 0
m₁ = β₁×0 + (1-β₁)×g₁ = 0.1×g₁ ← BIASED!

GIẢI PHÁP:
E[m_t] = (1 - β₁^t) × E[g]
m̂_t = m_t / (1 - β₁^t) ← UNBIASED!

t=1: correction = 1/0.1 = 10× (lớn)
t=100: correction ≈ 1× (nhỏ)
```

---

# B7. Transformer & Attention

## Self-Attention - Tính toán với số

```
INPUT: 4 patches, embedding dim = 3
X = [[1,0,1], [0,1,0], [1,1,0], [0,0,1]]

WEIGHT MATRICES (learned):
W_Q, W_K, W_V = 3×3 matrices

STEP 1: Tính Q, K, V
Q = X @ W_Q  # (4, 3)
K = X @ W_K  # (4, 3)
V = X @ W_V  # (4, 3)

STEP 2: Attention scores = Q @ K^T
scores[i][j] = "patch i attend đến patch j bao nhiêu?"

STEP 3: Scale bởi √d_k
scaled_scores = scores / √3

STEP 4: Softmax (theo hàng)
attention_weights = softmax(scaled_scores)
→ Mỗi hàng sum = 1

STEP 5: Weighted sum
output = attention_weights @ V

→ Mỗi patch output = weighted combination của tất cả patches!
```

## Công thức Self-Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Tại sao chia √d_k?

```
Khi d_k lớn, dot product Q·K có thể rất lớn
→ softmax saturate (output gần 0 hoặc 1)
→ Gradient gần 0

Chia √d_k giữ variance ổn định:
Var(Q·K) ≈ d_k
Var(Q·K / √d_k) ≈ 1 ✓
```

## Multi-Head Attention

```
1 HEAD = 1 loại relationship

4 HEADS có thể học 4 loại khác nhau:
- Head 1: "Texture similarity"
- Head 2: "Spatial proximity"
- Head 3: "Contrast detection"
- Head 4: "Abnormality clustering"

COMPUTATION:
head_i = Attention(XW_Qi, XW_Ki, XW_Vi)
MultiHead = Concat(head_1, ..., head_h) @ W_O
```

## Position Embedding

```
VẤN ĐỀ: Attention là permutation invariant!
Input [A, B, C] và [C, A, B] cho cùng output nếu không có position info.

GIẢI PHÁP: Learnable position embeddings
pos_embed = Parameter(shape=(num_patches, embed_dim))
output = patch_embed + pos_embed

→ Network TỰ HỌC được 2D spatial structure!
```

## Layer Norm vs Batch Norm

```
BATCH NORM: Normalize across BATCH (vertical)
- Cần batch statistics
- Khác training/inference
- Không tốt cho sequence

LAYER NORM: Normalize across FEATURES (horizontal)  
- Independent of batch size
- Same training/inference
- Tốt cho Transformers!
```

---

# B8. Transfer Learning

## Feature Extraction vs Fine-tuning

```
FEATURE EXTRACTION (Freeze all):
┌─────────────────────────────────┐
│  Pretrained layers              │ ← FROZEN
├─────────────────────────────────┤
│  New classification head        │ ← TRAINABLE
└─────────────────────────────────┘
✓ Fast, works with small data
✗ Cannot adapt features

FINE-TUNING (Unfreeze some/all):
┌─────────────────────────────────┐
│  Early layers (generic)         │ ← Small LR
├─────────────────────────────────┤
│  Later layers (task-specific)   │ ← Medium LR
├─────────────────────────────────┤
│  Classification head            │ ← Large LR
└─────────────────────────────────┘
✓ Better performance
✗ Needs careful tuning
```

## Gradual Unfreezing Strategy

```
Epoch 1-3: Only train head
Epoch 4-6: Unfreeze top 2 layers
Epoch 7+:  Unfreeze all with small LR

→ Avoid destroying pretrained features!
```

## Domain Adaptation: ImageNet → X-ray

```
SOURCE (ImageNet):           TARGET (X-ray):
- Natural images             - Medical images
- RGB color                  - Grayscale
- Objects centered           - Subtle differences
- Large viewpoint variation  - Fixed viewpoint

BRIDGING THE GAP:
Early layers: edges, textures ✓ (universal)
Middle layers: need adapt
Top layers: must relearn

STRATEGIES:
1. Grayscale → RGB: Copy channel 3 times
2. Use ImageNet normalization
3. Domain-specific augmentation
4. Progressive training
```

---

# B9. Evaluation Metrics

## Confusion Matrix

```
                 Predicted
              Positive  Negative
Actual    ┌─────────┬─────────┐
Positive  │   TP    │   FN    │
          ├─────────┼─────────┤
Negative  │   FP    │   TN    │
          └─────────┴─────────┘

TP: Có bệnh, dự đoán có bệnh ✓
TN: Không bệnh, dự đoán không bệnh ✓
FP: Không bệnh, dự đoán có bệnh ✗ (False Alarm)
FN: Có bệnh, dự đoán không bệnh ✗ (Missed!)
```

## Metrics từ Confusion Matrix

```
ACCURACY = (TP + TN) / (TP + TN + FP + FN)
⚠️ Misleading với imbalanced data!

PRECISION = TP / (TP + FP)
"Trong dự đoán positive, bao nhiêu % đúng?"
→ Quan trọng khi FP costly

RECALL = TP / (TP + FN)
"Trong actual positive, bao nhiêu % được detect?"
→ Quan trọng khi FN costly (MEDICAL!)

F1-SCORE = 2 × Precision × Recall / (Precision + Recall)
→ Harmonic mean, balance cả hai

SPECIFICITY = TN / (TN + FP)
"Trong actual negative, bao nhiêu % đúng?"
```

## ROC và AUC

```
ROC Curve:
- X-axis: FPR = FP / (FP + TN) = 1 - Specificity
- Y-axis: TPR = TP / (TP + FN) = Recall

AUC (Area Under Curve):
- 0.5 = Random guess
- 0.7 = Acceptable
- 0.8 = Good
- 0.9 = Excellent
- 1.0 = Perfect

INTERPRETATION:
AUC = P(random positive ranked higher than random negative)
```

## Multi-label Metrics

```
HAMMING LOSS:
= % labels predicted wrong
True:  [1, 0, 1, 1, 0]
Pred:  [1, 1, 1, 0, 0]
        ✓  ✗  ✓  ✗  ✓  → 2/5 = 0.4

MACRO AVERAGING:
Tính metric cho MỖI class, rồi average
→ Treats all classes equally
→ Good for rare classes

MICRO AVERAGING:
Pool ALL TP, FP, FN rồi tính
→ Dominated by frequent classes
→ Good for overall performance
```

## Medical Context: Recall vs Precision

```
SCREENING (detect disease):
FN = Miss bệnh → NGUY HIỂM!
→ Optimize RECALL
→ Lower threshold (0.3)

SURGERY DECISION:
FP = Unnecessary surgery → NGUY HIỂM!
→ Optimize PRECISION
→ Higher threshold (0.7)

CHEST X-RAY PROJECT:
- Screening → High Recall
- Diagnosis → Balance (F1-score)
- Comparison → AUC-ROC
```

---

# ═══════════════════════════════════════════════════════════════
# PHẦN C: GIẢI THÍCH CODE CHI TIẾT
# ═══════════════════════════════════════════════════════════════

# C1. Data Pipeline - Giải thích chuyên sâu

## C1.1. DatasetParser Class - Phân tích từng dòng

```python
class DatasetParser():
    """
    Class quản lý và xử lý dataset NIH Chest X-ray14.
    Chức năng chính:
    1. Load và index tất cả ảnh PNG
    2. Parse labels từ CSV
    3. Chuyển đổi multi-label sang one-hot encoding
    4. Hỗ trợ weighted sampling cho class imbalance
    """
    
    def __init__(self, root_dir, images_dir, labels_csv):
        """
        PARAMETERS:
        - root_dir: Thư mục gốc chứa data (vd: '/path/to/archive/sample')
        - images_dir: Thư mục con chứa ảnh (vd: 'sample/images')
        - labels_csv: File CSV chứa labels (vd: 'sample_labels.csv')
        """
        
        # ═══════════════════════════════════════════════════════════
        # BƯỚC 1: Load tất cả đường dẫn ảnh PNG
        # ═══════════════════════════════════════════════════════════
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, images_dir, "*.png")))
        # 
        # GIẢI THÍCH:
        # - glob.glob(): Tìm tất cả files match pattern "*.png"
        # - os.path.join(): Nối path an toàn (tự xử lý / hay \)
        # - sorted(): Sắp xếp để đảm bảo reproducibility
        #
        # VÍ DỤ:
        # root_dir = '/data/chest-xray'
        # images_dir = 'images'
        # Pattern = '/data/chest-xray/images/*.png'
        # Kết quả: ['00000001_000.png', '00000001_001.png', ...]
        
        # ═══════════════════════════════════════════════════════════
        # BƯỚC 2: Load và parse labels từ CSV
        # ═══════════════════════════════════════════════════════════
        self.labels_df = self._labels_by_task(root_dir=root_dir, labels_csv=labels_csv)
        
        # ═══════════════════════════════════════════════════════════
        # BƯỚC 3: Định nghĩa 15 class labels
        # ═══════════════════════════════════════════════════════════
        self.labels = [
            'Cardiomegaly',       # 0: Tim to
            'Emphysema',          # 1: Khí phế thũng
            'Effusion',           # 2: Tràn dịch màng phổi
            'Hernia',             # 3: Thoát vị cơ hoành
            'Nodule',             # 4: Nốt phổi
            'Pneumothorax',       # 5: Tràn khí màng phổi
            'Atelectasis',        # 6: Xẹp phổi
            'Pleural_Thickening', # 7: Dày màng phổi
            'Mass',               # 8: Khối u
            'Edema',              # 9: Phù phổi
            'Consolidation',      # 10: Đông đặc phổi
            'Infiltration',       # 11: Thâm nhiễm
            'Fibrosis',           # 12: Xơ phổi
            'Pneumonia',          # 13: Viêm phổi
            'No Finding'          # 14: Bình thường
        ]
        # Thứ tự QUAN TRỌNG: One-hot encoding sẽ theo thứ tự này!
```

### _labels_by_task() - Parse CSV Labels

```python
    def _labels_by_task(self, root_dir=None, labels_csv=None):
        """
        Parse file CSV và tạo DataFrame với cột ['Id', 'Label']
        Label là LIST các bệnh (cho multi-label)
        """
        
        # ═══════════════════════════════════════════════════════════
        # BƯỚC 1: Load CSV gốc
        # ═══════════════════════════════════════════════════════════
        labels_df = pd.read_csv(os.path.join(root_dir, labels_csv))
        #
        # CSV FORMAT (Data_Entry_2017_v2020.csv):
        # ┌───────────────────┬─────────────────────────────┬─────────────┐
        # │ Image Index       │ Finding Labels              │ Patient ID  │
        # ├───────────────────┼─────────────────────────────┼─────────────┤
        # │ 00000001_000.png  │ Cardiomegaly|Emphysema     │ 1           │
        # │ 00000002_000.png  │ No Finding                  │ 2           │
        # └───────────────────┴─────────────────────────────┴─────────────┘
        
        # ═══════════════════════════════════════════════════════════
        # BƯỚC 2: Tạo dictionary {filename: full_path}
        # ═══════════════════════════════════════════════════════════
        image_path = {
            os.path.basename(x): x 
            for x in glob.glob(os.path.join(root_dir, 'images', '*.png'))
        }
        # VÍ DỤ: {'00000001_000.png': '/data/images/00000001_000.png', ...}
        
        # ═══════════════════════════════════════════════════════════
        # BƯỚC 3: Lọc chỉ giữ ảnh có trong thư mục
        # ═══════════════════════════════════════════════════════════
        labels_df = labels_df[
            labels_df['Image Index'].map(os.path.basename).isin(image_path)
        ]
        # ⚠️ QUAN TRỌNG: CSV có thể chứa nhiều ảnh hơn thư mục thực tế
        # Bước này loại bỏ các entries không có ảnh tương ứng
        
        # ═══════════════════════════════════════════════════════════
        # BƯỚC 4: Tạo DataFrame mới với format chuẩn
        # ═══════════════════════════════════════════════════════════
        new_labels_df = pd.DataFrame()
        new_labels_df['Id'] = labels_df['Image Index'].copy()
        
        # Chuyển "Cardiomegaly|Emphysema" → ['Cardiomegaly', 'Emphysema']
        new_labels_df['Label'] = labels_df['Finding Labels'].apply(
            lambda val: val.split('|')
        )
        
        # ═══════════════════════════════════════════════════════════
        # BƯỚC 5: Giải phóng bộ nhớ
        # ═══════════════════════════════════════════════════════════
        del labels_df  # CSV gốc có thể rất lớn (~112K rows)
        
        return new_labels_df
```

### get_labels_df() - One-Hot Encoding

```python
    def get_labels_df(self):
        """
        Chuyển đổi Label từ LIST bệnh sang ONE-HOT VECTOR
        
        VÍ DỤ:
        Input:  ['Cardiomegaly', 'Emphysema']
        Output: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                 ↑  ↑
                 Cardiomegaly (index 0)
                    Emphysema (index 1)
        """
        new_labels_df = self.labels_df.copy()
        
        for i in range(len(new_labels_df)):
            # Khởi tạo vector zeros
            one_hot = [0 for element in self.labels]  # [0,0,0,...,0] (15 zeros)
            
            # Set 1 cho mỗi bệnh có trong list
            for element in new_labels_df['Label'][i]:
                one_hot[self.labels.index(element)] = 1
            
            # Ghi đè cột Label
            new_labels_df['Label'][i] = one_hot
        
        return new_labels_df
        
        # ⚠️ PERFORMANCE NOTE:
        # Code này chậm do dùng iterative approach
        # Có thể optimize với sklearn.preprocessing.MultiLabelBinarizer:
        # 
        # from sklearn.preprocessing import MultiLabelBinarizer
        # mlb = MultiLabelBinarizer(classes=self.labels)
        # one_hot_labels = mlb.fit_transform(new_labels_df['Label'])
```

### sample() - Weighted Sampling cho Class Imbalance

```python
    def sample(self, num_samples, is_weighted=False):
        """
        Lấy mẫu từ dataset với option weighted sampling.
        
        PARAMETERS:
        - num_samples: Số lượng mẫu cần lấy
        - is_weighted: Nếu True, ưu tiên ảnh có nhiều bệnh hơn
        
        WHY WEIGHTED SAMPLING?
        Dataset có ~75% ảnh "No Finding" (chỉ 1 label)
        Weighted sampling giúp model học được các trường hợp multi-label
        """
        
        if not is_weighted:
            # Random sampling đều
            return self.labels_df.sample(num_samples)
        else:
            # ═══════════════════════════════════════════════════════
            # Weighted sampling: Ưu tiên ảnh có nhiều labels
            # ═══════════════════════════════════════════════════════
            
            # Tính weight = số lượng labels + smoothing factor
            sample_weights = self.labels_df['Label'].map(
                lambda x: len(x)  # Số bệnh trong ảnh
            ).values + 4e-2       # Smoothing để tránh division by zero
            
            # Normalize thành probability distribution
            sample_weights /= sample_weights.sum()
            
            # Sample với weights
            return self.labels_df.sample(num_samples, weights=sample_weights)
            
            # VÍ DỤ:
            # Ảnh A: 1 bệnh → weight = 1 + 0.04 = 1.04
            # Ảnh B: 3 bệnh → weight = 3 + 0.04 = 3.04
            # Ảnh B có xác suất được chọn cao hơn ~3 lần!
```

## C1.2. ImageDataGenerator - Data Augmentation Chi Tiết

```python
# ═══════════════════════════════════════════════════════════════
# TRAINING DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════
train_datagen = ImageDataGenerator(
    # ────────────────────────────────────────────────────────────
    # 1. RESCALE: Normalize pixel values
    # ────────────────────────────────────────────────────────────
    rescale=1./255,
    # Original: [0, 255] integers
    # After:    [0.0, 1.0] floats
    #
    # TẠI SAO?
    # - Neural networks học tốt hơn với small values
    # - Tránh exploding gradients
    # - Consistent với pretrained models (đều dùng [0,1])
    
    # ────────────────────────────────────────────────────────────
    # 2. HORIZONTAL FLIP: Lật ngang
    # ────────────────────────────────────────────────────────────
    horizontal_flip=True,
    # ✓ VALID cho X-ray vì:
    # - Cơ thể người đối xứng (gần như)
    # - Bệnh có thể xuất hiện ở phổi trái hoặc phải
    # - Tăng data diversity 2x
    
    # ────────────────────────────────────────────────────────────
    # 3. VERTICAL FLIP: Lật dọc
    # ────────────────────────────────────────────────────────────
    vertical_flip=False,
    # ✗ INVALID cho X-ray vì:
    # - X-ray có orientation cố định (đầu trên, chân dưới)
    # - Lật dọc tạo ra ảnh không realistic
    # - Tim ở vị trí cố định (trái-dưới)
    
    # ────────────────────────────────────────────────────────────
    # 4. HEIGHT/WIDTH SHIFT: Dịch chuyển
    # ────────────────────────────────────────────────────────────
    height_shift_range=0.05,   # Dịch dọc ±5%
    width_shift_range=0.1,     # Dịch ngang ±10%
    # Giả lập:
    # - Patient positioning variations
    # - Different X-ray machine setups
    
    # ────────────────────────────────────────────────────────────
    # 5. ROTATION: Xoay nhẹ
    # ────────────────────────────────────────────────────────────
    rotation_range=5,          # ±5 degrees
    # CHỈ xoay NHẸ vì:
    # - X-ray thường được chụp thẳng
    # - Xoay nhiều tạo artifacts không realistic
    
    # ────────────────────────────────────────────────────────────
    # 6. SHEAR: Biến dạng góc
    # ────────────────────────────────────────────────────────────
    shear_range=0.1,           # Shear intensity 0.1
    # Giả lập oblique X-ray angles
    
    # ────────────────────────────────────────────────────────────
    # 7. ZOOM: Phóng to/thu nhỏ
    # ────────────────────────────────────────────────────────────
    zoom_range=0.15,           # Zoom ±15%
    # Giả lập:
    # - Different patient distances from X-ray source
    # - Different lung sizes
    
    # ────────────────────────────────────────────────────────────
    # 8. FILL MODE: Cách điền pixels trống
    # ────────────────────────────────────────────────────────────
    fill_mode='reflect'
    # Khi shift/rotate, sẽ có vùng trống
    # 'reflect': Mirror pixels ở boundary
    # Alternatives: 'constant' (đen), 'nearest', 'wrap'
)

# ═══════════════════════════════════════════════════════════════
# VALIDATION/TEST DATA: KHÔNG AUGMENTATION!
# ═══════════════════════════════════════════════════════════════
val_datagen = ImageDataGenerator(rescale=1./255)
# Chỉ rescale, không augment
# TẠI SAO? Validation phải phản ánh real-world performance
```

## C1.3. Data Generators - Flow from DataFrame

```python
# ═══════════════════════════════════════════════════════════════
# TRAINING GENERATOR
# ═══════════════════════════════════════════════════════════════
train_generator = train_datagen.flow_from_dataframe(
    # ────────────────────────────────────────────────────────────
    # DataFrame chứa file paths và labels
    # ────────────────────────────────────────────────────────────
    dataframe=train,
    # Columns: ['Id', 'Label']
    # Id: '00000001_000.png'
    # Label: ['Cardiomegaly', 'Emphysema']  # hoặc one-hot vector
    
    # ────────────────────────────────────────────────────────────
    # Thư mục chứa ảnh
    # ────────────────────────────────────────────────────────────
    directory='/path/to/images',
    
    # ────────────────────────────────────────────────────────────
    # Column mapping
    # ────────────────────────────────────────────────────────────
    x_col="Id",        # Column chứa filename
    y_col="Label",     # Column chứa labels
    
    # ────────────────────────────────────────────────────────────
    # Batch size cho training
    # ────────────────────────────────────────────────────────────
    batch_size=32,
    # 32 là common choice:
    # - Đủ lớn để stable gradients
    # - Đủ nhỏ để fit GPU memory
    # - Good balance speed vs convergence
    
    # ────────────────────────────────────────────────────────────
    # Target image size
    # ────────────────────────────────────────────────────────────
    target_size=(224, 224),
    # TẠI SAO 224×224?
    # - Standard size cho ImageNet pretrained models
    # - ViT-B/16 expects 224×224
    # - ResNet expects 224×224
    # - Cân bằng giữa detail và computation
    
    # ────────────────────────────────────────────────────────────
    # Class labels (thứ tự quan trọng!)
    # ────────────────────────────────────────────────────────────
    classes=parser.labels
    # Đảm bảo one-hot encoding consistent với model output
)

# OUTPUT của generator:
# images: (batch_size, 224, 224, 3) - float32 [0,1]
# labels: (batch_size, 15) - one-hot vectors
```

---

# C2. CNN Model - Giải thích chuyên sâu

## C2.1. Kiến trúc CNN đầy đủ

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def create_cnn_classifier():
    """
    Tạo CNN classifier đơn giản cho multi-label classification.
    
    ARCHITECTURE OVERVIEW:
    Input (224,224,3) → Conv → Pool → Conv → Pool → Flatten → Dense → Dense → Output (15)
    """
    
    model = Sequential([
        # ════════════════════════════════════════════════════════
        # LAYER 1: First Convolution Block
        # ════════════════════════════════════════════════════════
        Conv2D(
            filters=32,              # Số lượng kernels/filters
            kernel_size=(3, 3),      # Kích thước mỗi kernel
            activation='relu',        # Activation function
            input_shape=(224, 224, 3) # H×W×C (RGB)
        ),
        # 
        # INPUT:  (batch, 224, 224, 3)
        # OUTPUT: (batch, 222, 222, 32)
        #
        # TÍNH TOÁN OUTPUT SIZE:
        # out = (input - kernel + 2*padding) / stride + 1
        # out = (224 - 3 + 0) / 1 + 1 = 222
        #
        # PARAMETERS:
        # weights: 3×3×3×32 = 864
        # bias: 32
        # Total: 896 params
        #
        # CONV2D HỌC GÌ?
        # Layer đầu học LOW-LEVEL FEATURES:
        # - Edges (cạnh)
        # - Corners (góc)
        # - Textures (kết cấu)
        # - Color gradients
        
        # ════════════════════════════════════════════════════════
        # LAYER 2: First Max Pooling
        # ════════════════════════════════════════════════════════
        MaxPooling2D(pool_size=(2, 2)),
        #
        # INPUT:  (batch, 222, 222, 32)
        # OUTPUT: (batch, 111, 111, 32)
        #
        # HOẠT ĐỘNG:
        # Lấy MAX trong mỗi vùng 2×2, stride=2
        # 222 / 2 = 111
        #
        # TẠI SAO MAX POOLING?
        # 1. Giảm spatial dimensions (computation)
        # 2. Giữ features mạnh nhất
        # 3. Tăng receptive field
        # 4. Translation invariance
        #
        # PARAMETERS: 0 (không học được!)
        
        # ════════════════════════════════════════════════════════
        # LAYER 3: Second Convolution Block
        # ════════════════════════════════════════════════════════
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            name="last_conv_layer"  # Name cho visualization
        ),
        #
        # INPUT:  (batch, 111, 111, 32)
        # OUTPUT: (batch, 109, 109, 64)
        #
        # PARAMETERS:
        # weights: 3×3×32×64 = 18,432
        # bias: 64
        # Total: 18,496 params
        #
        # LAYER 2 HỌC GÌ?
        # MID-LEVEL FEATURES:
        # - Combinations of edges → shapes
        # - Patterns specific to lungs
        # - Rib cage structures
        
        # ════════════════════════════════════════════════════════
        # LAYER 4: Second Max Pooling
        # ════════════════════════════════════════════════════════
        MaxPooling2D(pool_size=(2, 2)),
        #
        # INPUT:  (batch, 109, 109, 64)
        # OUTPUT: (batch, 54, 54, 64)
        #
        # 109 / 2 = 54 (integer division)
        
        # ════════════════════════════════════════════════════════
        # LAYER 5: Flatten
        # ════════════════════════════════════════════════════════
        Flatten(),
        #
        # INPUT:  (batch, 54, 54, 64)
        # OUTPUT: (batch, 186624)
        #
        # TÍNH TOÁN:
        # 54 × 54 × 64 = 186,624 neurons!
        #
        # ⚠️ ĐÂY LÀ VẤN ĐỀ!
        # Flatten tạo ra vector quá lớn
        # → Dense layer tiếp theo sẽ có quá nhiều params
        
        # ════════════════════════════════════════════════════════
        # LAYER 6: Dense (Fully Connected)
        # ════════════════════════════════════════════════════════
        Dense(512, activation='relu'),
        #
        # INPUT:  (batch, 186624)
        # OUTPUT: (batch, 512)
        #
        # PARAMETERS:
        # weights: 186,624 × 512 = 95,551,488
        # bias: 512
        # Total: 95,552,000 params
        #
        # 🔴 CRITICAL ISSUE:
        # 95.5M params trong 1 layer!
        # = 99% tổng số params của model!
        # → SEVERE OVERFITTING
        
        # ════════════════════════════════════════════════════════
        # LAYER 7: Output Layer
        # ════════════════════════════════════════════════════════
        Dense(num_classes, activation='sigmoid')
        #
        # INPUT:  (batch, 512)
        # OUTPUT: (batch, 15)
        #
        # PARAMETERS:
        # weights: 512 × 15 = 7,680
        # bias: 15
        # Total: 7,695 params
        #
        # TẠI SAO SIGMOID (không phải SOFTMAX)?
        # - Multi-label classification
        # - Mỗi output INDEPENDENT
        # - Có thể nhiều bệnh cùng lúc
        # - Output: probability cho MỖI class
    ])
    
    return model
```

## C2.2. Phân tích vấn đề CNN

```
╔═══════════════════════════════════════════════════════════════╗
║                    CNN PARAMETER BREAKDOWN                     ║
╠═══════════════════════════════════════════════════════════════╣
║  Layer              │  Output Shape    │  Parameters          ║
╠═════════════════════╪══════════════════╪══════════════════════╣
║  Conv2D (32)        │  (222, 222, 32)  │  896                 ║
║  MaxPooling2D       │  (111, 111, 32)  │  0                   ║
║  Conv2D (64)        │  (109, 109, 64)  │  18,496              ║
║  MaxPooling2D       │  (54, 54, 64)    │  0                   ║
║  Flatten            │  (186624,)       │  0                   ║
║  Dense (512)        │  (512,)          │  95,552,000  ← 99%!  ║
║  Dense (15)         │  (15,)           │  7,695               ║
╠═════════════════════╪══════════════════╪══════════════════════╣
║  TOTAL              │                  │  95,579,087          ║
╚═══════════════════════════════════════════════════════════════╝

🔴 PROBLEMS:

1. PARAMETER IMBALANCE:
   - 99% params trong Dense layer
   - Conv layers chỉ có ~20K params
   - Model chủ yếu "memorize" thay vì "learn features"

2. OVERFITTING:
   - Train accuracy: ~95%
   - Val accuracy: ~70%
   - Gap 25% = severe overfitting!

3. INSUFFICIENT DEPTH:
   - Chỉ 2 conv layers
   - Receptive field nhỏ
   - Không học được high-level features

4. NO REGULARIZATION:
   - Không có BatchNorm
   - Không có Dropout
   - Không có L2 regularization

✅ RECOMMENDED FIXES:

1. REPLACE Flatten với GlobalAveragePooling:
   Before: (54,54,64) → Flatten → 186,624 neurons
   After:  (54,54,64) → GAP → 64 neurons
   → Giảm 2900 lần params!

2. ADD BatchNormalization sau mỗi Conv

3. ADD Dropout trước Dense layers

4. INCREASE Conv layers (4-5 layers)

5. HOẶC: Dùng Transfer Learning!
```

## C2.3. Training Loop - run_experiment()

```python
def run_experiment(model):
    """
    Compile và train model với các settings chuẩn.
    """
    
    # ════════════════════════════════════════════════════════════
    # OPTIMIZER: AdamW
    # ════════════════════════════════════════════════════════════
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,  # 1e-4 = 0.0001
        weight_decay=weight_decay     # 1e-6 = L2 regularization
    )
    #
    # TẠI SAO AdamW?
    # - Adam: Adaptive learning rate (momentum + RMSprop)
    # - W: Weight decay (L2 regularization decoupled)
    # - Tốt hơn Adam thường cho regularization
    #
    # LEARNING RATE 1e-4:
    # - Không quá lớn (unstable)
    # - Không quá nhỏ (slow convergence)
    # - Standard cho fine-tuning
    
    # ════════════════════════════════════════════════════════════
    # COMPILE MODEL
    # ════════════════════════════════════════════════════════════
    model.compile(
        optimizer=optimizer,
        
        # ──────────────────────────────────────────────────────
        # LOSS FUNCTION: Binary Cross-Entropy
        # ──────────────────────────────────────────────────────
        loss='binary_crossentropy',
        # TẠI SAO BCE?
        # - Multi-label: Mỗi output là binary classification
        # - 15 independent binary classifiers
        # - Sigmoid output + BCE loss
        #
        # BCE = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
        # Average over all 15 classes
        
        # ──────────────────────────────────────────────────────
        # METRICS
        # ──────────────────────────────────────────────────────
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            # Tính % predictions đúng
            # (ŷ > 0.5) == y
            
            keras.metrics.AUC(name="auc"),
            # Area Under ROC Curve
            # Không phụ thuộc threshold
            # 0.5 = random, 1.0 = perfect
        ]
    )
    
    # ════════════════════════════════════════════════════════════
    # TRAINING
    # ════════════════════════════════════════════════════════════
    history = model.fit(
        train_generator,
        epochs=num_epochs,           # 10 epochs
        validation_data=validation_generator,
        
        callbacks=[
            # ──────────────────────────────────────────────────
            # ModelCheckpoint: Save best model
            # ──────────────────────────────────────────────────
            ModelCheckpoint(
                os.path.join("files", "model.keras"),
                monitor='val_loss',   # Theo dõi validation loss
                verbose=1,
                save_best_only=True   # Chỉ save khi improve
            )
            # TẠI SAO monitor val_loss?
            # - Phát hiện overfitting sớm
            # - Val_loss tăng = overfitting
        ]
    )
    
    return history

# TRAINING OUTPUT EXAMPLE:
# Epoch 1/10
# 1875/1875 [==============================] - 45s 24ms/step
# loss: 0.2345 - accuracy: 0.8234 - auc: 0.7123
# val_loss: 0.3456 - val_accuracy: 0.7654 - val_auc: 0.6543
```

---

# C3. ResNet Model - Giải thích chuyên sâu

## C3.1. Residual Block - Trái tim của ResNet

```python
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, 
    MaxPooling2D, GlobalAveragePooling2D, Dense, Add
)
from tensorflow.keras.models import Model

def block(x, filters, strides=1):
    """
    Basic Residual Block cho ResNet-34
    
    ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │                        INPUT x                              │
    │                           │                                 │
    │           ┌───────────────┼───────────────┐                │
    │           │               │               │                │
    │           ▼               │               │                │
    │    ┌─────────────┐        │               │                │
    │    │ Conv 3×3    │        │               │                │
    │    │ BatchNorm   │        │    Identity   │                │
    │    │ ReLU        │        │    Shortcut   │                │
    │    └─────────────┘        │               │                │
    │           │               │               │                │
    │           ▼               │               │                │
    │    ┌─────────────┐        │               │                │
    │    │ Conv 3×3    │        │    (Projection│                │
    │    │ BatchNorm   │        │    if needed) │                │
    │    └─────────────┘        │               │                │
    │           │               │               │                │
    │           └───────────────┼───────────────┘                │
    │                           │                                 │
    │                           ▼                                 │
    │                      ADD (x + F(x))                        │
    │                           │                                 │
    │                           ▼                                 │
    │                        ReLU                                 │
    │                           │                                 │
    │                        OUTPUT                               │
    └─────────────────────────────────────────────────────────────┘
    
    PARAMETERS:
    - x: Input tensor
    - filters: Số filters cho conv layers
    - strides: 1 (same size) hoặc 2 (downsampling)
    """
    
    # ════════════════════════════════════════════════════════════
    # SAVE IDENTITY FOR SKIP CONNECTION
    # ════════════════════════════════════════════════════════════
    identity = x
    # Giữ lại input gốc để cộng vào sau
    
    # ════════════════════════════════════════════════════════════
    # FIRST CONV-BN-RELU
    # ════════════════════════════════════════════════════════════
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=strides,      # Có thể là 1 hoặc 2
        padding='same'        # Giữ spatial size (khi stride=1)
    )(x)
    #
    # VÍ DỤ với strides=2:
    # Input: (56, 56, 64)
    # Output: (28, 28, 128)  ← Halved spatial, doubled channels
    
    x = BatchNormalization()(x)
    #
    # BATCH NORMALIZATION:
    # Normalize activations trong mỗi mini-batch
    # x̂ = (x - μ) / σ
    # y = γx̂ + β (learnable scale và shift)
    #
    # BENEFITS:
    # 1. Faster training
    # 2. Higher learning rates
    # 3. Reduces internal covariate shift
    # 4. Regularization effect
    
    x = Activation('relu')(x)
    
    # ════════════════════════════════════════════════════════════
    # SECOND CONV-BN (no activation yet!)
    # ════════════════════════════════════════════════════════════
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,            # Luôn là 1 cho conv thứ 2
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    # KHÔNG có ReLU ở đây!
    # ReLU sẽ được apply SAU khi cộng với identity
    
    # ════════════════════════════════════════════════════════════
    # PROJECTION SHORTCUT (if needed)
    # ════════════════════════════════════════════════════════════
    if strides != 1 or identity.shape[-1] != filters:
        # CẦN projection khi:
        # 1. strides != 1: Spatial size thay đổi
        # 2. channels thay đổi: identity.shape[-1] != filters
        
        identity = Conv2D(
            filters=filters,
            kernel_size=1,    # 1×1 convolution
            strides=strides,  # Match stride của main path
            padding='same'
        )(identity)
        identity = BatchNormalization()(identity)
        
        # 1×1 CONVOLUTION:
        # - Chỉ thay đổi số channels
        # - Với strides=2: cũng downsampling
        # - Rất ít params: filters × in_channels
    
    # ════════════════════════════════════════════════════════════
    # RESIDUAL CONNECTION: y = F(x) + x
    # ════════════════════════════════════════════════════════════
    x = x + identity
    # hoặc: x = Add()([x, identity])
    #
    # ĐÂY LÀ MAGIC CỦA RESNET!
    #
    # MATHEMATICAL INSIGHT:
    # Gradient qua block:
    # ∂L/∂x = ∂L/∂y × (∂F/∂x + 1)
    #                        ↑
    #                   IDENTITY TERM!
    #
    # Gradient LUÔN có term "+1"
    # → Gradient KHÔNG THỂ vanish hoàn toàn!
    # → Có thể train networks rất sâu (152+ layers)
    
    x = Activation('relu')(x)
    # ReLU cuối cùng sau khi cộng
    
    return x
```

## C3.2. Full ResNet-34 Architecture

```python
def create_resnet():
    """
    Tạo ResNet-34 từ scratch.
    
    ARCHITECTURE:
    - STEM: 7×7 conv, maxpool
    - STAGE 1: 3 blocks, 64 filters
    - STAGE 2: 4 blocks, 128 filters
    - STAGE 3: 6 blocks, 256 filters
    - STAGE 4: 3 blocks, 512 filters
    - HEAD: GlobalAvgPool, Dense(15)
    
    WHY "34"?
    1 (stem conv) + 2×(3+4+6+3) = 1 + 32 = 33 conv layers
    + 1 dense layer = 34 layers
    """
    
    inputs = Input(shape=input_shape)  # (224, 224, 3)
    
    # ════════════════════════════════════════════════════════════
    # STEM: Initial Feature Extraction
    # ════════════════════════════════════════════════════════════
    x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    # Input: (224, 224, 3) → Output: (112, 112, 64)
    # 7×7 kernel captures large-scale features
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    # (112, 112, 64) → (56, 56, 64)
    # Aggressive downsampling ở đầu
    
    # ════════════════════════════════════════════════════════════
    # STAGE 1: 3 Residual Blocks, 64 filters
    # ════════════════════════════════════════════════════════════
    x = block(x, 64)  # Giữ nguyên size: (56, 56, 64)
    x = block(x, 64)
    x = block(x, 64)
    # Spatial: 56×56, Channels: 64
    # LOW-LEVEL FEATURES: edges, textures
    
    # ════════════════════════════════════════════════════════════
    # STAGE 2: 4 Residual Blocks, 128 filters
    # ════════════════════════════════════════════════════════════
    x = block(x, 128, strides=2)  # Downsample: (56,56,64) → (28,28,128)
    x = block(x, 128)
    x = block(x, 128)
    x = block(x, 128)
    # Spatial: 28×28, Channels: 128
    # MID-LEVEL FEATURES: shapes, local patterns
    
    # ════════════════════════════════════════════════════════════
    # STAGE 3: 6 Residual Blocks, 256 filters
    # ════════════════════════════════════════════════════════════
    x = block(x, 256, strides=2)  # Downsample: (28,28,128) → (14,14,256)
    x = block(x, 256)
    x = block(x, 256)
    x = block(x, 256)
    x = block(x, 256)
    x = block(x, 256)
    # Spatial: 14×14, Channels: 256
    # HIGH-LEVEL FEATURES: complex patterns, anatomical structures
    
    # ════════════════════════════════════════════════════════════
    # STAGE 4: 3 Residual Blocks, 512 filters
    # ════════════════════════════════════════════════════════════
    x = block(x, 512, strides=2)  # Downsample: (14,14,256) → (7,7,512)
    x = block(x, 512)
    x = block(x, 512)
    # Spatial: 7×7, Channels: 512
    # SEMANTIC FEATURES: disease-specific patterns
    
    # ════════════════════════════════════════════════════════════
    # CLASSIFICATION HEAD
    # ════════════════════════════════════════════════════════════
    x = GlobalAveragePooling2D()(x)
    # (7, 7, 512) → (512,)
    # Average mỗi 7×7 feature map thành 1 số
    #
    # COMPARE WITH FLATTEN:
    # Flatten: 7×7×512 = 25,088 neurons
    # GAP:     512 neurons
    # → Giảm 49× số params trong Dense layer!
    # → Less overfitting
    
    outputs = Dense(num_classes, activation='sigmoid')(x)
    # (512,) → (15,)
    # params: 512 × 15 + 15 = 7,695
    
    model = Model(inputs, outputs)
    return model
```

## C3.3. ResNet Parameter Summary

```
╔═══════════════════════════════════════════════════════════════════╗
║                    RESNET-34 ARCHITECTURE                          ║
╠═══════════════════════════════════════════════════════════════════╣
║  Stage    │ Output Size │ Blocks │ Filters │ Params (approx)      ║
╠═══════════╪═════════════╪════════╪═════════╪══════════════════════╣
║  STEM     │ 56×56       │ -      │ 64      │ 9,472                ║
║  Stage 1  │ 56×56       │ 3      │ 64      │ 222,720              ║
║  Stage 2  │ 28×28       │ 4      │ 128     │ 1,116,416            ║
║  Stage 3  │ 14×14       │ 6      │ 256     │ 6,690,304            ║
║  Stage 4  │ 7×7         │ 3      │ 512     │ 13,304,832           ║
║  HEAD     │ 15          │ -      │ -       │ 7,695                ║
╠═══════════╪═════════════╪════════╪═════════╪══════════════════════╣
║  TOTAL    │             │ 16     │         │ ~21.3M               ║
╚═══════════════════════════════════════════════════════════════════╝

✅ ADVANTAGES OVER CNN:
1. Skip connections → no vanishing gradient
2. Can go much deeper (34 vs 2 layers)
3. BatchNorm → stable training
4. GlobalAvgPool → less overfitting
5. Better feature hierarchy
```

---

# C4. ViT Models - Giải thích chuyên sâu

## C4.1. Patches Layer - Chia ảnh thành patches

```python
from tensorflow.keras import layers
import tensorflow as tf

class Patches(layers.Layer):
    """
    Custom Keras layer để chia ảnh thành non-overlapping patches.
    
    VÍ DỤ:
    Image 224×224 với patch_size=32
    → 7×7 = 49 patches
    → Mỗi patch: 32×32×3 = 3072 pixels
    
    Image 224×224 với patch_size=16
    → 14×14 = 196 patches
    → Mỗi patch: 16×16×3 = 768 pixels
    """
    
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
    
    def call(self, images):
        """
        Forward pass: Chia ảnh thành patches.
        
        PARAMETERS:
        - images: Tensor shape (batch, height, width, channels)
        
        RETURNS:
        - patches: Tensor shape (batch, num_patches, patch_dim)
        """
        
        # Get batch size dynamically
        batch_size = tf.shape(images)[0]
        
        # ════════════════════════════════════════════════════════════
        # EXTRACT PATCHES using TensorFlow built-in
        # ════════════════════════════════════════════════════════════
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            # sizes: [batch, height, width, channels]
            # 1 image at a time, patch_size×patch_size, all channels
            
            strides=[1, self.patch_size, self.patch_size, 1],
            # Non-overlapping: stride = patch_size
            # Nếu stride < patch_size → overlapping patches
            
            rates=[1, 1, 1, 1],
            # Dilation rate (1 = no dilation)
            
            padding='VALID'
            # No padding → phải chia hết
        )
        
        # ════════════════════════════════════════════════════════════
        # RESHAPE: (batch, H', W', patch_dim) → (batch, num_patches, patch_dim)
        # ════════════════════════════════════════════════════════════
        #
        # SAU extract_patches:
        # patches.shape = (batch, num_patches_h, num_patches_w, patch_dim)
        # VD: (32, 7, 7, 3072) với 224×224 image, 32×32 patches
        #
        # patch_dim = patch_size × patch_size × channels
        #           = 32 × 32 × 3 = 3072
        
        patch_dims = patches.shape[-1]  # 3072
        
        # Reshape thành (batch, 49, 3072)
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        # -1: Tự động tính = 7×7 = 49
        
        return patches
    
    def get_config(self):
        """Cho serialization/deserialization của model."""
        config = super(Patches, self).get_config()
        config.update({"patch_size": self.patch_size})
        return config

# ════════════════════════════════════════════════════════════════════
# VISUALIZATION: Cách patches hoạt động
# ════════════════════════════════════════════════════════════════════
"""
ORIGINAL IMAGE (224×224):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│     P1      P2      P3      P4      P5      P6      P7        │
│   32×32   32×32   32×32   32×32   32×32   32×32   32×32       │
│                                                                 │
│     P8      P9      P10     P11     P12     P13     P14       │
│   32×32   32×32   32×32   32×32   32×32   32×32   32×32       │
│                                                                 │
│     ...    ...     ...     ...     ...     ...     ...        │
│                                                                 │
│     P43     P44     P45     P46     P47     P48     P49       │
│   32×32   32×32   32×32   32×32   32×32   32×32   32×32       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

→ 7 × 7 = 49 patches
→ Mỗi patch flatten: 32×32×3 = 3072 dimensions

OUTPUT TENSOR:
patches.shape = (batch_size, 49, 3072)
                     ↑       ↑     ↑
                  samples  seq_len  embed_dim (raw)
"""
```

## C4.2. PatchEncoder - Linear Projection + Position Embedding

```python
class PatchEncoder(layers.Layer):
    """
    Encode patches với:
    1. Linear projection: Giảm dimensionality
    2. Position embedding: Thêm spatial information
    
    INTUITION:
    Sau Patches layer: Mỗi patch là 3072 dims (quá lớn!)
    PatchEncoder: Project xuống projection_dim (64 hoặc 768)
    + Add position info để biết patch ở đâu trong ảnh
    """
    
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches      # 49 hoặc 196
        self.projection_dim = projection_dim # 64 hoặc 768
        
        # ════════════════════════════════════════════════════════════
        # LINEAR PROJECTION
        # ════════════════════════════════════════════════════════════
        self.projection = layers.Dense(units=projection_dim)
        # Input: 3072 → Output: projection_dim
        # Giống như "word embedding" trong NLP
        # Mỗi patch → vector trong embedding space
        
        # ════════════════════════════════════════════════════════════
        # POSITION EMBEDDING (Learnable)
        # ════════════════════════════════════════════════════════════
        self.position_embedding = layers.Embedding(
            input_dim=num_patches,     # Số vị trí có thể: 49
            output_dim=projection_dim  # Embedding dimension: 64
        )
        # Embedding table: (49, 64)
        # position_embedding[0] = vector cho patch đầu tiên
        # position_embedding[48] = vector cho patch cuối
        #
        # TẠI SAO LEARNABLE (không dùng sinusoidal)?
        # - Model có thể TỰ HỌC spatial relationships
        # - Thực tế học được 2D grid structure
        # - Simpler implementation
    
    def call(self, patch):
        """
        Forward pass: Project patches + add positions.
        
        INPUT: (batch, num_patches, patch_dim)  # (32, 49, 3072)
        OUTPUT: (batch, num_patches, projection_dim)  # (32, 49, 64)
        """
        
        # ════════════════════════════════════════════════════════════
        # TẠO POSITION INDICES
        # ════════════════════════════════════════════════════════════
        positions = tf.expand_dims(
            tf.range(start=0, limit=self.num_patches, delta=1),
            axis=0
        )
        # positions = [[0, 1, 2, ..., 48]]
        # Shape: (1, 49)
        
        # ════════════════════════════════════════════════════════════
        # LINEAR PROJECTION
        # ════════════════════════════════════════════════════════════
        projected_patches = self.projection(patch)
        # (batch, 49, 3072) @ W(3072, 64) → (batch, 49, 64)
        
        # ════════════════════════════════════════════════════════════
        # ADD POSITION EMBEDDINGS
        # ════════════════════════════════════════════════════════════
        encoded = projected_patches + self.position_embedding(positions)
        # projected_patches: (batch, 49, 64)
        # position_embedding(positions): (1, 49, 64) → broadcast
        # Result: (batch, 49, 64)
        #
        # INTUITION:
        # Patch embedding: "Đây là patch có lung tissue"
        # Position embedding: "Patch này ở góc trên trái"
        # Combined: "Lung tissue ở góc trên trái"
        
        return encoded
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config
```

## C4.3. Transformer Encoder Block

```python
def create_vit_classifier():
    """
    Tạo complete ViT classifier.
    """
    
    inputs = Input(shape=input_shape)  # (224, 224, 3)
    
    # ════════════════════════════════════════════════════════════
    # PATCH EMBEDDING
    # ════════════════════════════════════════════════════════════
    patches = Patches(patch_size)(inputs)  # (batch, 49, 3072)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    # (batch, 49, 64)
    
    # ════════════════════════════════════════════════════════════
    # TRANSFORMER ENCODER BLOCKS × 8
    # ════════════════════════════════════════════════════════════
    for _ in range(transformer_layers):  # transformer_layers = 8
        
        # ──────────────────────────────────────────────────────
        # MULTI-HEAD SELF-ATTENTION SUBLAYER
        # ──────────────────────────────────────────────────────
        
        # Layer Normalization TRƯỚC attention (Pre-LN)
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-Head Self-Attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,      # 4 heads
            key_dim=projection_dim,   # 64
            dropout=0.1
        )(x1, x1)  # Query=Key=Value=x1 → SELF-attention
        #
        # MULTI-HEAD ATTENTION BREAKDOWN:
        #
        # 1. Split into heads:
        #    x1: (batch, 49, 64) → 4 heads × (batch, 49, 16)
        #
        # 2. Compute Q, K, V for each head:
        #    Q = x1 @ W_Q  (49, 16)
        #    K = x1 @ W_K  (49, 16)
        #    V = x1 @ W_V  (49, 16)
        #
        # 3. Attention per head:
        #    scores = Q @ K^T / sqrt(16)  # (49, 49)
        #    weights = softmax(scores)    # (49, 49)
        #    output = weights @ V         # (49, 16)
        #
        # 4. Concatenate heads:
        #    concat: 4 × (49, 16) → (49, 64)
        #
        # 5. Final projection:
        #    output = concat @ W_O  # (49, 64)
        #
        # PARAMS per MHA:
        # Q, K, V projections: 3 × 64 × 64 = 12,288
        # Output projection: 64 × 64 = 4,096
        # Total: ~16K params
        
        # Residual connection
        x2 = Add()([attention_output, encoded_patches])
        
        # ──────────────────────────────────────────────────────
        # MLP (FEED-FORWARD) SUBLAYER
        # ──────────────────────────────────────────────────────
        
        # Layer Normalization TRƯỚC MLP
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP: Expand then contract
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # transformer_units = [128, 64]
        # 64 → 128 (expand) → 64 (contract)
        #
        # TẠI SAO EXPAND-CONTRACT?
        # - Larger intermediate dimension = more capacity
        # - Typical ratio: 4× (64 → 256 trong standard ViT)
        # - Giống như "bottleneck" ngược
        
        # Residual connection
        encoded_patches = Add()([x3, x2])
    
    # ════════════════════════════════════════════════════════════
    # CLASSIFICATION HEAD
    # ════════════════════════════════════════════════════════════
    
    # Final Layer Normalization
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    # Flatten all patches
    representation = Flatten()(representation)
    # (batch, 49, 64) → (batch, 3136)
    #
    # ⚠️ ALTERNATIVE: Use [CLS] token
    # Standard ViT thêm learnable [CLS] token ở đầu
    # Chỉ dùng [CLS] token output cho classification
    # Đơn giản hơn và tốt hơn Flatten
    
    representation = Dropout(0.5)(representation)
    
    # MLP Head
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # mlp_head_units = [2048, 1024]
    # 3136 → 2048 → 1024
    
    # Output layer
    logits = Dense(num_classes, activation='sigmoid')(features)
    # 1024 → 15
    
    model = Model(inputs=inputs, outputs=logits)
    return model
```

## C4.4. MLP Helper Function

```python
def mlp(x, hidden_units, dropout_rate, regularizer_rate=0.01):
    """
    Multi-Layer Perceptron với GELU activation.
    
    PARAMETERS:
    - x: Input tensor
    - hidden_units: List các layer sizes, vd [128, 64]
    - dropout_rate: Dropout probability
    - regularizer_rate: L2 regularization weight
    
    GELU (Gaussian Error Linear Unit):
    GELU(x) = x × Φ(x)
    
    Trong đó Φ(x) = CDF của standard normal distribution
    
    TẠI SAO GELU thay vì ReLU?
    - Smoother than ReLU
    - No "dead neurons" problem
    - Better for Transformers
    - Used in BERT, GPT, ViT
    """
    
    for units in hidden_units:
        x = Dense(
            units,
            activation=tf.nn.gelu,
            kernel_regularizer=l2(regularizer_rate)  # L2 regularization
        )(x)
        x = Dropout(dropout_rate)(x)
    
    return x
```

## C4.5. ViT-Pretrained (Best Model) - PyTorch

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# ════════════════════════════════════════════════════════════════════
# CUSTOM DATASET CLASS
# ════════════════════════════════════════════════════════════════════
class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset cho Chest X-ray images.
    
    Khác với TensorFlow ImageDataGenerator:
    - More control over loading process
    - Works with PyTorch DataLoader
    - Custom transforms pipeline
    """
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths  # List of file paths
        self.labels = labels            # NumPy array of one-hot labels
        self.transform = transform      # Torchvision transforms
    
    def __len__(self):
        """Số lượng samples trong dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Lấy 1 sample (image, label) tại index idx.
        
        ĐƯỢC GỌI BỞI: DataLoader khi iterate
        """
        # ──────────────────────────────────────────────────────
        # LOAD IMAGE
        # ──────────────────────────────────────────────────────
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # OpenCV loads as BGR, convert to RGB
        
        # ──────────────────────────────────────────────────────
        # APPLY TRANSFORMS
        # ──────────────────────────────────────────────────────
        if self.transform:
            image = self.transform(image)
        
        # ──────────────────────────────────────────────────────
        # GET LABEL
        # ──────────────────────────────────────────────────────
        label = self.labels[idx]
        
        return image, label

# ════════════════════════════════════════════════════════════════════
# DATA TRANSFORMS (PyTorch style)
# ════════════════════════════════════════════════════════════════════
transform = transforms.Compose([
    transforms.ToPILImage(),
    # OpenCV → PIL Image (required for torchvision transforms)
    
    transforms.Resize((224, 224)),
    # Resize về 224×224 (ViT input size)
    
    transforms.RandomHorizontalFlip(),
    # Random flip với p=0.5
    
    transforms.RandomRotation(20),
    # Random rotation ±20 degrees
    # ⚠️ Nhiều hơn TensorFlow version (5 degrees)
    
    transforms.ToTensor(),
    # PIL Image → Tensor
    # Chuyển [0,255] → [0,1]
    # Chuyển (H,W,C) → (C,H,W)
    
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # ImageNet normalization statistics!
    # QUAN TRỌNG: Pretrained model được train với stats này
    # Phải dùng CÙNG normalization khi inference
])

# ════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════════════════════════════════
batch_size = 16  # Smaller than TF version do GPU memory

loader_train = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,      # Shuffle training data
    num_workers=4      # Parallel data loading
)

loader_val = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,     # Don't shuffle validation
    num_workers=4
)

# ════════════════════════════════════════════════════════════════════
# LOAD PRETRAINED ViT
# ════════════════════════════════════════════════════════════════════
model = timm.create_model(
    'vit_base_patch16_224',  # ViT-Base with 16×16 patches
    pretrained=True          # Load ImageNet-21k pretrained weights
)
#
# ARCHITECTURE DETAILS:
# - patch_size: 16×16
# - num_patches: 14×14 = 196
# - embed_dim: 768
# - depth: 12 transformer blocks
# - num_heads: 12
# - mlp_ratio: 4 (MLP hidden = 768×4 = 3072)
# - Total params: ~86M
#
# PRETRAINED ON:
# - ImageNet-21k (14 million images, 21,843 classes)
# - Then fine-tuned on ImageNet-1k
# - Learned powerful visual representations!

# ════════════════════════════════════════════════════════════════════
# MODIFY CLASSIFICATION HEAD
# ════════════════════════════════════════════════════════════════════
num_classes = 15
model.head = nn.Linear(model.head.in_features, num_classes)
# Replace: Linear(768 → 1000) với Linear(768 → 15)
#
# GIỮ NGUYÊN:
# - Patch embedding
# - All 12 transformer blocks
# - Position embeddings
#
# CHỈ THAY:
# - Classification head (final layer)

# ════════════════════════════════════════════════════════════════════
# MOVE TO GPU
# ════════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ════════════════════════════════════════════════════════════════════
# LOSS & OPTIMIZER
# ════════════════════════════════════════════════════════════════════
criterion = nn.BCEWithLogitsLoss()
# Binary Cross-Entropy WITH LOGITS
# = Sigmoid + BCELoss combined (numerically stable)
#
# QUAN TRỌNG: Model output KHÔNG có sigmoid
# BCEWithLogitsLoss tự apply sigmoid internally

optimizer = Adam(model.parameters(), lr=1e-4)
# Adam optimizer với learning rate nhỏ
# Vì đang fine-tuning, không cần train from scratch
```

## C4.6. Training Loop (PyTorch)

```python
def train_model(model, criterion, optimizer, loader_train, loader_val, num_epochs=10):
    """
    Complete training loop cho PyTorch model.
    """
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        
        # ════════════════════════════════════════════════════════════
        # TRAINING PHASE
        # ════════════════════════════════════════════════════════════
        model.train()  # Set model to training mode
        # Enables: Dropout, BatchNorm training behavior
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for inputs, labels in loader_train:
            # ──────────────────────────────────────────────────────
            # MOVE TO GPU
            # ──────────────────────────────────────────────────────
            inputs = inputs.to(device)  # (batch, 3, 224, 224)
            labels = labels.to(device)  # (batch, 15)
            
            # ──────────────────────────────────────────────────────
            # FORWARD PASS
            # ──────────────────────────────────────────────────────
            optimizer.zero_grad()  # Clear gradients
            
            outputs = model(inputs)  # (batch, 15) - raw logits
            loss = criterion(outputs, labels)
            
            # ──────────────────────────────────────────────────────
            # PREDICTIONS
            # ──────────────────────────────────────────────────────
            preds = outputs.sigmoid() > 0.5  # Apply sigmoid, threshold
            # preds: (batch, 15) boolean tensor
            
            # ──────────────────────────────────────────────────────
            # BACKWARD PASS
            # ──────────────────────────────────────────────────────
            loss.backward()   # Compute gradients
            optimizer.step()  # Update weights
            
            # ──────────────────────────────────────────────────────
            # ACCUMULATE METRICS
            # ──────────────────────────────────────────────────────
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels.byte()).sum().item()
            total_samples += labels.numel()  # Total elements
        
        # ──────────────────────────────────────────────────────────
        # EPOCH METRICS
        # ──────────────────────────────────────────────────────────
        epoch_loss = running_loss / len(loader_train.dataset)
        epoch_acc = running_corrects / total_samples * 100
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
        
        # ════════════════════════════════════════════════════════════
        # VALIDATION PHASE
        # ════════════════════════════════════════════════════════════
        val_loss, val_acc = validate_model(model, loader_val, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
    return train_losses, val_losses, train_accuracies, val_accuracies


def validate_model(model, loader_val, criterion, threshold=0.5):
    """
    Evaluate model on validation set.
    """
    
    model.eval()  # Set model to evaluation mode
    # Disables: Dropout, BatchNorm uses running stats
    
    total_samples = 0
    total_correct = 0
    running_loss = 0.0
    
    with torch.no_grad():  # Disable gradient computation
        # Saves memory, faster inference
        
        for inputs, labels in loader_val:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            predicted = outputs.sigmoid() > threshold
            
            running_loss += loss.item() * inputs.size(0)
            total_correct += (predicted == labels.byte()).sum().item()
            total_samples += labels.numel()
    
    val_loss = running_loss / len(loader_val.dataset)
    accuracy = total_correct / total_samples * 100
    
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return val_loss, accuracy
```

## C4.7. So sánh các ViT Versions

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        VIT MODELS COMPARISON                                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Attribute         │ ViT-v1        │ ViT-v2        │ ViT-Pretrained          ║
╠════════════════════╪═══════════════╪═══════════════╪═════════════════════════╣
║  Framework         │ TensorFlow    │ TensorFlow    │ PyTorch + timm          ║
║  Patch Size        │ 32×32         │ 32×32         │ 16×16                   ║
║  Num Patches       │ 49            │ 49            │ 196                     ║
║  Embed Dim         │ 64            │ 64            │ 768                     ║
║  Transformer Blocks│ 8             │ 8             │ 12                      ║
║  Attention Heads   │ 4             │ 4             │ 12                      ║
║  MLP Ratio         │ 2×            │ 2×            │ 4×                      ║
║  Total Params      │ ~3M           │ ~3M           │ ~86M                    ║
║  Pretrained        │ ❌ No         │ ❌ No         │ ✅ ImageNet-21k         ║
║  Regularization    │ Dropout only  │ Dropout + L2  │ Dropout                 ║
║  Optimizer         │ AdamW         │ SGD + Schedule│ Adam                    ║
║  Expected AUC      │ 0.60-0.68     │ 0.68-0.75     │ 0.82-0.88               ║
╠════════════════════╪═══════════════╪═══════════════╪═════════════════════════╣
║  Main Issues       │ ops not       │ l2 not        │ Different data path     ║
║                    │ defined       │ imported      │ than others             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

KEY INSIGHTS:

1. PATCH SIZE MATTERS:
   32×32 → 49 patches → Less tokens → Less computation
   16×16 → 196 patches → More tokens → Better detail → Better performance

2. PRETRAINED IS KEY:
   ViT needs LOTS of data to train from scratch
   ImageNet-21k pretraining provides:
   - Low-level features (edges, textures)
   - Mid-level features (shapes, patterns)
   - High-level visual concepts
   
   Only need to fine-tune for medical domain!

3. MODEL SIZE vs PERFORMANCE:
   Small ViT (3M params) without pretraining → struggles
   Large ViT (86M params) with pretraining → excellent
   
   "Pretrained > Size > Architecture"
```

---

# 📚 TÀI LIỆU THAM KHẢO

## Papers

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Giới thiệu Transformer architecture
   - Self-attention mechanism
   - arXiv:1706.03762

2. **An Image is Worth 16x16 Words** (Dosovitskiy et al., 2020)
   - Vision Transformer (ViT)
   - Patch-based image processing
   - arXiv:2010.11929

3. **Deep Residual Learning for Image Recognition** (He et al., 2015)
   - ResNet architecture
   - Skip connections
   - arXiv:1512.03385

4. **ChestX-ray8** (Wang et al., 2017)
   - NIH dataset paper
   - Multi-label classification
   - CVPR 2017

5. **CheXNet** (Rajpurkar et al., 2017)
   - Radiologist-level performance
   - DenseNet-121 baseline
   - arXiv:1711.05225

6. **Focal Loss for Dense Object Detection** (Lin et al., 2017)
   - Class imbalance solution
   - arXiv:1708.02002

7. **Adam: A Method for Stochastic Optimization** (Kingma & Ba, 2014)
   - Adam optimizer
   - arXiv:1412.6980

8. **Understanding the Difficulty of Training Deep Feedforward Neural Networks** (Glorot & Bengio, 2010)
   - Xavier initialization
   - AISTATS 2010

## Libraries & Frameworks

- **TensorFlow/Keras**: https://tensorflow.org
- **PyTorch**: https://pytorch.org
- **timm (PyTorch Image Models)**: https://github.com/huggingface/pytorch-image-models
- **scikit-learn**: https://scikit-learn.org

## Online Resources

- **ViT Paper Explained**: https://jalammar.github.io/illustrated-transformer/
- **ResNet Paper Explained**: https://towardsdatascience.com/residual-networks-resnets-cb474c7c834a
- **NIH Chest X-ray Dataset**: https://nihcc.app.box.com/v/ChestXray-NIHCC

---

# 📋 PHỤ LỤC: BUGS & FIXES

## Bug 1: data.ipynb - Random Sampling Error

```python
# BUG:
idxs = random  # Sai! random là module, không phải list

# FIX:
idxs = random.sample(idxs, num_images)
```

## Bug 2: ViT-v1.ipynb - ops Not Defined

```python
# BUG:
positions = ops.expand_dims(...)  # 'ops' không được import

# FIX Option 1: Import keras.ops
from keras import ops
positions = ops.expand_dims(...)

# FIX Option 2: Dùng tf.expand_dims
positions = tf.expand_dims(
    tf.range(start=0, limit=self.num_patches, delta=1),
    axis=0
)
```

## Bug 3: ViT-v2.ipynb - l2 Not Imported

```python
# BUG:
kernel_regularizer=l2(regularizer_rate)  # l2 không được import

# FIX:
from keras.regularizers import l2
# hoặc
from tensorflow.keras.regularizers import l2
```

## Bug 4: ViT-v2.ipynb - EarlyStopping restore_best_weights

```python
# BUG:
restore_best_weights=False  # Không khôi phục weights tốt nhất!

# FIX:
early_stopping_callback = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True  # Quan trọng!
)
```

## Bug 5: CNN - Missing Imports

```python
# BUG: Sequential, Conv2D, etc. không được import

# FIX: Add imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
```

## Bug 6: ResNet - Missing GlobalAveragePooling2D Import

```python
# BUG: GlobalAveragePooling2D không được import

# FIX: Update import
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, 
    MaxPooling2D, GlobalAveragePooling2D, Dense
)
```

---

*Tài liệu tổng hợp hoàn chỉnh cho dự án ViT-Chest-Xray*
*Bao gồm: Lý thuyết Deep Learning + Giải thích Code Chi tiết*
*AI Expert Analysis System - January 2025*
