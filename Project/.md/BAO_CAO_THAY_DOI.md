# BÁO CÁO THAY ĐỔI - DỰ ÁN CHEST X-RAY CLASSIFICATION

## Lịch sử Commit

| Commit | Ngày | Mô tả |
|--------|------|-------|
| `bad7269` | - | First commit - Khởi tạo dự án |
| `b535efd` | - | Add detailed analysis for data download and ResNet model |
| `99eefe9` | - | Refactor code structure for improved readability |
| `81af87c` | - | Clear output |
| `15e684e` | 30/01/2026 | Refactor path configs for cross-platform + add config.py |
| `7aa8947` | 31/01/2026 | **Migrate from TensorFlow to PyTorch** for Python 3.14 |
| `7e667f7` | 01/02/2026 | **Migrate all notebooks to PyTorch + Fix AUC NaN + Training results** |

---

# THAY ĐỔI 1: CHUYỂN TỪ TENSORFLOW SANG PYTORCH

## Ngày thực hiện: 31/01/2026
## Commit: `7aa8947`

## Tóm tắt
Do Python 3.14 chưa được TensorFlow hỗ trợ chính thức, chúng tôi đã thực hiện chuyển đổi toàn bộ notebook `data.ipynb` từ TensorFlow/Keras sang PyTorch để đảm bảo tính tương thích và hoạt động ổn định.

## Vấn đề gốc
- **Lỗi import**: `ModuleNotFoundError: No module named 'tensorflow'`
- **Nguyên nhân**: Python 3.14 là phiên bản quá mới, TensorFlow chưa có wheels tương thích
- **Giải pháp**: Chuyển sang PyTorch (đã được cài đặt và tương thích với Python 3.14)

## Các thay đổi chính

### 1. **Thay đổi imports**
```python
# CŨ (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

# MỚI (PyTorch)
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
```

### 2. **Sửa cấu hình đường dẫn dữ liệu**
```python
# CŨ
IMAGES_DIR = "images"  # Tìm trong thư mục con images/
ROOT_DIR = os.path.join(PROJECT_ROOT, "Project", "input")

# MỚI  
IMAGES_DIR = "."  # Tìm trực tiếp trong thư mục input/
ROOT_DIR = os.path.join(PROJECT_ROOT, "Project", "input")
```

**Lý do**: Hình ảnh được lưu trực tiếp trong thư mục `input/` chứ không phải `input/images/`

### 3. **Cải tiến lớp DatasetParser**
```python
# Thêm tham số images_dir vào _labels_by_task()
def _labels_by_task(self, root_dir=None, labels_csv=None, images_dir="."):
    # Logic xử lý linh hoạt cho cả thư mục con và thư mục gốc
    if images_dir == ".":
        image_paths = glob.glob(os.path.join(root_dir, '*.png'))
    else:
        image_paths = glob.glob(os.path.join(root_dir, images_dir, '*.png'))
```

**Sửa lỗi**: Trong method `visualize_random_images()` - thay `idxs = random(idxs, num_images)` thành `idxs = random.sample(idxs, num_images)`

### 4. **Tạo lớp Dataset cho PyTorch**
```python
class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, images_path, labels, transform=None, is_training=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.images_path = images_path
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Load image using OpenCV and PIL
        # Convert labels to one-hot tensor
        # Apply transforms
        return image, label
```

### 5. **Thay thế ImageDataGenerator bằng PyTorch transforms**
```python
# CŨ (Keras)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=5,
    # ... other augmentations
)

# MỚI (PyTorch)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 6. **Thay thế data generators bằng DataLoaders**
```python
# CŨ (Keras)
train_generator = train_datagen.flow_from_dataframe(...)
validation_generator = val_datagen.flow_from_dataframe(...)

# MỚI (PyTorch)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
```

## Kết quả sau thay đổi

### ✅ **Thành công**
- **Tổng số hình ảnh phát hiện**: 112,120 images
- **Kích thước dataset**: 
  - Training: 60 samples (2 batches)
  - Validation: 20 samples (1 batch) 
  - Test: 20 samples (1 batch)
- **Tất cả imports hoạt động**: Không còn lỗi ModuleNotFoundError
- **Data loading thành công**: CSV và images được load đúng cách
- **PyTorch datasets và dataloaders**: Tạo thành công và sẵn sàng training

### 📊 **Thống kê dữ liệu**
```
Data root: d:\MSE\10.Deep Learning\Group_Final\ViT-Chest-Xray\Project\input
Total Trainable Data: 112120
Training set size: 60
Validation set size: 20
Test set size: 20
Images path: d:\MSE\10.Deep Learning\Group_Final\ViT-Chest-Xray\Project\input
Training batches: 2
Validation batches: 1
Test batches: 1
```

## Lợi ích của việc chuyển sang PyTorch

1. **Tương thích**: Hoạt động hoàn hảo với Python 3.14
2. **Hiệu suất**: PyTorch có hiệu suất tốt cho Vision Transformers
3. **Linh hoạt**: Dễ dàng custom dataset và transforms
4. **Cộng đồng**: Nhiều pre-trained ViT models có sẵn trên PyTorch
5. **Debugging**: PyTorch có dynamic computation graph, dễ debug hơn

## Những file bị ảnh hưởng

- `data.ipynb`: Thay đổi toàn bộ logic data loading
- Không có file nào khác bị ảnh hưởng

## Bước tiếp theo

Notebook hiện tại đã sẵn sàng để:
1. Xây dựng Vision Transformer models với PyTorch
2. Training với chest X-ray dataset  
3. Sử dụng pre-trained ViT models từ `timm` library
4. Implement các kiến trúc deep learning khác với PyTorch

## Ghi chú kỹ thuật

- **Environment**: Python 3.14.0 với PyTorch 2.10.0
- **Dataset**: NIH Chest X-ray Dataset với 15 nhãn bệnh
- **Image format**: PNG files, được resize về 224x224 cho training
- **Label encoding**: One-hot encoding cho multi-label classification
- **Data augmentation**: Horizontal flip, rotation, color jitter
- **Batch size**: 32 (có thể điều chỉnh theo GPU memory)

## Files thay đổi trong commit `7aa8947`

| File | Thêm | Xóa | Mô tả |
|------|------|-----|-------|
| `Project/data.ipynb` | +259 | -138 | Chuyển từ Keras sang PyTorch DataLoader |
| `Project/cnn.ipynb` | +225 | - | Chuyển model từ Keras sang PyTorch |
| `Project/BAO_CAO_THAY_DOI.md` | +167 | - | Tài liệu ghi chép thay đổi |
| `Project/comprehensive_analysis.py` | +839 | - | Script phân tích project |

---

# THAY ĐỔI 2: FIX AUC NaN TRONG CNN TRAINING

## Ngày thực hiện: 01/02/2026
## Commit: *(chưa commit)*

### Vấn đề
Khi chạy training với `cnn.ipynb`, giá trị AUC hiển thị là `nan` thay vì giá trị số hợp lệ.

### Nguyên nhân
- `sklearn.metrics.roc_auc_score` với `average='macro'` yêu cầu mỗi class phải có **ít nhất 1 mẫu positive (1)** và **ít nhất 1 mẫu negative (0)**
- Với dữ liệu nhỏ hoặc mất cân bằng (multi-label classification với 15 classes), một số class có thể chỉ có toàn 0 hoặc toàn 1 trong một epoch
- Điều này khác với Keras `keras.metrics.AUC()` - tự động xử lý các edge cases

### So sánh cách tính AUC

| Keras (Original) | PyTorch (Current) |
|------------------|-------------------|
| `keras.metrics.AUC()` tích hợp | `roc_auc_score()` từ sklearn |
| Tự động xử lý edge cases | Cần xử lý thủ công |
| Trả về 0 nếu không đủ data | Trả về NaN/Error |

### Giải pháp áp dụng

```python
# CŨ - Gây lỗi NaN
try:
    epoch_auc = roc_auc_score(all_targets, all_outputs, average='macro')
except ValueError:
    epoch_auc = 0.0

# MỚI - Chỉ tính AUC cho các class hợp lệ
try:
    # Tìm các class có cả positive và negative samples
    valid_classes = []
    for i in range(all_targets.shape[1]):
        if len(np.unique(all_targets[:, i])) > 1:
            valid_classes.append(i)
    
    if len(valid_classes) > 0:
        epoch_auc = roc_auc_score(
            all_targets[:, valid_classes], 
            all_outputs[:, valid_classes], 
            average='macro'
        )
    else:
        epoch_auc = 0.0
except ValueError:
    epoch_auc = 0.0
```

### Logic xử lý
1. Duyệt qua từng class (15 classes)
2. Kiểm tra xem class đó có cả giá trị 0 và 1 không (`np.unique()`)
3. Chỉ đưa các class hợp lệ vào tính AUC
4. Nếu không có class nào hợp lệ, trả về AUC = 0.0

### File bị ảnh hưởng
- `cnn.ipynb`: Cập nhật hàm `train_model()` - phần tính AUC cho cả training và validation

### Kết quả
- ✅ AUC không còn hiển thị NaN
- ✅ Tính toán AUC chính xác cho các class có đủ dữ liệu
- ✅ Tương thích với dữ liệu mất cân bằng

---

# THAY ĐỔI 3: CHUYỂN RESNET.IPYNB SANG PYTORCH

## Ngày thực hiện: 01/02/2026
## Commit: *(chưa commit)*

### Vấn đề
- `resnet.ipynb` vẫn sử dụng TensorFlow/Keras (`train_generator`, `validation_generator`)
- Nhưng `data.ipynb` đã được chuyển sang PyTorch (`train_loader`, `val_loader`)
- Gây ra lỗi: `NameError: name 'train_generator' is not defined`

### Giải pháp
Chuyển toàn bộ `resnet.ipynb` sang PyTorch để đồng nhất với project.

### Các thay đổi chi tiết

#### 1. **Imports** (Cell 2)
```python
# CŨ (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ...
from tensorflow.keras.models import Model

# MỚI (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm
```

#### 2. **Device Configuration** (Cell 4)
```python
# CŨ
gpus = tf.config.list_physical_devices('GPU')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    if gpus:
        device = '/GPU:0'
    else:
        device = '/CPU:0'

# MỚI  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

#### 3. **Model Architecture** (Cell 6)
```python
# CŨ (Keras Functional API)
def block(x, filters, strides=1):
    x = Conv2D(filters, 3, ...)(x)
    x = BatchNormalization()(x)
    ...

def create_resnet():
    inputs = Input(shape=input_shape)
    ...
    model = Model(inputs, outputs)
    return model

# MỚI (PyTorch nn.Module)
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(...)
        self.bn1 = nn.BatchNorm2d(...)
        ...
    
    def forward(self, x):
        ...
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=15):
        ...
    
    def forward(self, x):
        ...
        return x

def create_resnet34(num_classes=15):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
```

#### 4. **Training Function** (Cell 7)
```python
# CŨ (Keras)
def run_experiment(model):
    optimizer = keras.optimizers.AdamW(...)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', ...)
    history = model.fit(train_generator, validation_data=validation_generator, ...)
    return history

# MỚI (PyTorch) - Với AUC fix
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    # Training loop với tqdm progress bar
    # AUC calculation với valid_classes check (fix NaN)
    # Model checkpoint saving
    return history
```

#### 5. **Training Execution** (Cell 8)
```python
# CŨ
resnet = create_resnet()
history_resnet = run_experiment(resnet)

# MỚI
model = create_resnet34(num_classes=num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)
```

#### 6. **Plot Function** (Cell 9)
```python
# CŨ (Keras history object)
plt.plot(history.history['loss'], ...)
plt.plot(history.history['val_loss'], ...)

# MỚI (Python dictionary)
plt.plot(history['train_loss'], ...)
plt.plot(history['val_loss'], ...)
plt.plot(history['train_auc'], ...)
plt.plot(history['val_auc'], ...)
```

### So sánh kiến trúc ResNet

| Aspect | Keras (Cũ) | PyTorch (Mới) |
|--------|------------|---------------|
| Model type | Functional API | nn.Module class |
| Block definition | Function | BasicBlock class |
| Residual connection | `x += identity` | `out += identity` |
| Pooling | GlobalAveragePooling2D | AdaptiveAvgPool2d |
| Classifier | Dense layer | Linear layer |
| Weight init | Default | Kaiming normal |

### File bị ảnh hưởng
- `resnet.ipynb`: Chuyển toàn bộ từ TensorFlow/Keras sang PyTorch

### Kết quả
- ✅ Tương thích với `data.ipynb` (sử dụng `train_loader`, `val_loader`)
- ✅ Sử dụng GPU với PyTorch CUDA
- ✅ AUC fix đã được áp dụng (không còn NaN)
- ✅ ResNet-34 architecture với proper weight initialization

---

# THAY ĐỔI 4: CHUYỂN ViT-v1.ipynb SANG PYTORCH

## Ngày thực hiện: 01/02/2026
## Commit: *(chưa commit)*

### Vấn đề
- `ViT-v1.ipynb` sử dụng TensorFlow/Keras
- Cần chuyển sang PyTorch để tương thích với `data.ipynb` và môi trường Python 3.13

### Các thay đổi chi tiết

#### 1. **Imports**
```python
# CŨ (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# MỚI (PyTorch)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm
```

#### 2. **MLP Block**
```python
# CŨ (Keras)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

# MỚI (PyTorch)
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
```

#### 3. **Patch Embedding**
```python
# CŨ (Keras custom layer)
class Patches(layers.Layer):
    def call(self, images):
        patches = tf.image.extract_patches(...)
        return patches

# MỚI (PyTorch nn.Module)
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=32, in_channels=3, embed_dim=64):
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
```

#### 4. **Transformer Encoder Block**
```python
# CŨ (Keras layers)
x1 = LayerNormalization()(encoded_patches)
attention_output = MultiHeadAttention(num_heads, key_dim)(x1, x1)
x2 = Add()([attention_output, encoded_patches])

# MỚI (PyTorch nn.Module)
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)
```

#### 5. **Vision Transformer Model**
```python
# MỚI - VisionTransformer class với:
# - PatchEmbedding layer
# - Learnable positional embedding
# - Transformer encoder blocks
# - Classification head với MLP
# - Proper weight initialization (trunc_normal_)
```

#### 6. **Training Function**
- Sử dụng PyTorch training loop với tqdm progress bar
- AUC calculation với valid_classes check (fix NaN)
- Model checkpoint saving

### File bị ảnh hưởng
- `ViT-v1.ipynb`: Chuyển toàn bộ từ TensorFlow/Keras sang PyTorch

### Kết quả
- ✅ Vision Transformer from scratch với PyTorch
- ✅ Tương thích với `data.ipynb` DataLoaders
- ✅ AUC fix đã được áp dụng
- ✅ ROC curve plotting

---

# THAY ĐỔI 5: CHUYỂN ViT-v2.ipynb SANG PYTORCH

## Ngày thực hiện: 01/02/2026
## Commit: *(chưa commit)*

### Vấn đề
- `ViT-v2.ipynb` là phiên bản cải tiến của ViT-v1 với regularization
- Cần chuyển sang PyTorch tương tự ViT-v1

### Các thay đổi chi tiết
Tương tự ViT-v1, với các bổ sung:

#### 1. **Early Stopping**
```python
# CŨ (Keras callback)
early_stopping = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=False)

# MỚI (PyTorch manual implementation)
patience_counter = 0
if epoch_val_loss < best_val_loss:
    patience_counter = 0
    torch.save(model.state_dict(), save_path)
else:
    patience_counter += 1
    if patience_counter >= patience:
        print('Early stopping triggered')
        break
```

#### 2. **Learning Rate Scheduler**
```python
# CŨ (Keras callback)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

# MỚI (PyTorch)
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
scheduler.step(epoch_val_loss)
```

#### 3. **Multiple Optimizer Options**
```python
def get_optimizer(model, optimizer_name='sgd'):
    optimizers = {
        "adam": optim.Adam(model.parameters(), lr=learning_rate),
        "adamw": optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        "sgd": optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True),
        ...
    }
    return optimizers.get(optimizer_name)
```

### File bị ảnh hưởng
- `ViT-v2.ipynb`: Chuyển toàn bộ từ TensorFlow/Keras sang PyTorch

### Kết quả
- ✅ ViT-v2 với early stopping và LR scheduler
- ✅ Multiple optimizer options
- ✅ Tracking learning rate trong history
- ✅ AUC fix đã được áp dụng

---

# THAY ĐỔI 6: CẬP NHẬT ViT-ResNet.ipynb

## Ngày thực hiện: 01/02/2026
## Commit: *(chưa commit)*

### Vấn đề
- `ViT-ResNet.ipynb` đã dùng PyTorch nhưng có data loading riêng
- Cần cập nhật để sử dụng `data.ipynb` giống các notebook khác

### Các thay đổi chi tiết

#### 1. **Xóa data loading code**
- Xóa `ChestXRayDataset` class (đã có trong data.ipynb)
- Xóa path configuration
- Xóa data split và DataLoader creation

#### 2. **Sử dụng data.ipynb**
```python
# MỚI
%run data.ipynb
# Sử dụng train_loader, val_loader, test_loader từ data.ipynb
```

#### 3. **Thêm AUC fix**
- Áp dụng valid_classes check trong training function
- Tương tự các notebook khác

### File bị ảnh hưởng
- `ViT-ResNet.ipynb`: Cập nhật data loading và AUC calculation

### Kết quả
- ✅ Sử dụng chung data.ipynb với các notebook khác
- ✅ Pre-trained ViT (vit_base_patch16_224) từ timm
- ✅ AUC fix đã được áp dụng

---

## TỔNG KẾT THAY ĐỔI

| Notebook | Trạng thái ban đầu | Thay đổi | Trạng thái cuối |
|----------|-------------------|----------|-----------------|
| `data.ipynb` | TensorFlow | → PyTorch | ✅ PyTorch DataLoaders |
| `cnn.ipynb` | TensorFlow | → PyTorch + AUC fix | ✅ PyTorch CNN |
| `resnet.ipynb` | TensorFlow | → PyTorch + AUC fix | ✅ PyTorch ResNet-34 |
| `ViT-v1.ipynb` | TensorFlow | → PyTorch + AUC fix | ✅ PyTorch ViT from scratch |
| `ViT-v2.ipynb` | TensorFlow | → PyTorch + AUC fix | ✅ PyTorch ViT v2 (w/ early stopping) |
| `ViT-ResNet.ipynb` | PyTorch (độc lập) | → Dùng data.ipynb + AUC fix | ✅ Pre-trained ViT (timm) |

## Cách chạy notebooks

1. **Chuyển kernel sang Python 3.13 (.venv313)**
2. **Chạy theo thứ tự**:
   - `data.ipynb` (load data)
   - Sau đó chạy bất kỳ model notebook nào

## Lưu ý quan trọng
- **Không dùng Python 3.14** với TensorFlow (không tương thích)
- **Dùng .venv313** (Python 3.13.7) với PyTorch CUDA
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU

---

# THAY ĐỔI 7: FIX ReduceLROnPlateau VERBOSE ERROR

## Ngày thực hiện: 01/02/2026
## Commit: *(chưa commit)*

### Vấn đề
Khi chạy ViT-v2.ipynb, gặp lỗi:
```
TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'
```

### Nguyên nhân
- PyTorch 2.10+ đã loại bỏ tham số `verbose` trong `ReduceLROnPlateau`
- Code cũ sử dụng `verbose=True` không còn tương thích

### Giải pháp
```python
# CŨ (gây lỗi)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6, verbose=True)

# MỚI (đã fix)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
```

### File bị ảnh hưởng
- `ViT-v2.ipynb`: Cell tạo scheduler

### Kết quả
- ✅ ViT-v2.ipynb chạy thành công
- ✅ Training hoàn tất với early stopping

---

## KẾT QUẢ TRAINING (01/02/2026)

| Model | Parameters | Test Accuracy | Test AUC | Best |
|-------|------------|---------------|----------|------|
| **ViT-v1** | 9M | **91.33%** 🏆 | 0.5854 | Accuracy |
| **ViT-v2** | 9M | 89.67% | 0.6303 | - |
| **Pre-trained ViT** | 86M | 87.00% | **0.6694** 🏆 | AUC |

### Ghi chú
- Dataset nhỏ (60 training samples) nên kết quả chưa đáng tin cậy
- Cần tăng dataset để đánh giá thực tế

---
**Thời gian hoàn thành**: ~30 phút (data migration) + 10 phút (AUC fix) + 15 phút (ResNet) + 20 phút (ViT-v1) + 15 phút (ViT-v2) + 10 phút (ViT-ResNet) + 5 phút (fix verbose)  
**Status**: ✅ HOÀN THÀNH