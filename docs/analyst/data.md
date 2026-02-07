# Phân Tích Chi Tiết: Data Processing Module

## 📋 Tổng Quan

File `data.ipynb` chứa các class và hàm để **xử lý dữ liệu** cho việc training các mô hình deep learning trên NIH Chest X-ray dataset. Module này được sử dụng bởi tất cả các notebooks khác thông qua `%run data.ipynb`.

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
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
warnings.filterwarnings('ignore')
```

### Phân tích thư viện:

| Thư viện | Mục đích sử dụng |
|----------|------------------|
| `glob` | Tìm kiếm files theo pattern |
| `os` | Thao tác với file system |
| `random` | Lấy mẫu ngẫu nhiên |
| `cv2` | Đọc và xử lý ảnh |
| `pandas` | Xử lý CSV labels |
| `train_test_split` | Chia tập dữ liệu |
| `ImageDataGenerator` | Data augmentation |

### ⚠️ Nhận xét:
- `InceptionResNetV2` import nhưng **không sử dụng**
- `torch` import nhưng **không cần thiết** trong data processing

---

## 🔧 Cell 2: DatasetParser Class

```python
class DatasetParser():
    def __init__(self, root_dir, images_dir, labels_csv):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, images_dir,"*.png")))
        self.labels_df = self._labels_by_task(root_dir=root_dir, labels_csv=labels_csv)
        
        self.labels = ['Cardiomegaly','Emphysema','Effusion',
                       'Hernia','Nodule','Pneumothorax','Atelectasis',
                       'Pleural_Thickening','Mass','Edema','Consolidation',
                       'Infiltration','Fibrosis','Pneumonia', 'No Finding']
```

### Kiến trúc Class:

```
┌─────────────────────────────────────────────────────────────────┐
│                      DatasetParser                              │
├─────────────────────────────────────────────────────────────────┤
│ Attributes:                                                     │
│   - image_paths: List[str]     # Danh sách đường dẫn ảnh       │
│   - labels_df: DataFrame       # DataFrame chứa labels         │
│   - labels: List[str]          # 15 tên classes                │
├─────────────────────────────────────────────────────────────────┤
│ Methods:                                                        │
│   + __init__(root_dir, images_dir, labels_csv)                 │
│   + visualize_random_images(num_images, label, display_label)  │
│   + _labels_by_task(root_dir, labels_csv)                      │
│   + get_labels_df()                                             │
│   + sample(num_samples, is_weighted)                           │
└─────────────────────────────────────────────────────────────────┘
```

### 15 Disease Labels:

```
┌─────────────────┬─────────────────┬─────────────────┐
│   Cardiomegaly  │   Emphysema     │    Effusion     │
├─────────────────┼─────────────────┼─────────────────┤
│     Hernia      │    Nodule       │  Pneumothorax   │
├─────────────────┼─────────────────┼─────────────────┤
│   Atelectasis   │ Pleural_Thick.  │      Mass       │
├─────────────────┼─────────────────┼─────────────────┤
│     Edema       │ Consolidation   │  Infiltration   │
├─────────────────┼─────────────────┼─────────────────┤
│    Fibrosis     │   Pneumonia     │   No Finding    │
└─────────────────┴─────────────────┴─────────────────┘
```

---

## 🔧 Method: `_labels_by_task()`

```python
def _labels_by_task(self, root_dir=None, labels_csv=None):
    labels_df = pd.read_csv(os.path.join(root_dir, labels_csv))
    image_path = {os.path.basename(x): x for x in glob.glob(os.path.join(root_dir, 'images', '*.png'))}
    
    labels_df = labels_df[labels_df['Image Index'].map(os.path.basename).isin(image_path)]

    new_labels_df = pd.DataFrame()
    new_labels_df['Id'] = labels_df['Image Index'].copy()
    new_labels_df['Label'] = labels_df['Finding Labels'].apply(lambda val: val.split('|'))
    
    del labels_df
    return new_labels_df
```

### Flow xử lý:

```
┌─────────────────────────────────────────────────────────────┐
│                    CSV File (Labels)                        │
│  Image Index    │ Finding Labels                            │
│  00000001.png   │ Cardiomegaly|Emphysema                    │
│  00000002.png   │ No Finding                                │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Filter: Only existing images                    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Split labels by '|' delimiter                   │
│  "Cardiomegaly|Emphysema" → ['Cardiomegaly', 'Emphysema']   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output DataFrame                          │
│  Id             │ Label                                      │
│  00000001.png   │ ['Cardiomegaly', 'Emphysema']             │
│  00000002.png   │ ['No Finding']                            │
└─────────────────────────────────────────────────────────────┘
```

### ⚠️ Vấn đề:
- `del labels_df` không cần thiết - Python GC sẽ tự xử lý
- Không có validation cho missing images

---

## 🔧 Method: `visualize_random_images()`

```python
def visualize_random_images(self, num_images=1, label=None, display_label=False):
    fig = plt.figure(figsize=(20,20))
    fig.tight_layout(pad=10.0)
    if label is None:
        idxs = random.sample(range(len(self.image_paths)), num_images)
    else:
        idxs = [idx for idx in range(len(self.labels_df['Label'])) if label in self.labels_df['Label'][idx]]
        if len(idxs) < num_images:
            num_images = len(idxs)
        else:
            idxs = random
            (idxs, num_images)  # ⚠️ BUG!
```

### ⚠️ BUG nghiêm trọng:

```python
idxs = random
(idxs, num_images)  # Đây là 2 dòng riêng biệt, không gọi random.sample()
```

**Fix đúng:**
```python
idxs = random.sample(idxs, num_images)
```

---

## 🔧 Method: `get_labels_df()`

```python
def get_labels_df(self):
    new_labels_df = self.labels_df.copy()
    
    for i in range(len(new_labels_df)):
        one_hot = [0 for element in self.labels]
        for element in new_labels_df['Label'][i]:
            one_hot[self.labels.index(element)] = 1
        new_labels_df['Label'][i] = one_hot
            
    return new_labels_df
```

### One-Hot Encoding Process:

```
Input: ['Cardiomegaly', 'Effusion']

Labels order: [Cardiomegaly, Emphysema, Effusion, Hernia, ...]
                    ↓            ↓         ↓        ↓
Output:          [  1     ,     0    ,    1    ,   0   , ...]
                    ↑                      ↑
              Cardiomegaly              Effusion
```

### ⚠️ Vấn đề hiệu suất:
- Sử dụng `for` loop trên DataFrame → **rất chậm**
- Modify DataFrame trong loop → inefficient

### 💡 Vectorized version (tốt hơn):
```python
def get_labels_df(self):
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=self.labels)
    one_hot = mlb.fit_transform(self.labels_df['Label'])
    new_df = self.labels_df.copy()
    new_df['Label'] = list(one_hot)
    return new_df
```

---

## 🔧 Method: `sample()`

```python
def sample(self, num_samples, is_weighted=False):
    if not is_weighted:
        return self.labels_df.sample(num_samples)
    else:
        sample_weights = self.labels_df['Label'].map(lambda x: len(x)).values + 4e-2
        sample_weights /= sample_weights.sum()
        return self.labels_df.sample(num_samples, weights=sample_weights)
```

### Weighted Sampling Logic:

```
┌────────────────────────────────────────────────────────────┐
│              Weighted Sampling Strategy                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  weight(image) = num_labels(image) + 0.04                  │
│                                                            │
│  Ảnh có nhiều bệnh → weight cao hơn → được chọn nhiều hơn │
│                                                            │
│  Ví dụ:                                                    │
│  - Image A: ['No Finding']      → weight = 1 + 0.04 = 1.04│
│  - Image B: ['Cardiomegaly',    → weight = 3 + 0.04 = 3.04│
│              'Effusion',                                   │
│              'Pneumonia']                                  │
│                                                            │
│  Image B có xác suất được chọn cao hơn ~3x                 │
└────────────────────────────────────────────────────────────┘
```

### Ý nghĩa:
- `4e-2 = 0.04`: Smoothing factor để tránh weight = 0
- Ưu tiên ảnh có **multi-label** → cân bằng dữ liệu tốt hơn

---

## 🔧 Cell 3-4: Initialize Parser

```python
parser = DatasetParser(
    root_dir="/Users/ananyajain/Desktop/CSC413/CSC413-Final-Project/archive/sample",
    images_dir="sample/images",
    labels_csv="sample_labels.csv"
)
print("Total Trainable Data: ", parser.labels_df.shape[0])
```

### ⚠️ Vấn đề:
- **Hardcoded path** cho Mac OS → không chạy được trên Windows
- Nên dùng relative path hoặc environment variable

### 💡 Fix:
```python
import os
ROOT_DIR = os.environ.get('DATA_DIR', './input')
parser = DatasetParser(
    root_dir=ROOT_DIR,
    images_dir="images",
    labels_csv="Data_Entry_2017_v2020.csv"
)
```

---

## 🔧 Cell 5-6: Train/Val/Test Split

```python
df = parser.sample(100, is_weighted=True)

train_val, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.25, random_state=42)

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)
```

### Split Ratio:

```
Total: 100 samples
        │
        ├── 80% (80 samples) ──┬── 75% (60 samples) → Training
        │                      └── 25% (20 samples) → Validation
        │
        └── 20% (20 samples) → Test

Final: Train=60, Val=20, Test=20 (60/20/20 split)
```

### ⚠️ Vấn đề:
- Chỉ dùng **100 samples** → quá ít cho deep learning
- Dataset thực tế có 112,120 ảnh!

---

## 🔧 Cell 7: Data Augmentation

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, 
    vertical_flip=False, 
    height_shift_range=0.05, 
    width_shift_range=0.1, 
    rotation_range=5, 
    shear_range=0.1,
    fill_mode='reflect',
    zoom_range=0.15
)

val_datagen = ImageDataGenerator(rescale=1./255)
```

### Augmentation Techniques:

| Technique | Value | Visualization |
|-----------|-------|---------------|
| `rescale` | 1/255 | [0-255] → [0-1] |
| `horizontal_flip` | True | ↔️ Mirror |
| `vertical_flip` | False | ↕️ Disabled (vì X-ray có chiều cố định) |
| `height_shift` | 5% | ↑↓ Dịch chuyển |
| `width_shift` | 10% | ←→ Dịch chuyển |
| `rotation` | 5° | 🔄 Xoay nhẹ |
| `shear` | 0.1 | ◇ Nghiêng |
| `zoom` | 15% | 🔍 Phóng to/thu nhỏ |
| `fill_mode` | 'reflect' | Mirror padding |

### ⚠️ Considerations cho Medical Imaging:
- ✅ `vertical_flip=False`: Đúng! X-ray luôn có hướng cố định
- ⚠️ `rotation_range=5`: Có thể tăng lên 10-15°
- ⚠️ Thiếu **brightness/contrast** augmentation

### 💡 Improved Augmentation:
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],  # Thêm!
    fill_mode='reflect'
)
```

---

## 🔧 Data Generators

```python
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory='/Users/.../images',
    x_col="Id",
    y_col="Label",
    batch_size=32,
    target_size=(224, 224),
    classes=parser.labels
)
```

### Generator Output:

```
┌─────────────────────────────────────────────────────────────┐
│                   flow_from_dataframe()                     │
├─────────────────────────────────────────────────────────────┤
│  Input:                                                     │
│    - DataFrame với columns: Id, Label                       │
│    - Directory chứa images                                  │
│                                                             │
│  Output: Generator yields (images, labels) batches          │
│    - images: (batch_size, 224, 224, 3) tensor              │
│    - labels: (batch_size, 15) one-hot encoded              │
└─────────────────────────────────────────────────────────────┘
```

### ⚠️ Vấn đề:
1. **Hardcoded directory path** → không portable
2. **class_mode không được chỉ định** → có thể gây lỗi với multi-label

---

## 📊 Đánh Giá Tổng Thể

### ✅ Điểm mạnh:
1. Class structure tốt, modular
2. Weighted sampling cho imbalanced data
3. Proper train/val/test split
4. Reasonable augmentation for medical images

### ❌ Điểm yếu:

| Vấn đề | Mức độ | Giải pháp |
|--------|--------|-----------|
| BUG trong visualize_random_images | 🔴 Critical | Fix random.sample() |
| Hardcoded paths | 🔴 Critical | Use relative paths |
| Chỉ dùng 100 samples | 🟠 High | Sử dụng full dataset |
| Inefficient one-hot encoding | 🟡 Medium | Use vectorized ops |
| Missing imports check | 🟡 Medium | Add try-except |

### 💡 Improved DatasetParser:

```python
class DatasetParser:
    def __init__(self, root_dir, images_dir, labels_csv):
        self.root_dir = root_dir
        self.labels = ['Cardiomegaly', 'Emphysema', 'Effusion',
                       'Hernia', 'Nodule', 'Pneumothorax', 'Atelectasis',
                       'Pleural_Thickening', 'Mass', 'Edema', 'Consolidation',
                       'Infiltration', 'Fibrosis', 'Pneumonia', 'No Finding']
        
        # Load and validate
        self.image_paths = self._load_images(images_dir)
        self.labels_df = self._load_labels(labels_csv)
        
        print(f"Loaded {len(self.labels_df)} samples")
    
    def _load_images(self, images_dir):
        pattern = os.path.join(self.root_dir, images_dir, "*.png")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No images found at {pattern}")
        return paths
    
    def _load_labels(self, labels_csv):
        csv_path = os.path.join(self.root_dir, labels_csv)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Labels file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        # ... rest of processing
        return df
```

---

## 📚 Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. DatasetParser                                               │
│     - Load images from directory                                │
│     - Parse CSV labels                                          │
│     - Split multi-labels by '|'                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Weighted Sampling                                           │
│     - Prioritize multi-label images                             │
│     - Balance class distribution                                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Train/Val/Test Split (60/20/20)                            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. ImageDataGenerator                                          │
│     - Rescale to [0, 1]                                        │
│     - Apply augmentation (train only)                          │
│     - Resize to 224x224                                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Generators → Model Training                                 │
│     - Batch size: 32                                            │
│     - Output: (32, 224, 224, 3), (32, 15)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📚 References

1. Keras ImageDataGenerator: https://keras.io/api/data_loading/image/
2. NIH Chest X-ray Dataset Paper: Wang et al., CVPR 2017
3. Data Augmentation for Medical Imaging: A Survey
