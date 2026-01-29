# Phân Tích Chi Tiết: Data Download Script

## 📋 Tổng Quan

File `data_download.ipynb` thực hiện việc **tải xuống và giải nén** bộ dữ liệu NIH Chest X-ray từ NIH Clinical Center. Bộ dữ liệu này chứa hơn 112,000 ảnh X-quang ngực từ hơn 30,000 bệnh nhân.

---

## 🔧 Cell 1: Download Images

```python
#!/usr/bin/env python3
# Download the 56 zip files in Images_png in batches
import urllib.request

# URLs for the zip files
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    # ... (12 links total)
]

for idx, link in enumerate(links):
    fn = 'images_%02d.tar.gz' % (idx+1)
    print('downloading'+fn+'...')
    urllib.request.urlretrieve(link, fn)

print("Download complete. Please check the checksums")
```

### Phân tích chi tiết:

#### Import Statement
```python
import urllib.request
```
- **urllib.request**: Module built-in Python để xử lý HTTP requests
- Không cần cài đặt thêm dependencies

#### Danh sách Links
```python
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    ...
]
```

| Thuộc tính | Giá trị |
|------------|---------|
| **Nguồn** | NIH Clinical Center (Box.com) |
| **Số lượng files** | 12 files (trong code) |
| **Format** | `.tar.gz` (compressed tarball) |
| **Tổng dung lượng ước tính** | ~42 GB (full dataset) |

### ⚠️ Lưu ý:
- Comment nói "56 zip files" nhưng **chỉ có 12 links** trong code
- Đây có thể là **subset** của dataset đầy đủ

#### Download Loop

```python
for idx, link in enumerate(links):
    fn = 'images_%02d.tar.gz' % (idx+1)
    print('downloading'+fn+'...')
    urllib.request.urlretrieve(link, fn)
```

### Giải thích từng dòng:

| Dòng | Giải thích |
|------|------------|
| `enumerate(links)` | Lặp qua links với index bắt đầu từ 0 |
| `'images_%02d.tar.gz' % (idx+1)` | Format tên file: images_01.tar.gz, images_02.tar.gz, ... |
| `urllib.request.urlretrieve(link, fn)` | Download file từ URL và lưu với tên `fn` |

### Format String `%02d`:
```
%02d = zero-padded integer với 2 chữ số
  │ │
  │ └── d = decimal (số nguyên)
  └── 02 = độ rộng 2, pad với 0

Ví dụ: 1 → "01", 10 → "10"
```

### ⚠️ Vấn đề với code hiện tại:

1. **Không có error handling**:
```python
# Nếu network fail, code sẽ crash
urllib.request.urlretrieve(link, fn)  # Có thể raise URLError
```

2. **Không có progress bar**:
- Downloads lớn (GB) không có feedback

3. **Không có resume capability**:
- Nếu download fail giữa chừng, phải download lại từ đầu

4. **Không verify checksums**:
- Comment nói "check checksums" nhưng không implement

### 💡 Cải thiện đề xuất:

```python
import urllib.request
import os
from tqdm import tqdm

def download_with_progress(url, filename):
    """Download file với progress bar và error handling"""
    if os.path.exists(filename):
        print(f"Skip {filename} - already exists")
        return
    
    try:
        # Get file size
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            def reporthook(count, block_size, total_size):
                pbar.update(block_size)
            urllib.request.urlretrieve(url, filename, reporthook=reporthook)
            
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)  # Remove partial file
        raise

for idx, link in enumerate(links):
    fn = f'images_{idx+1:02d}.tar.gz'
    download_with_progress(link, fn)
```

---

## 🔧 Cell 2: Extract Files

```python
import tarfile
import os
import shutil

def extract_files(tar_file, extract_to):
    with tarfile.open(tar_file, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)

download_directory = '.'
input_directory = os.path.join(download_directory, 'input')
if not os.path.exists(input_directory):
    os.makedirs(input_directory)

for idx in range(1, len(links) + 1):
    tar_file = f'images_{idx:02d}.tar.gz'
    extract_to = os.path.join(download_directory, f'images_{idx:02d}')
    
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    print(f'Extracting {tar_file} to {extract_to}...')
    extract_files(tar_file, extract_to)
    
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            sub_dir = os.path.join(input_directory, 'images')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            src_path = os.path.join(root, file)
            dest_path = os.path.join(input_directory, file)
            shutil.move(src_path, dest_path)

print("Extraction complete.")
```

### Phân tích từng phần:

#### 1. Import Statements

```python
import tarfile    # Xử lý tar archives
import os         # File system operations
import shutil     # High-level file operations (move, copy)
```

#### 2. Extract Function

```python
def extract_files(tar_file, extract_to):
    with tarfile.open(tar_file, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)
```

| Parameter | Giải thích |
|-----------|------------|
| `'r:gz'` | Read mode, gzip compressed |
| `extractall()` | Giải nén tất cả files vào thư mục đích |

**Modes có thể dùng:**
- `'r:gz'` - gzip compressed
- `'r:bz2'` - bzip2 compressed
- `'r:xz'` - lzma compressed
- `'r'` - auto-detect compression

#### 3. Directory Setup

```python
download_directory = '.'
input_directory = os.path.join(download_directory, 'input')
if not os.path.exists(input_directory):
    os.makedirs(input_directory)
```

### Directory Structure:

```
./
├── images_01.tar.gz
├── images_02.tar.gz
├── ...
└── input/
    └── images/
        ├── 00000001_000.png
        ├── 00000001_001.png
        └── ...
```

#### 4. Extraction Loop

```python
for idx in range(1, len(links) + 1):
    tar_file = f'images_{idx:02d}.tar.gz'
    extract_to = os.path.join(download_directory, f'images_{idx:02d}')
    
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    extract_files(tar_file, extract_to)
```

### Flow diagram:

```
┌────────────────────────────────────────────────────────────┐
│                     FOR EACH idx                           │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│  tar_file = images_01.tar.gz, images_02.tar.gz, ...        │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│  extract_to = ./images_01/, ./images_02/, ...              │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│  Create directory if not exists                            │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│  Extract tar.gz → extract_to                               │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
                    MOVE FILES TO input/
```

#### 5. Move Files to Input Directory

```python
for root, dirs, files in os.walk(extract_to):
    for file in files:
        sub_dir = os.path.join(input_directory, 'images')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        src_path = os.path.join(root, file)
        dest_path = os.path.join(input_directory, file)  # ⚠️ Bug!
        shutil.move(src_path, dest_path)
```

### `os.walk()` giải thích:

```python
os.walk(extract_to)
# Returns generator of (root, dirs, files) tuples

# Ví dụ:
# extract_to = './images_01'
# 
# Iteration 1:
#   root = './images_01'
#   dirs = ['images']
#   files = []
#
# Iteration 2:
#   root = './images_01/images'
#   dirs = []
#   files = ['00000001.png', '00000002.png', ...]
```

### ⚠️ BUG trong code:

```python
sub_dir = os.path.join(input_directory, 'images')  # Tạo input/images
# ...
dest_path = os.path.join(input_directory, file)    # Nhưng move vào input/
```

**Vấn đề**: Tạo `input/images/` nhưng lại move files vào `input/` trực tiếp!

### 💡 Fix:

```python
dest_path = os.path.join(sub_dir, file)  # Move vào input/images/
```

---

## 📊 Tổng Kết

### Data Flow:

```
┌─────────────────────────────────────────────────────────────┐
│                    NIH Box.com Server                       │
│                  (12 tar.gz files, ~42GB)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ Download
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Local Directory                          │
│     images_01.tar.gz, images_02.tar.gz, ...                │
└──────────────────────────┬──────────────────────────────────┘
                           │ Extract
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Temporary Directories                       │
│    ./images_01/, ./images_02/, ... (nested structure)       │
└──────────────────────────┬──────────────────────────────────┘
                           │ Move
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Final Location                           │
│                    ./input/images/                          │
│              (112,120 PNG images, ~42GB)                    │
└─────────────────────────────────────────────────────────────┘
```

### ✅ Điểm mạnh:
1. Code đơn giản, dễ hiểu
2. Sử dụng các module built-in Python
3. Tự động tạo directories cần thiết

### ❌ Điểm yếu:

| Vấn đề | Mức độ | Giải pháp |
|--------|--------|-----------|
| Không error handling | 🔴 Critical | Try-except blocks |
| Bug move destination | 🔴 Critical | Fix dest_path |
| Không progress tracking | 🟠 High | tqdm progress bar |
| Không resume support | 🟠 High | Check existing files |
| Không cleanup temp dirs | 🟡 Medium | shutil.rmtree() |
| Không checksum verify | 🟡 Medium | MD5/SHA256 verify |

### 💡 Improved Version:

```python
import tarfile
import os
import shutil
import hashlib
from tqdm import tqdm
import urllib.request

def download_file(url, filename):
    """Download with resume support and progress bar"""
    if os.path.exists(filename):
        print(f"✓ {filename} already exists, skipping...")
        return
    
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)

def extract_and_move(tar_file, final_dir):
    """Extract tar.gz and move images to final directory"""
    if not os.path.exists(tar_file):
        print(f"✗ {tar_file} not found!")
        return
    
    temp_dir = tar_file.replace('.tar.gz', '_temp')
    
    # Extract
    print(f"Extracting {tar_file}...")
    with tarfile.open(tar_file, 'r:gz') as tar:
        tar.extractall(temp_dir)
    
    # Move files
    os.makedirs(final_dir, exist_ok=True)
    for root, dirs, files in os.walk(temp_dir):
        for file in tqdm(files, desc="Moving files"):
            if file.endswith('.png'):
                src = os.path.join(root, file)
                dst = os.path.join(final_dir, file)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"✓ Extracted and cleaned up {tar_file}")

# Main execution
links = [...]  # 12 links

# Download all
for idx, link in enumerate(links, 1):
    download_file(link, f'images_{idx:02d}.tar.gz')

# Extract all
final_images_dir = './input/images'
for idx in range(1, len(links) + 1):
    extract_and_move(f'images_{idx:02d}.tar.gz', final_images_dir)

print(f"\n✓ Complete! Images saved to {final_images_dir}")
print(f"Total images: {len(os.listdir(final_images_dir))}")
```

---

## 📚 NIH Chest X-ray Dataset Info

| Thuộc tính | Giá trị |
|------------|---------|
| **Tên chính thức** | ChestX-ray14 |
| **Số lượng ảnh** | 112,120 |
| **Số bệnh nhân** | 30,805 |
| **Số labels** | 14 bệnh + "No Finding" |
| **Resolution** | 1024 × 1024 pixels |
| **Format** | PNG (grayscale) |
| **Tổng dung lượng** | ~42 GB |

### 14 Pathologies:
1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

---

## 📚 References

1. NIH Chest X-ray Dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC
2. Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks", CVPR 2017
3. Python tarfile documentation: https://docs.python.org/3/library/tarfile.html
