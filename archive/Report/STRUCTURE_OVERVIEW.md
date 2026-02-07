# 📊 CẤU TRÚC DỰ ÁN ViT-Chest-Xray - TỔNG QUAN

## 🎯 Mục tiêu
Tái cấu trúc thư mục LaTeX để quản lý hiệu quả hơn, tách biệt nội dung theo module và ngôn ngữ.

---

## 📁 CẤU TRÚC MỚI (ĐỀ XUẤT)

```
Report/
│
├── 📄 main_vn.tex                 ⭐ FILE CHÍNH MỚI (Tiếng Việt)
├── 📄 Group1_Deeplearning.tex     📘 File gốc (Tiếng Anh - giữ nguyên)
├── 📄 README.md                   📖 Hướng dẫn sử dụng
│
├── 📂 chapters/                   🆕 Thư mục chapters mới
│   ├── 📂 models/                 ⭐ Tài liệu từng mô hình (modular)
│   │   ├── cnn.tex               (95M params, AUC 0.60)
│   │   ├── resnet.tex            (21M params, AUC 0.53)
│   │   ├── vit_scratch.tex       (9M params, AUC 0.58-0.63)
│   │   └── vit_pretrained.tex    (86M params, AUC 0.67) ✅ BEST
│   │
│   ├── 📂 figures/                Hình ảnh, biểu đồ
│   └── 📂 tables/                 Bảng số liệu
│
├── 📂 backup/                     🗄️ File cũ đã backup
│   ├── BaoCao_ChestXray_Classification.tex
│   ├── Critical_Analysis_Report.tex
│   ├── Critical_Analysis_Report_Extended.tex
│   └── latex.tex
│
└── 📂 LaTeX/                      📦 Thư mục gốc (giữ nguyên)
    └── LaTeX_EN/
```

---

## 📊 SO SÁNH CẤU TRÚC

| Khía cạnh | Cũ ❌ | Mới ✅ |
|-----------|------|-------|
| **Số file LaTeX rời rạc** | 5+ files trộn lẫn | 1 file chính + modules |
| **Tổ chức nội dung** | Tất cả trong 1 file lớn | Tách theo từng mô hình |
| **Backup file cũ** | Không | Có (thư mục backup/) |
| **Dễ bảo trì** | Khó (file lớn) | Dễ (file nhỏ, riêng biệt) |
| **Tái sử dụng** | Không | Có (include vào nhiều báo cáo) |
| **Hỗ trợ đa ngôn ngữ** | Không rõ ràng | Rõ ràng (main_vn, main_en) |

---

## 🎨 KIẾN TRÚC MODULE

### 1. File main_vn.tex (File tổng)
```latex
\documentclass{article}
% ... preamble ...
\begin{document}

% Tự động include các module
\input{chapters/models/cnn}
\input{chapters/models/resnet}
\input{chapters/models/vit_scratch}
\input{chapters/models/vit_pretrained}

\end{document}
```

### 2. Các module model (Độc lập)
Mỗi file chứa:
- ✅ Kiến trúc chi tiết
- ✅ Cấu hình huấn luyện
- ✅ Kết quả thực nghiệm
- ✅ Code minh họa
- ✅ Bảng biểu, số liệu

---

## 📝 HƯỚNG DẪN SỬ DỤNG

### ✏️ Biên dịch LaTeX

**Báo cáo Tiếng Việt:**
```bash
cd Report/
xelatex main_vn.tex
xelatex main_vn.tex  # Lần 2 để update references
```

**Báo cáo Tiếng Anh:**
```bash
xelatex Group1_Deeplearning.tex
xelatex Group1_Deeplearning.tex
```

### ➕ Thêm mô hình mới

1. Tạo file `chapters/models/new_model.tex`
2. Thêm vào `main_vn.tex`:
   ```latex
   \input{chapters/models/new_model}
   ```

### 🔧 Chỉnh sửa mô hình cụ thể

Mở file tương ứng trong `chapters/models/`:
- CNN → `cnn.tex`
- ResNet → `resnet.tex`
- ViT scratch → `vit_scratch.tex`
- ViT pretrained → `vit_pretrained.tex`

---

## 📈 KẾT QUẢ CÁC MÔ HÌNH

| Mô hình | File | Tham số | Val AUC | Test AUC | Xếp hạng |
|---------|------|---------|---------|----------|----------|
| CNN Baseline | cnn.tex | 95M | 0.5998 | ~0.58 | 4 |
| ResNet-34 | resnet.tex | 21M | 0.5293 | ~0.53 | 5 |
| ViT-v1 | vit_scratch.tex | 9M | 0.6431 | 0.5854 | 3 |
| ViT-v2 | vit_scratch.tex | 9M | 0.5947 | 0.6303 | 2 |
| **ViT Pretrained** ⭐ | vit_pretrained.tex | 86M | **0.6836** | **0.6694** | **1** |

---

## ✨ ƯU ĐIỂM CẤU TRÚC MỚI

1. **🎯 Modular**: Mỗi mô hình = 1 file → dễ tìm, dễ sửa
2. **♻️ Tái sử dụng**: Include vào nhiều báo cáo khác nhau
3. **📚 Rõ ràng**: Phân chia logic theo chức năng
4. **💾 An toàn**: File cũ được backup đầy đủ
5. **🚀 Mở rộng**: Thêm mô hình mới dễ dàng
6. **🌍 Đa ngôn ngữ**: Tách biệt tiếng Việt/Anh
7. **📖 Tài liệu**: README.md hướng dẫn chi tiết

---

## 🔍 FILE MAPPING

| File cũ (backup/) | File mới | Ghi chú |
|-------------------|----------|---------|
| model_documentation_vn.tex | chapters/models/*.tex | Đã tách thành 4 files |
| BaoCao_ChestXray_Classification.tex | backup/ | Lưu trữ |
| Critical_Analysis_Report.tex | backup/ | Lưu trữ |
| latex.tex | backup/ | Lưu trữ |
| - | main_vn.tex | **MỚI - File chính** |
| Group1_Deeplearning.tex | Group1_Deeplearning.tex | Giữ nguyên |

---

## 🎓 KHUYẾN NGHỊ

### ✅ NÊN
- Sử dụng `main_vn.tex` cho báo cáo chính
- Chỉnh sửa trong `chapters/models/` khi update mô hình
- Giữ file backup để tham khảo nếu cần
- Compile bằng **XeLaTeX** (hỗ trợ tiếng Việt)

### ❌ KHÔNG NÊN
- Chỉnh sửa trực tiếp file trong `backup/`
- Xóa file backup trước khi kiểm tra kỹ
- Dùng PDFLaTeX (không hỗ trợ UTF-8 tốt)

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề:
1. Đọc [README.md](README.md)
2. Kiểm tra log file (*.log)
3. Đảm bảo XeLaTeX đã cài đặt
4. Kiểm tra đường dẫn `\input{}` trong main file

---

**Ngày cập nhật:** 2026-02-04  
**Version:** 2.0 (Restructured)
