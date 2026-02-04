# Cấu trúc thư mục LaTeX Report

Thư mục này chứa tài liệu LaTeX cho đồ án Deep Learning về phân loại bệnh từ X-quang ngực.

## Cấu trúc thư mục

```
Report/
├── main_vn.tex                    # File LaTeX chính (Tiếng Việt)
├── Group1_Deeplearning.tex        # File báo cáo gốc (Tiếng Anh)
│
├── chapters/                       # Thư mục chứa các chương
│   ├── models/                    # Tài liệu chi tiết từng mô hình
│   │   ├── cnn.tex               # Mô hình CNN baseline
│   │   ├── resnet.tex            # Mô hình ResNet-34
│   │   ├── vit_scratch.tex       # ViT huấn luyện từ đầu (v1 & v2)
│   │   └── vit_pretrained.tex    # ViT pretrained (timm)
│   │
│   ├── figures/                   # Thư mục hình ảnh
│   └── tables/                    # Thư mục bảng biểu
│
├── backup/                        # Các file LaTeX cũ đã backup
│   ├── BaoCao_ChestXray_Classification.tex
│   ├── Critical_Analysis_Report.tex
│   ├── Critical_Analysis_Report_Extended.tex
│   └── latex.tex
│
└── LaTeX/                         # Thư mục LaTeX gốc (giữ nguyên)
    └── LaTeX_EN/
```

## Các file chính

### 1. main_vn.tex (MỚI - ĐỀ XUẤT)
- **Ngôn ngữ**: Tiếng Việt
- **Mục đích**: Báo cáo chính cho đồ án, tổng hợp tất cả nội dung
- **Cấu trúc**:
  - Abstract (Tóm tắt)
  - Phần 1: Các mô hình Deep Learning
    - CNN baseline
    - ResNet-34
    - ViT scratch (v1 & v2)
    - ViT pretrained
  - Phần 2: Tổng kết và khuyến nghị
  - Tài liệu tham khảo

### 2. Group1_Deeplearning.tex (GỐC)
- **Ngôn ngữ**: Tiếng Anh
- **Mục đích**: Báo cáo nghiên cứu chi tiết, phong cách học thuật
- **Bao gồm**: Giới thiệu, phương pháp, kết quả, so sánh với paper gốc

### 3. chapters/models/
Các file LaTeX riêng cho từng mô hình, dễ bảo trì và tái sử dụng:

| File | Nội dung | Số tham số | AUC |
|------|----------|------------|-----|
| `cnn.tex` | CNN Baseline 2 lớp Conv | ~95M | 0.60 |
| `resnet.tex` | ResNet-34 với skip connections | ~21M | 0.53 |
| `vit_scratch.tex` | ViT v1 & v2 từ đầu | 9M | 0.58-0.63 |
| `vit_pretrained.tex` | ViT pretrained timm | ~86M | 0.67 |

## Cách sử dụng

### Biên dịch báo cáo tiếng Việt
```bash
cd "D:\MSE\10.Deep Learning\Group_Final\ViT-Chest-Xray\Report"
xelatex main_vn.tex
xelatex main_vn.tex  # Chạy lần 2 để cập nhật cross-references
```

### Biên dịch báo cáo tiếng Anh
```bash
xelatex Group1_Deeplearning.tex
xelatex Group1_Deeplearning.tex
```

## Ưu điểm của cấu trúc mới

1. **Modular**: Mỗi mô hình là một file riêng, dễ chỉnh sửa
2. **Tái sử dụng**: Có thể include các file model vào nhiều báo cáo khác nhau
3. **Rõ ràng**: Phân chia logic theo chức năng
4. **Backup an toàn**: File cũ được lưu trong thư mục `backup/`
5. **Dễ mở rộng**: Thêm mô hình mới chỉ cần tạo file .tex trong `models/`

## Lưu ý

- Sử dụng **XeLaTeX** để compile (hỗ trợ tiếng Việt tốt hơn)
- File `model_documentation_vn.tex` gốc đã được tách thành các file nhỏ
- Các file trong `backup/` có thể xóa sau khi xác nhận không cần thiết

## Quy tắc đặt tên

- `main_*.tex`: File báo cáo chính
- `chapters/*.tex`: Các chương của báo cáo
- `chapters/models/*.tex`: Tài liệu chi tiết từng mô hình
- Label format: `sec:model_name`, `tab:model_name_*`, `fig:model_name_*`
