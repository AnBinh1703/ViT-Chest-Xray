# LaTeX Report: Expert Analysis of Vision Transformer for Chest X-ray Classification

## Overview

Đây là báo cáo LaTeX phân tích chuyên sâu paper:

**"A Comparative Study of CNN, ResNet, and Vision Transformers for Multi-Classification of Chest Diseases"**

- **Authors:** Ananya Jain, Aviral Bhardwaj, Kaushik Murali, Isha Surani
- **Institution:** University of Toronto
- **arXiv:** [2406.00237](https://arxiv.org/abs/2406.00237)

## Cấu trúc báo cáo

```
LaTeX/
├── main.tex                    # Main document
└── chapters/
    ├── 00_titlepage.tex        # Trang bìa
    ├── 01_abstract.tex         # Tóm tắt
    ├── 02_introduction.tex     # Giới thiệu bối cảnh y tế
    ├── 03_dataset.tex          # Phân tích NIH Chest X-ray
    ├── 04_cnn.tex              # CNN: Lý thuyết + Code mapping
    ├── 05_resnet.tex           # ResNet: Skip connections + Code
    ├── 06_vit.tex              # ViT: Deep expert analysis (MAIN)
    ├── 07_experiments.tex      # Kết quả và phân tích
    ├── 08_implementation.tex   # PyTorch implementation details
    ├── 09_conclusion.tex       # Kết luận
    └── 10_references.tex       # Tài liệu tham khảo
```

## Biên dịch

### Yêu cầu
- XeLaTeX (để hỗ trợ font Unicode)
- Font: Times New Roman, Arial, Consolas

### Lệnh biên dịch
```bash
xelatex main.tex
xelatex main.tex  # Chạy 2 lần để cập nhật references
```

Hoặc sử dụng Overleaf với compiler XeLaTeX.

## Nội dung chính

### Chapter 4: CNN
- Convolution operation mathematics
- Max pooling analysis
- Paper-to-code mapping chi tiết
- Parameter count analysis

### Chapter 5: ResNet
- Degradation problem explanation
- Skip connection mathematics
- BasicBlock và ResNet-34 implementation
- Gradient flow analysis

### Chapter 6: Vision Transformer (MAIN FOCUS)
- Patch embedding với Conv2d trick
- Positional embedding (learnable vs fixed)
- [CLS] token mechanism
- Multi-head self-attention deep dive
- Complete ViT implementation walkthrough
- ViT-ResNet hybrid architecture
- Attention visualization concepts

### Chapter 8: PyTorch Implementation
- TensorFlow → PyTorch migration
- AUC NaN bug fix
- Complete training loop
- Code snippets với explanations

## Highlights

1. **Expert-level analysis:** Giải thích từ mathematical foundations đến implementation
2. **Paper-to-code mapping:** Mỗi mô tả trong paper được map đến code cụ thể
3. **Visual diagrams:** TikZ diagrams cho architectures
4. **Practical insights:** Bug fixes, tips, và best practices

## Authors

- Original Paper: University of Toronto Team
- PyTorch Implementation & LaTeX Report: [Your Name]

## License

Educational use only.
