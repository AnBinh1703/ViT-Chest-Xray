# LaTeX Report - English Version

## Expert Analysis of Research Paper

**Paper:** "A Comparative Study of CNN, ResNet, and Vision Transformers for Multi-Classification of Chest Diseases"  
**Authors:** Ananya Jain, Aviral Bhardwaj, Kaushik Murali, Isha Surani  
**Source:** arXiv:2406.00237

---

## Report Structure

```
LaTeX_EN/
├── main.tex                    # Main document
├── chapters/
│   ├── 00_titlepage.tex        # Title page
│   ├── 01_abstract.tex         # Abstract
│   ├── 02_introduction.tex     # Introduction
│   ├── 03_dataset.tex          # NIH Chest X-ray dataset analysis
│   ├── 04_cnn.tex              # CNN deep analysis
│   ├── 05_resnet.tex           # ResNet deep analysis
│   ├── 06_vit.tex              # Vision Transformer deep analysis (Main Focus)
│   ├── 07_experiments.tex      # Experiments and results
│   ├── 08_implementation.tex   # PyTorch implementation
│   ├── 09_conclusion.tex       # Conclusion
│   └── 10_references.tex       # References
└── README.md                   # This file
```

---

## Compilation Instructions

### Required LaTeX Distribution
- **TeX Live 2022+** or **MiKTeX**
- XeLaTeX compiler (for custom fonts)

### Required Packages
```
fontspec, geometry, tikz, tcolorbox, listings,
booktabs, longtable, amsmath, hyperref, enumitem
```

### Compile Commands

**Using XeLaTeX (Recommended):**
```bash
cd Report/LaTeX_EN
xelatex main.tex
xelatex main.tex   # Run twice for TOC and references
```

**Using Overleaf:**
1. Upload all files
2. Set compiler to XeLaTeX in settings
3. Compile

---

## Report Contents

### Chapter 1: Abstract
- Paper summary
- Main results table
- Report scope and objectives

### Chapter 2: Introduction
- Clinical background
- Research questions
- Paper contributions

### Chapter 3: NIH Chest X-ray Dataset
- Dataset statistics (112,120 images, 15 classes)
- Class imbalance analysis
- Multi-label nature
- Comparison with other datasets

### Chapter 4: CNN Architecture
- Convolution theory
- Mathematical formulations
- CNNClassifier implementation
- Parameter analysis (102M)

### Chapter 5: ResNet Architecture
- Degradation problem
- Skip connections theory
- Gradient flow analysis
- BasicBlock and ResNet-34 implementation (21M params)

### Chapter 6: Vision Transformer (Main Focus)
- Patch embedding with Conv2d trick
- Positional embedding types
- [CLS] token mechanism
- Multi-head self-attention mathematics
- Complete ViT implementation
- ViT-ResNet hybrid architecture

### Chapter 7: Experiments
- Training configurations
- Evaluation metrics (AUC-ROC)
- Per-class performance analysis
- Learning curves
- Ablation studies

### Chapter 8: PyTorch Implementation
- TensorFlow to PyTorch migration
- Data pipeline with DataLoader
- Training loop implementation
- AUC NaN bug fix
- API mapping guide

### Chapter 9: Conclusion
- Key findings summary
- Architectural insights
- Limitations
- Future directions

### Chapter 10: References
- Primary paper citation
- Foundational papers (CNN, ResNet, ViT)
- Medical imaging literature
- Deep learning techniques

---

## Key Features

### Expert Analysis
- Deep dive into mathematical formulations
- Paper quotes mapped to code
- Architecture diagrams with TikZ
- Parameter count analysis

### Code Mapping
- Every paper concept linked to implementation
- PyTorch best practices
- Complete runnable examples

### Visual Aids
- TikZ architecture diagrams
- Color-coded information boxes
- Comparison tables

---

## Fonts Used

| Purpose | Font |
|---------|------|
| Main text | Times New Roman |
| Headings | Arial |
| Code | Consolas |

---

## Notes

- This is the **English version** of the expert analysis report
- Vietnamese version available in `Report/LaTeX/`
- Both reports share the same structure and content
- Code snippets use PyTorch 2.x syntax

---

## Contact

For questions about this report, please refer to the main project repository.

---

*Generated: Expert Analysis Report for Deep Learning Course*
