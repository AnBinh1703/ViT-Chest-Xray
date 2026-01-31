# 🔬 Research Review: Chest X-ray Multi-Classification with Deep Learning

[![Paper](https://img.shields.io/badge/arXiv-2406.00237-b31b1b.svg)](https://arxiv.org/abs/2406.00237)
[![Original Repo](https://img.shields.io/badge/GitHub-Original%20Repo-blue)](https://github.com/Aviral-03/ViT-Chest-Xray)
[![Framework](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Disclaimer - Academic Research & Review

> **⚠️ IMPORTANT NOTICE**
>
> This repository is a **research review and academic study** of the original work:
>
> - **Original Repository:** [https://github.com/Aviral-03/ViT-Chest-Xray](https://github.com/Aviral-03/ViT-Chest-Xray)
> - **Original Paper:** [arXiv:2406.00237](https://arxiv.org/abs/2406.00237) - *"A Comparative Study of CNN, ResNet, and Vision Transformers for Multi-Classification of Chest Diseases"*
> - **Original Authors:** Ananya Jain, Aviral Bhardwaj, Kaushik Murali, Isha Surani (University of Toronto)
>
> **This work is conducted purely for academic purposes** as part of my **Master's degree in Data Science at FPT School of Business (FSB)**. There is **no intention of plagiarism**. All credit for the original research goes to the original authors.

---

## 🎯 Purpose of This Research Review

### What I Did

| Activity | Description |
|----------|-------------|
| **📖 Literature Review** | Deep analysis of the paper's methodology, architecture, and results |
| **🔧 Code Migration** | Converted original TensorFlow/Keras code to **PyTorch 2.x** for learning |
| **🐛 Bug Fixes** | Fixed issues (AUC NaN, memory leaks) to run on personal machine |
| **📝 Documentation** | Created comprehensive documentation (Vietnamese + English) |
| **🔬 Experimentation** | Tested and validated models on local hardware |

### Why This Study

1. **Deep Learning Course Requirement** - Final project for Master's program at FSB
2. **Hands-on Learning** - Understanding CNN, ResNet, and Vision Transformer architectures
3. **Code Understanding** - Learning by reimplementing in PyTorch
4. **Research Skills** - Practicing paper review and analysis

---

## 📚 Original Paper Summary

**Title:** *A Comparative Study of CNN, ResNet, and Vision Transformers for Multi-Classification of Chest Diseases*

**Key Contributions (from original authors):**
- Comparative study of 5 deep learning architectures for chest X-ray classification
- Multi-label classification on NIH Chest X-ray dataset (112,120 images, 15 classes)
- Analysis of CNN vs ResNet vs Vision Transformer performance

**Results from Paper:**

| Model | Train Accuracy | Test AUC | Parameters |
|-------|---------------|----------|------------|
| CNN | 91.0% | 0.82 | 102M |
| ResNet-34 | 93.0% | **0.86** | 21M |
| ViT-v1/32 | 92.63% | 0.86 | ~3M |
| ViT-v2/32 | 92.83% | 0.84 | ~3M |
| ViT-ResNet/16 | **93.9%** | 0.85 | ~15M |

---

## 🗂️ Repository Structure

```
ViT-Chest-Xray/
│
├── 📄 README.md                 # This file (Research Review Documentation)
├── 📄 COMPLETE_DOCUMENTATION.md # Comprehensive analysis (Vietnamese)
├── 📄 IMPROVEMENT_PLAN.md       # Future improvement roadmap
├── 📄 requirements.txt          # Python dependencies
├── 📄 2406.00237v1.pdf          # Original paper PDF
│
├── 📁 Project/                  # Main code (PyTorch migrated)
│   ├── 📓 data_download.ipynb   # Dataset download script
│   ├── 📓 data.ipynb            # Data preprocessing & EDA
│   ├── 📓 cnn.ipynb             # CNN model (PyTorch)
│   ├── 📓 resnet.ipynb          # ResNet-34 model (PyTorch)
│   ├── 📓 ViT-v1.ipynb          # Vision Transformer v1 (PyTorch)
│   ├── 📓 ViT-v2.ipynb          # Vision Transformer v2 (PyTorch)
│   ├── 📓 ViT-ResNet.ipynb      # Hybrid ViT-ResNet (PyTorch)
│   ├── 📁 analyst/              # Per-notebook analysis (Markdown)
│   ├── 📁 data/                 # Dataset storage (not in git)
│   └── 📁 input/                # Metadata CSV files
│
├── 📁 Report/                   # LaTeX Reports
│   ├── 📁 LaTeX/                # Vietnamese version
│   └── 📁 LaTeX_EN/             # English version
│
└── 📁 Proposal/                 # Initial project proposal
```

---

## 🔄 My Modifications & Contributions

### 1. Framework Migration: TensorFlow → PyTorch

```python
# Original (TensorFlow/Keras)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    ...
])

# My Migration (PyTorch)
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        ...
```

### 2. Bug Fixes

| Bug | Problem | My Fix |
|-----|---------|--------|
| AUC NaN | Single-class batches cause NaN | Added validation for unique labels |
| Memory Leak | No `torch.no_grad()` in eval | Proper evaluation mode |
| Data Loading | Path issues on Windows | Cross-platform path handling |

### 3. Documentation Created

- **COMPLETE_DOCUMENTATION.md** - 2700+ lines of deep analysis
- **Report/LaTeX/** - Vietnamese expert report (11 chapters)
- **Report/LaTeX_EN/** - English expert report (11 chapters)
- **Project/analyst/** - Per-notebook analysis files

---

## 🚀 How to Run (My Setup)

### Prerequisites

```bash
Python 3.10+
CUDA 11.8+ (optional, for GPU)
~50GB disk space (for dataset)
```

### Installation

```bash
# Clone this review repository
git clone https://github.com/YOUR_USERNAME/ViT-Chest-Xray.git
cd ViT-Chest-Xray

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

```bash
# Option 1: Full dataset (~42GB)
# Run Project/data_download.ipynb

# Option 2: Sample dataset (for testing)
# Download from: https://www.kaggle.com/datasets/nih-chest-xrays/sample
```

### Run Notebooks

```bash
cd Project
jupyter notebook

# Run in order:
# 1. data.ipynb (preprocessing)
# 2. cnn.ipynb / resnet.ipynb / ViT-*.ipynb (training)
```

---

## 📊 My Learning Outcomes

### Concepts Understood

- [x] **Convolutional Neural Networks** - Feature extraction via learned filters
- [x] **Residual Connections** - Skip connections for gradient flow
- [x] **Vision Transformers** - Patch embedding + Self-attention
- [x] **Multi-label Classification** - BCE loss for independent labels
- [x] **AUC-ROC Metric** - Threshold-independent evaluation
- [x] **Class Imbalance** - Handling skewed medical datasets

### Skills Practiced

- [x] PyTorch model implementation from scratch
- [x] Reading and understanding research papers
- [x] Code migration between frameworks
- [x] Technical documentation writing
- [x] LaTeX report preparation

---

## 📖 References

### Primary Sources

1. **Original Paper:** Jain, A., Bhardwaj, A., Murali, K., & Surani, I. (2024). *A Comparative Study of CNN, ResNet, and Vision Transformers for Multi-Classification of Chest Diseases*. arXiv:2406.00237.

2. **Original Repository:** [https://github.com/Aviral-03/ViT-Chest-Xray](https://github.com/Aviral-03/ViT-Chest-Xray)

### Foundational Papers

3. He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
4. Dosovitskiy, A., et al. (2021). *An Image is Worth 16x16 Words*. ICLR.
5. Wang, X., et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database*. CVPR.

---

## 🙏 Acknowledgments

- **Original Authors:** Ananya Jain, Aviral Bhardwaj, Kaushik Murali, Isha Surani for their excellent research
- **University of Toronto** for making the research publicly available
- **NIH Clinical Center** for the ChestX-ray14 dataset
- **FPT School of Business (FSB)** for the academic opportunity
- **Deep Learning Course Instructors** for guidance

---

## 📜 License

This research review follows the MIT License of the original repository. All original work and intellectual property belong to the original authors.

---

## 👤 Reviewer Information

| Field | Information |
|-------|-------------|
| **Name** | [Your Name] |
| **Program** | Master of Science in Data Science |
| **Institution** | FPT School of Business (FSB) |
| **Course** | Deep Learning |
| **Semester** | [Current Semester] |
| **Purpose** | Academic Research & Learning |

---

*Last Updated: February 2025*

---

## 📈 Future Improvements

See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for detailed roadmap on how to enhance this project beyond the original paper.
