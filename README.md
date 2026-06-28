# ViT-Chest-Xray

Comparative chest X-ray disease classification with CNN, ResNet, and Vision Transformer models in PyTorch.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-NIH%20ChestXray14-blue)](https://www.kaggle.com/datasets/nih-chest-xrays/data)

## Overview

This repository contains:

- Model implementations for CNN, ResNet, and ViT classifiers.
- Data utilities for dataset parsing, transforms, and patient-level splits.
- Config-driven training scaffolding with YAML inheritance.
- Experiment notebooks, saved checkpoints, report sources, and tests.

The target task is multi-label prediction on the NIH ChestX-ray14 dataset with 15 labels, including `No Finding`.

## Repository Status

The active codebase is organized around the current top-level modules under `src/`, `configs/`, `scripts/`, and `notebooks/`.

- `main.py` provides a small CLI for `train`, `verify`, `models`, `evaluate`, and `predict`.
- `scripts/train.py` contains the training entry point and configuration plumbing.
- `src/data/`, `src/models/`, `src/losses/`, and `src/utils/` contain reusable project code.
- `models/checkpoints/` stores tracked example weights and plots.

Legacy duplicate files from the old `Project/` tree were removed during cleanup.

## Project Layout

```text
ViT-Chest-Xray/
|-- main.py
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- base.yaml
|   |-- cnn.yaml
|   |-- resnet.yaml
|   `-- vit_small.yaml
|-- scripts/
|   |-- demo.py
|   `-- train.py
|-- src/
|   |-- data/
|   |-- losses/
|   |-- models/
|   `-- utils/
|-- notebooks/
|   |-- analysis/
|   `-- experiments/
|-- models/
|   `-- checkpoints/
|-- assets/
|   `-- image/
|-- docs/
`-- tests/
```

## Data Layout

The default configuration now expects data in these locations:

- Images root: `data/processed/data`
- Labels CSV: `data/raw/Data_Entry_2017_v2020.csv`

You can override those paths with environment variables:

```powershell
$env:DATA_ROOT="D:\path\to\images"
$env:LABELS_FILE="D:\path\to\Data_Entry_2017_v2020.csv"
```

Dataset facts used by the project:

- 112,120 images
- 30,805 patients
- 14 disease labels plus `No Finding`
- multi-label classification setting

Patient-level splitting is implemented to reduce leakage between train, validation, and test sets.

## Quick Start

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Verify the environment:

```bash
python main.py verify
```

List or smoke-test available models:

```bash
python main.py models
python main.py models --test
```

Train with a config:

```bash
python main.py train --config configs/vit_small.yaml
python main.py train --config configs/resnet.yaml
python main.py train --config configs/cnn.yaml
```

Run tests:

```bash
python -m pytest tests -v
```

## Configs

All model configs inherit from `configs/base.yaml`.

- `configs/cnn.yaml`: baseline CNN settings.
- `configs/resnet.yaml`: ResNet with configurable variant and pretrained backbone support.
- `configs/vit_small.yaml`: ViT small configuration with cosine schedule and combined loss.

Useful config features:

- `_base_` inheritance
- environment-variable interpolation such as `${DATA_ROOT:...}`
- CLI overrides passed after the config path

Example:

```bash
python main.py train --config configs/vit_small.yaml training.lr=1e-4 training.epochs=20
```

## Models and Losses

Implemented model families:

- CNN baseline
- ResNet variants: `resnet18`, `resnet34`, `resnet50`, `resnet101`
- Vision Transformer variants: `small`, `base`, `large`
- pretrained wrappers in `src/models/pretrained.py`

Implemented loss functions:

- BCE-based baseline
- focal loss
- weighted BCE
- asymmetric loss
- dice loss
- combined loss
- label smoothing BCE
- distillation loss

## Notebooks

The notebook tree is split into two areas:

- `notebooks/analysis/`: dataset download and exploratory analysis.
- `notebooks/experiments/`: baseline runs, ViT runs, transfer learning, and improvement studies.

Key notebooks:

- `notebooks/analysis/data_download.ipynb`
- `notebooks/analysis/data.ipynb`
- `notebooks/experiments/cnn.ipynb`
- `notebooks/experiments/resnet.ipynb`
- `notebooks/experiments/ViT-v1.ipynb`
- `notebooks/experiments/ViT-v2.ipynb`
- `notebooks/experiments/ViT-ResNet.ipynb`
- `notebooks/experiments/Final_ViT_ChestXray.ipynb`

## Results Artifacts

Tracked artifacts are stored under `models/checkpoints/` and include:

- trained `.pth` weights
- training history figures
- ROC curve figures

Report figures are stored in `assets/image/`.

## Documentation

Project documentation is under `docs/`.

- `docs/report_vi.tex`
- `docs/report_en.tex`
- `docs/Proposal/`
- `docs/Paper.pdf`

## Notes

- `main.py evaluate` and `main.py predict` are placeholders and currently print basic arguments only.
- The repository includes both reusable modules and experiment notebooks; not every notebook is part of the CLI path.

## Reference

This repository is an academic review and implementation study based on the original ViT chest X-ray comparison work:

- Paper: https://arxiv.org/abs/2406.00237
- Original repository: https://github.com/Aviral-03/ViT-Chest-Xray

Last updated: 2026-03-25
