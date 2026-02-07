# ⚠️ ARCHIVED LEGACY CODE

**This folder contains legacy/original code and should NOT be used for new development.**

## Why is this archived?

The code in this folder has been refactored and moved to the new project structure:

| Original Location | New Location |
|-------------------|--------------|
| `Project/*.ipynb` | `notebooks/` |
| `Project/improve/*.py` | `src/losses/` |
| `Project/config.py` | `configs/*.yaml` |
| Model definitions | `src/models/` |
| Training utilities | `src/utils/` |

## New Project Structure

```
ViT-Chest-Xray/
├── src/                    # Main source code
│   ├── models/             # Model architectures (CNN, ResNet, ViT)
│   ├── data/               # Dataset and data loading
│   ├── losses/             # Loss functions
│   └── utils/              # Utilities (training, callbacks, config)
├── configs/                # YAML configuration files
├── scripts/                # CLI scripts (train.py, evaluate.py)
├── notebooks/              # Jupyter notebooks (clean versions)
└── tests/                  # Unit tests
```

## Migration Guide

### Training with new structure:
```bash
# Old way
python Project/train.py --model cnn

# New way
python scripts/train.py --config configs/vit_small.yaml
```

### Importing models:
```python
# Old way
from Project.improve.focal_loss import FocalLoss

# New way
from src.losses.focal_loss import FocalLoss
```

## Important Notes

1. **DO NOT MODIFY** files in this folder
2. All new development should use the `src/` package
3. This folder will be removed in a future cleanup

---
*Archived on: February 2026*
*Reason: Code restructuring for research-grade organization*
