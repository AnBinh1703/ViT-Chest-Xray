"""
Loss Functions Package
======================

A comprehensive collection of loss functions for medical image classification.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

from .config import LossConfig
from .focal_loss import FocalLoss, FocalLossSingleLabel
from .weighted_loss import WeightedBCELoss
from .smoothing_loss import LabelSmoothingBCE, LabelSmoothingCrossEntropy
from .asymmetric_loss import AsymmetricLoss
from .combined_loss import CombinedLoss
from .dice_loss import DiceLoss, DiceBCELoss
from .distillation_loss import DistillationLoss
from .utils import (
    compute_class_weights,
    compute_class_weights_from_dataframe,
    compute_sample_weights,
    create_loss_function
)
from .comparator import LossComparator, save_loss_comparison_results

__all__ = [
    # Configuration
    'LossConfig',

    # Loss Functions
    'FocalLoss',
    'FocalLossSingleLabel',
    'WeightedBCELoss',
    'LabelSmoothingBCE',
    'LabelSmoothingCrossEntropy',
    'AsymmetricLoss',
    'CombinedLoss',
    'DiceLoss',
    'DiceBCELoss',
    'DistillationLoss',

    # Utilities
    'compute_class_weights',
    'compute_class_weights_from_dataframe',
    'compute_sample_weights',
    'create_loss_function',

    # Comparison Tools
    'LossComparator',
    'save_loss_comparison_results',
]

__version__ = "1.0.0"