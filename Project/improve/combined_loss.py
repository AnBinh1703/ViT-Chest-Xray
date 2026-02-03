"""
Combined Loss Function
======================

Combines multiple loss strategies for robust training on imbalanced datasets.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from .config import LossConfig
    from .focal_loss import FocalLoss
    from .weighted_loss import WeightedBCELoss
    from .smoothing_loss import LabelSmoothingBCE
except ImportError:
    from config import LossConfig
    from focal_loss import FocalLoss
    from weighted_loss import WeightedBCELoss
    from smoothing_loss import LabelSmoothingBCE


class CombinedLoss(nn.Module):
    """
    Combined Loss Function using Multiple Strategies.

    Combines Focal Loss, Weighted BCE, and Label Smoothing for robust
    training on imbalanced medical imaging datasets.

    Mathematical Formulation:
    -------------------------
    L_combined = w_focal * L_focal + w_bce * L_wbce + w_smooth * L_smooth

    where weights are normalized: w_focal + w_bce + w_smooth = 1

    Args:
        pos_weight (Tensor, optional): Class weights for positive samples
        focal_weight (float): Weight for focal loss component. Default: 0.7
        bce_weight (float): Weight for weighted BCE. Default: 0.2
        smooth_weight (float): Weight for label smoothing. Default: 0.1
        alpha (float): Focal loss alpha. Default: 0.25
        gamma (float): Focal loss gamma. Default: 2.0
        smoothing (float): Label smoothing factor. Default: 0.1

    Rationale:
    ----------
    - Focal Loss (70%): Primary driver for handling hard examples
    - Weighted BCE (20%): Ensures rare classes get higher penalties
    - Label Smoothing (10%): Regularization against noisy labels
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        focal_weight: float = LossConfig.COMBINED_FOCAL_WEIGHT,
        bce_weight: float = LossConfig.COMBINED_BCE_WEIGHT,
        smooth_weight: float = LossConfig.COMBINED_SMOOTH_WEIGHT,
        alpha: float = LossConfig.FOCAL_ALPHA,
        gamma: float = LossConfig.FOCAL_GAMMA,
        smoothing: float = LossConfig.SMOOTHING_FACTOR
    ):
        super(CombinedLoss, self).__init__()

        # Normalize weights
        total = focal_weight + bce_weight + smooth_weight
        self.focal_weight = focal_weight / total
        self.bce_weight = bce_weight / total
        self.smooth_weight = smooth_weight / total

        # Initialize component losses
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
        self.bce_loss = WeightedBCELoss(pos_weight=pos_weight)
        self.smooth_loss = LabelSmoothingBCE(smoothing=smoothing, pos_weight=pos_weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Binary labels of shape (N, C)
        """
        focal = self.focal_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        smooth = self.smooth_loss(inputs, targets)

        combined = (self.focal_weight * focal +
                    self.bce_weight * bce +
                    self.smooth_weight * smooth)

        return combined

    def __repr__(self) -> str:
        return (f"CombinedLoss(focal={self.focal_weight:.2f}, "
                f"bce={self.bce_weight:.2f}, smooth={self.smooth_weight:.2f})")