"""
Focal Loss Variants
===================

Implementation of Focal Loss for both multi-label and single-label classification.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, Union

try:
    from .config import LossConfig
except ImportError:
    from config import LossConfig


class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-Label Classification.

    Automatically down-weights easy examples and focuses training on hard negatives.
    Essential for handling class imbalance in medical imaging.

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)

    Mathematical Formulation:
    -------------------------
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
    - p_t = p if y=1, else (1-p)
    - α balances positive/negative examples
    - γ is the focusing parameter (γ=0 reduces to standard BCE)

    Args:
        alpha (float): Weighting factor for positive class [0, 1]. Default: 0.25
        gamma (float): Focusing parameter γ ≥ 0. Default: 2.0
            - γ = 0: Standard cross-entropy loss
            - γ = 2: Commonly used value, works well in practice
            - γ ↑: More focus on hard, misclassified examples
        pos_weight (Tensor, optional): Per-class positive weights. Shape: (C,)
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'

    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(32, 15)  # (batch, num_classes)
        >>> targets = torch.randint(0, 2, (32, 15)).float()  # multi-label
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        alpha: float = LossConfig.FOCAL_ALPHA,
        gamma: float = LossConfig.FOCAL_GAMMA,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

        # Validate parameters
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if not 0 <= alpha <= 1:
            warnings.warn(f"alpha={alpha} is outside [0, 1] range")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Logits of shape (N, C) - before sigmoid
            targets: Binary labels of shape (N, C)

        Returns:
            Focal loss value
        """
        # Compute BCE loss (without reduction for focal weighting)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )

        # Get probability of correct prediction
        # pt = p if target=1, else (1-p)
        pt = torch.exp(-bce_loss)

        # Focal weight: (1 - pt)^gamma
        # - Easy examples (pt → 1): weight → 0 (down-weighted)
        # - Hard examples (pt → 0): weight → 1 (preserved)
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha and focal weight
        focal_loss = self.alpha * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

    def __repr__(self) -> str:
        return f"FocalLoss(alpha={self.alpha}, gamma={self.gamma})"


class FocalLossSingleLabel(nn.Module):
    """
    Focal Loss for Single-Label (Multi-Class) Classification.

    Variant of Focal Loss for standard classification tasks where
    each sample belongs to exactly one class.

    Args:
        alpha (float or Tensor): Class balancing weight(s). Default: 1.0
        gamma (float): Focusing parameter. Default: 2.0
        reduction (str): Reduction method. Default: 'mean'
    """

    def __init__(
        self,
        alpha: Union[float, torch.Tensor] = 1.0,
        gamma: float = LossConfig.FOCAL_GAMMA,
        reduction: str = 'mean'
    ):
        super(FocalLossSingleLabel, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Class indices of shape (N,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Apply per-class alpha if tensor
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss