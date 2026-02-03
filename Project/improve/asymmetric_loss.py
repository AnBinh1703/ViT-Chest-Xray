"""
Asymmetric Loss
===============

Asymmetric Loss (ASL) for multi-label classification with class imbalance.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from .config import LossConfig
except ImportError:
    from config import LossConfig


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification.

    Designed specifically for multi-label tasks with class imbalance.
    Uses different focusing parameters for positive and negative samples.

    Paper: "Asymmetric Loss For Multi-Label Classification" (Ridnik et al., ICCV 2021)

    Key Innovation:
    ---------------
    - Different gamma for positive (γ+) and negative (γ-) samples
    - Hard thresholding (clipping) for negative samples
    - Asymmetric focusing addresses label imbalance better than Focal Loss

    Mathematical Formulation:
    -------------------------
    L+ = (1 - p)^γ+ * log(p)           for positive samples
    L- = (p_clip)^γ- * log(1 - p_clip)  for negative samples

    where p_clip = max(p - m, 0) (hard thresholding with margin m)

    Args:
        gamma_neg (float): Focusing parameter for negatives. Default: 4
        gamma_pos (float): Focusing parameter for positives. Default: 1
        clip (float): Probability margin for hard negative mining. Default: 0.05
        reduction (str): Reduction method. Default: 'mean'

    Recommended Settings:
    --------------------
    - For highly imbalanced data: gamma_neg=4, gamma_pos=0
    - For moderately imbalanced: gamma_neg=4, gamma_pos=1
    - clip=0.05 works well in most cases
    """

    def __init__(
        self,
        gamma_neg: float = LossConfig.ASL_GAMMA_NEG,
        gamma_pos: float = LossConfig.ASL_GAMMA_POS,
        clip: Optional[float] = LossConfig.ASL_CLIP,
        reduction: str = 'mean',
        eps: float = 1e-8
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Binary labels of shape (N, C)
        """
        # Convert logits to probabilities
        p = torch.sigmoid(inputs)

        # Positive and negative probabilities
        p_pos = p
        p_neg = 1 - p

        # Asymmetric clipping for negatives (hard thresholding)
        if self.clip is not None and self.clip > 0:
            # Shift negative probabilities up, making them "easier"
            p_neg = (p_neg + self.clip).clamp(max=1)

        # Cross entropy components
        loss_pos = targets * torch.log(p_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(p_neg.clamp(min=self.eps))

        # Asymmetric focusing
        # Positive: (1-p)^γ+ focuses on hard positives (low p)
        # Negative: p^γ- focuses on hard negatives (high p)
        loss = -(loss_pos * ((1 - p_pos) ** self.gamma_pos) +
                 loss_neg * (p_pos ** self.gamma_neg))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def __repr__(self) -> str:
        return f"AsymmetricLoss(γ-={self.gamma_neg}, γ+={self.gamma_pos}, clip={self.clip})"