"""
Label Smoothing Losses
======================

Label smoothing implementations for both BCE and Cross-Entropy losses.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from .config import LossConfig
except ImportError:
    from config import LossConfig


class LabelSmoothingBCE(nn.Module):
    """
    Label Smoothing for Binary Cross-Entropy Loss.

    Prevents model overconfidence by converting hard labels to soft targets.
    Particularly useful for noisy labels (common in medical imaging datasets).

    Mathematical Formulation:
    -------------------------
    Smoothed target: y_smooth = y * (1 - ε) + 0.5 * ε

    Original: y ∈ {0, 1}
    Smoothed: y ∈ [ε/2, 1 - ε/2]

    Benefits:
    ---------
    - Reduces overfitting to noisy labels
    - Better model calibration
    - Improved generalization

    Args:
        smoothing (float): Smoothing factor ε ∈ [0, 1]. Default: 0.1
        pos_weight (Tensor, optional): Positive class weights
        reduction (str): Reduction method. Default: 'mean'

    Note:
        For NIH Chest X-ray dataset with ~10% label noise,
        smoothing=0.1 is recommended.
    """

    def __init__(
        self,
        smoothing: float = LossConfig.SMOOTHING_FACTOR,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super(LabelSmoothingBCE, self).__init__()

        if not 0 <= smoothing <= 1:
            raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")

        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Binary labels of shape (N, C)
        """
        # Smooth labels: 1 → (1 - ε), 0 → ε/2
        # This shifts both 0s and 1s toward 0.5 by ε
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        return F.binary_cross_entropy_with_logits(
            inputs, targets_smooth,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )

    def __repr__(self) -> str:
        return f"LabelSmoothingBCE(smoothing={self.smoothing})"


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing for Cross-Entropy Loss (single-label classification).

    Args:
        smoothing (float): Smoothing factor. Default: 0.1
        reduction (str): Reduction method. Default: 'mean'
    """

    def __init__(
        self,
        smoothing: float = LossConfig.SMOOTHING_FACTOR,
        reduction: str = 'mean'
    ):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Class indices of shape (N,)
        """
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create soft targets
        with torch.no_grad():
            targets_smooth = torch.zeros_like(log_probs)
            targets_smooth.fill_(self.smoothing / (num_classes - 1))
            targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = torch.sum(-targets_smooth * log_probs, dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss