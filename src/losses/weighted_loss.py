"""
Weighted Binary Cross-Entropy Loss
===================================

BCE loss with class weights for handling imbalanced datasets.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss with Class Weights.

    Applies higher penalties for misclassifying rare classes.

    Mathematical Formulation:
    -------------------------
    L = -w_pos * y * log(p) - (1 - y) * log(1 - p)

    where w_pos is computed based on class frequency:
        w_i = (N_total - N_positive_i) / N_positive_i

    Args:
        pos_weight (Tensor, optional): Weights for positive samples per class
        reduction (str): Reduction method. Default: 'mean'

    Example:
        >>> # Compute weights from class counts
        >>> class_counts = [60361, 19894, 11559, ..., 227]
        >>> pos_weight = compute_class_weights(class_counts)
        >>> criterion = WeightedBCELoss(pos_weight=pos_weight)
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Binary labels of shape (N, C)
        """
        return F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )

    def __repr__(self) -> str:
        has_weight = self.pos_weight is not None
        return f"WeightedBCELoss(has_pos_weight={has_weight})"