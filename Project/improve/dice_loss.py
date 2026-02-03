"""
Dice Loss Variants
==================

Dice loss and Dice-BCE combined loss for classification tasks.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from .weighted_loss import WeightedBCELoss
except ImportError:
    from weighted_loss import WeightedBCELoss


class DiceLoss(nn.Module):
    """
    Dice Loss (Sørensen–Dice coefficient) for classification.

    Originally from segmentation, adapted for classification tasks.
    Focuses on the overlap between predictions and ground truth.

    Mathematical Formulation:
    -------------------------
    Dice = 2 * |P ∩ G| / (|P| + |G|)
    Loss = 1 - Dice

    Args:
        smooth (float): Smoothing factor to avoid division by zero. Default: 1e-6
        reduction (str): Reduction method. Default: 'mean'
    """

    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (N, C)
            targets: Binary labels of shape (N, C)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE Loss.

    Combines the benefits of both losses:
    - Dice: Focuses on overall overlap
    - BCE: Provides per-pixel/per-sample gradients

    Args:
        dice_weight (float): Weight for Dice loss. Default: 0.5
        bce_weight (float): Weight for BCE loss. Default: 0.5
        pos_weight (Tensor, optional): BCE positive class weights
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
        smooth: float = 1e-6
    ):
        super(DiceBCELoss, self).__init__()

        total = dice_weight + bce_weight
        self.dice_weight = dice_weight / total
        self.bce_weight = bce_weight / total

        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = WeightedBCELoss(pos_weight=pos_weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)

        return self.dice_weight * dice + self.bce_weight * bce