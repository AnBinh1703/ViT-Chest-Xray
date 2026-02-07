"""
Knowledge Distillation Loss
===========================

Knowledge distillation for model compression and transfer learning.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for model compression.

    Trains a smaller "student" model to mimic a larger "teacher" model.

    Paper: "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)

    Mathematical Formulation:
    -------------------------
    L = α * KL(softmax(z_s/T), softmax(z_t/T)) * T² + β * CE(z_s, y)

    where:
    - z_s, z_t: student and teacher logits
    - T: temperature (higher = softer probabilities)
    - α, β: loss component weights

    Args:
        temperature (float): Softmax temperature. Default: 4.0
        alpha (float): Weight for distillation loss. Default: 0.7
        beta (float): Weight for hard label loss. Default: 0.3
        reduction (str): Reduction method. Default: 'mean'
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        reduction: str = 'mean'
    ):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            student_logits: Student model outputs (N, C)
            teacher_logits: Teacher model outputs (N, C)
            targets: Ground truth labels (N, C) for multi-label or (N,) for single-label
        """
        T = self.temperature

        # Soft targets from teacher (with temperature scaling)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        student_soft = F.log_softmax(student_logits / T, dim=1)

        # Distillation loss (KL divergence)
        distill_loss = F.kl_div(
            student_soft, teacher_soft,
            reduction='batchmean'
        ) * (T ** 2)  # Scale by T² to maintain gradient magnitude

        # Hard label loss
        if targets.dim() == 1:
            # Single-label classification
            hard_loss = F.cross_entropy(student_logits, targets)
        else:
            # Multi-label classification
            hard_loss = F.binary_cross_entropy_with_logits(
                student_logits, targets.float()
            )

        # Combined loss
        return self.alpha * distill_loss + self.beta * hard_loss

    def __repr__(self) -> str:
        return f"DistillationLoss(T={self.temperature}, α={self.alpha}, β={self.beta})"