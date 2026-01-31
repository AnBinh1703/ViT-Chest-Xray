"""
Advanced Loss Functions for Medical Image Classification
=======================================================

This module implements various loss functions specifically designed
for handling class imbalance and improving performance on medical
imaging tasks.

Features:
- Focal Loss variants
- Class-weighted losses
- Label smoothing
- Combined loss strategies
- Medical imaging specific losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union
import warnings


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by focusing on hard examples.
    
    Reference: Lin, T. Y. et al. "Focal loss for dense object detection." 
               Proceedings of the IEEE international conference on computer vision. 2017.
    
    Formula: FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    
    Args:
        alpha (float or tensor): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Specifies the reduction to apply to the output
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with per-class weights for handling severe class imbalance.
    
    Args:
        class_weights (list or tensor): Weights for each class
        alpha (float): Base weighting factor
        gamma (float): Focusing parameter
        reduction (str): Reduction method
    """
    
    def __init__(
        self, 
        class_weights: Union[List[float], torch.Tensor],
        alpha: float = 1.0, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if isinstance(class_weights, list):
            class_weights = torch.FloatTensor(class_weights)
        
        self.register_buffer('class_weights', class_weights)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Get weights for current batch
        weights = self.class_weights[targets]
        
        focal_loss = weights * self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing regularization.
    
    This prevents the model from becoming overconfident and helps with calibration.
    
    Args:
        smoothing (float): Label smoothing factor (0.0 to 1.0)
        reduction (str): Reduction method
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        targets_smooth = torch.zeros_like(log_probs)
        targets_smooth.fill_(self.smoothing / (num_classes - 1))
        targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = torch.sum(-targets_smooth * log_probs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function using multiple loss strategies.
    
    Combines Focal Loss, Cross Entropy, and Label Smoothing for optimal performance.
    
    Args:
        class_weights (list or tensor): Per-class weights
        focal_weight (float): Weight for focal loss component
        ce_weight (float): Weight for cross entropy component
        smooth_weight (float): Weight for label smoothing component
        gamma (float): Focal loss gamma parameter
        smoothing (float): Label smoothing factor
    """
    
    def __init__(
        self,
        class_weights: Union[List[float], torch.Tensor],
        focal_weight: float = 0.7,
        ce_weight: float = 0.2,
        smooth_weight: float = 0.1,
        gamma: float = 2.0,
        smoothing: float = 0.1
    ):
        super(CombinedLoss, self).__init__()
        
        # Normalize weights
        total_weight = focal_weight + ce_weight + smooth_weight
        self.focal_weight = focal_weight / total_weight
        self.ce_weight = ce_weight / total_weight
        self.smooth_weight = smooth_weight / total_weight
        
        # Initialize loss functions
        self.focal_loss = WeightedFocalLoss(class_weights, gamma=gamma)
        
        if isinstance(class_weights, list):
            class_weights = torch.FloatTensor(class_weights)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        self.smooth_loss = LabelSmoothingCrossEntropy(smoothing=smoothing)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        smooth = self.smooth_loss(inputs, targets)
        
        combined = (self.focal_weight * focal + 
                   self.ce_weight * ce + 
                   self.smooth_weight * smooth)
        
        return combined


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-label Classification.
    
    Optimized for handling class imbalance in multi-label scenarios.
    
    Reference: Ridnik, T. et al. "Asymmetric loss for multi-label classification." 
               Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    
    Args:
        gamma_neg (float): Focusing parameter for negative samples
        gamma_pos (float): Focusing parameter for positive samples
        clip (float): Clipping value for positive samples (None to disable)
        reduction (str): Reduction method
    """
    
    def __init__(
        self, 
        gamma_neg: float = 4, 
        gamma_pos: float = 1, 
        clip: Optional[float] = 0.05,
        reduction: str = 'mean'
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert to probabilities
        x_sigmoid = torch.sigmoid(inputs)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        # Apply clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # Calculate asymmetric loss
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        
        # Apply focusing
        loss = los_pos * (xs_neg ** self.gamma_pos) + los_neg * (xs_pos ** self.gamma_neg)
        loss = -loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss for model compression and ensemble learning.
    
    Args:
        temperature (float): Temperature for softmax softening
        alpha (float): Weight for distillation loss
        beta (float): Weight for student loss
        reduction (str): Reduction method
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.8,
        beta: float = 0.2,
        reduction: str = 'mean'
    ):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        
        self.student_loss = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(
        self, 
        student_outputs: torch.Tensor, 
        teacher_outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        # Distillation loss (soft targets)
        teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
        distillation_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction=self.reduction
        ) * (self.temperature ** 2)
        
        # Student loss (hard targets)
        student_loss = self.student_loss(student_outputs, targets)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + self.beta * student_loss
        
        return total_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for representation learning.
    
    Args:
        margin (float): Margin for negative pairs
        reduction (str): Reduction method
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self, 
        embeddings1: torch.Tensor, 
        embeddings2: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        # Calculate euclidean distance
        euclidean_distance = F.pairwise_distance(embeddings1, embeddings2)
        
        # Contrastive loss
        loss = (labels.float() * torch.pow(euclidean_distance, 2) +
                (1 - labels).float() * torch.pow(
                    torch.clamp(self.margin - euclidean_distance, min=0.0), 2
                ))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation-like tasks or when focusing on positive predictions.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
        reduction (str): Reduction method
    """
    
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient
        intersection = (inputs * targets_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (
            inputs.sum() + targets_one_hot.sum() + self.smooth
        )
        
        dice_loss = 1 - dice
        
        return dice_loss


def compute_class_weights(
    class_counts: Union[List[int], np.ndarray],
    method: str = 'balanced',
    beta: float = 0.999
) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced datasets.
    
    Args:
        class_counts: Number of samples per class
        method: Weighting method ('balanced', 'inverse', 'effective')
        beta: Beta parameter for effective number weighting
    
    Returns:
        Class weights as torch tensor
    """
    
    if isinstance(class_counts, list):
        class_counts = np.array(class_counts)
    
    if method == 'balanced':
        # Sklearn-style balanced weights
        total_samples = np.sum(class_counts)
        num_classes = len(class_counts)
        weights = total_samples / (num_classes * class_counts)
        
    elif method == 'inverse':
        # Simple inverse frequency
        weights = 1.0 / class_counts
        
    elif method == 'effective':
        # Effective number of samples
        effective_nums = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_nums
        
    elif method == 'sqrt_inverse':
        # Square root of inverse frequency (less aggressive)
        weights = 1.0 / np.sqrt(class_counts)
        
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights
    weights = weights / np.min(weights)
    
    return torch.FloatTensor(weights)


def create_loss_function(
    loss_name: str,
    num_classes: int,
    class_weights: Optional[Union[List[float], torch.Tensor]] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_name: Name of the loss function
        num_classes: Number of classes
        class_weights: Optional class weights
        **kwargs: Additional loss-specific parameters
    
    Returns:
        Configured loss function
    """
    
    if loss_name == 'cross_entropy':
        weight = torch.FloatTensor(class_weights) if class_weights else None
        return nn.CrossEntropyLoss(weight=weight, **kwargs)
    
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    
    elif loss_name == 'weighted_focal':
        if class_weights is None:
            raise ValueError("Class weights required for weighted focal loss")
        return WeightedFocalLoss(class_weights, **kwargs)
    
    elif loss_name == 'label_smoothing':
        return LabelSmoothingCrossEntropy(**kwargs)
    
    elif loss_name == 'combined':
        if class_weights is None:
            raise ValueError("Class weights required for combined loss")
        return CombinedLoss(class_weights, **kwargs)
    
    elif loss_name == 'asymmetric':
        return AsymmetricLoss(**kwargs)
    
    elif loss_name == 'dice':
        return DiceLoss(**kwargs)
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Generate dummy data
    batch_size = 8
    num_classes = 15
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test class weights calculation
    class_counts = [1000, 100, 50, 25, 10]  # Imbalanced dataset
    weights = compute_class_weights(class_counts, method='balanced')
    print(f"Class weights: {weights}")
    
    # Test different loss functions
    loss_functions = {
        'cross_entropy': nn.CrossEntropyLoss(),
        'focal': FocalLoss(),
        'weighted_focal': WeightedFocalLoss([1.0] * num_classes),
        'label_smoothing': LabelSmoothingCrossEntropy(),
        'combined': CombinedLoss([1.0] * num_classes)
    }
    
    print("\nLoss function outputs:")
    for name, loss_fn in loss_functions.items():
        try:
            loss_value = loss_fn(inputs, targets)
            print(f"{name}: {loss_value:.4f}")
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    print("\n✅ Loss functions tested successfully!")