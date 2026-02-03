"""
Loss Function Utilities
=======================

Class weight computation and sample weighting utilities.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Union, Optional

try:
    from .config import LossConfig
except ImportError:
    from config import LossConfig


def compute_class_weights(
    class_counts: Union[List[int], np.ndarray, torch.Tensor],
    method: str = 'balanced',
    beta: float = LossConfig.EFFECTIVE_BETA,
    min_weight: float = LossConfig.MIN_WEIGHT,
    max_weight: float = LossConfig.MAX_WEIGHT,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced datasets.

    Methods:
    --------
    1. 'balanced' (sklearn-style):
       w_i = N_total / (n_classes * N_i)

    2. 'inverse':
       w_i = 1 / N_i

    3. 'inverse_sqrt':
       w_i = 1 / sqrt(N_i)
       Less aggressive than inverse, recommended for moderate imbalance

    4. 'effective' (Class-Balanced Loss):
       w_i = (1 - β) / (1 - β^N_i)
       Paper: "Class-Balanced Loss Based on Effective Number of Samples"

    5. 'pos_neg_ratio':
       w_i = (N_total - N_i) / N_i
       Traditional positive weight for BCE loss

    Args:
        class_counts: Number of samples per class
        method: Weighting method. Options: 'balanced', 'inverse',
                'inverse_sqrt', 'effective', 'pos_neg_ratio'
        beta: Beta for effective number method. Default: 0.9999
        min_weight: Minimum allowed weight (clipping). Default: 0.5
        max_weight: Maximum allowed weight (clipping). Default: 100.0
        normalize: If True, normalize weights so min weight = 1.0

    Returns:
        torch.Tensor: Class weights of shape (num_classes,)

    Example:
        >>> class_counts = [60361, 19894, 11559, 5782, 227]  # Imbalanced
        >>> weights = compute_class_weights(class_counts, method='effective')
        >>> criterion = WeightedBCELoss(pos_weight=weights)
    """
    # Convert to numpy array
    if isinstance(class_counts, torch.Tensor):
        class_counts = class_counts.numpy()
    elif isinstance(class_counts, list):
        class_counts = np.array(class_counts, dtype=np.float64)
    else:
        class_counts = class_counts.astype(np.float64)

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    total_samples = np.sum(class_counts)
    num_classes = len(class_counts)

    if method == 'balanced':
        # Sklearn-style balanced weights
        weights = total_samples / (num_classes * class_counts)

    elif method == 'inverse':
        # Simple inverse frequency
        weights = 1.0 / class_counts

    elif method == 'inverse_sqrt':
        # Square root of inverse (less aggressive)
        weights = 1.0 / np.sqrt(class_counts)

    elif method == 'effective':
        # Effective number of samples (Class-Balanced Loss)
        # Accounts for diminishing returns of adding more samples
        effective_nums = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.maximum(effective_nums, 1e-8)

    elif method == 'pos_neg_ratio':
        # Positive-negative ratio (for BCE)
        weights = (total_samples - class_counts) / class_counts

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Choose from: 'balanced', 'inverse', 'inverse_sqrt', 'effective', 'pos_neg_ratio'"
        )

    # Clip weights to prevent extreme values
    weights = np.clip(weights, min_weight, max_weight)

    # Normalize so minimum weight is 1.0
    if normalize:
        weights = weights / np.min(weights)

    return torch.FloatTensor(weights)


def compute_class_weights_from_dataframe(
    df: pd.DataFrame,
    disease_columns: List[str],
    method: str = 'pos_neg_ratio',
    **kwargs
) -> torch.Tensor:
    """
    Compute class weights directly from a pandas DataFrame.

    Args:
        df: DataFrame with binary label columns
        disease_columns: List of column names for each disease class
        method: Weighting method (see compute_class_weights)
        **kwargs: Additional arguments for compute_class_weights

    Returns:
        torch.Tensor: Class weights
    """
    class_counts = df[disease_columns].sum().values
    return compute_class_weights(class_counts, method=method, **kwargs)


def compute_sample_weights(
    df: pd.DataFrame,
    disease_columns: List[str],
    strategy: str = 'max_class_weight'
) -> np.ndarray:
    """
    Compute per-sample weights for WeightedRandomSampler.

    Ensures each training batch has balanced representation of rare classes.

    Strategies:
    -----------
    1. 'max_class_weight': Sample weight = max weight of positive classes
       Prioritizes samples with rare diseases

    2. 'sum_class_weight': Sample weight = sum of positive class weights
       Considers all diseases in the sample

    3. 'mean_class_weight': Sample weight = mean of positive class weights
       Normalized version of sum strategy

    Args:
        df: DataFrame with binary label columns
        disease_columns: List of disease column names
        strategy: Weighting strategy. Default: 'max_class_weight'

    Returns:
        np.ndarray: Per-sample weights of shape (N,)

    Example:
        >>> sample_weights = compute_sample_weights(train_df, LABELS)
        >>> sampler = WeightedRandomSampler(
        ...     weights=sample_weights,
        ...     num_samples=len(train_df),
        ...     replacement=True
        ... )
    """
    # Compute inverse class frequency
    class_counts = df[disease_columns].sum().values
    class_weights = 1.0 / np.maximum(class_counts, 1)

    # Get label matrix
    labels = df[disease_columns].values  # Shape: (N, C)

    # Compute sample weights based on strategy
    if strategy == 'max_class_weight':
        # For each sample, take the max weight among positive classes
        weighted_labels = labels * class_weights  # Broadcast multiplication
        sample_weights = np.max(weighted_labels, axis=1)
        # For samples with no positive labels, use minimum weight
        no_positive_mask = labels.sum(axis=1) == 0
        sample_weights[no_positive_mask] = class_weights.min()

    elif strategy == 'sum_class_weight':
        # Sum of weights for all positive classes
        sample_weights = np.sum(labels * class_weights, axis=1)
        # Handle samples with no positive labels
        sample_weights[sample_weights == 0] = class_weights.min()

    elif strategy == 'mean_class_weight':
        # Mean weight across positive classes
        positive_counts = np.maximum(labels.sum(axis=1), 1)
        sample_weights = np.sum(labels * class_weights, axis=1) / positive_counts
        sample_weights[sample_weights == 0] = class_weights.min()

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Choose from: 'max_class_weight', 'sum_class_weight', 'mean_class_weight'"
        )

    return sample_weights


def create_loss_function(
    loss_name: str,
    pos_weight: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.

    Simplifies loss function creation with sensible defaults.

    Args:
        loss_name: Name of the loss function. Options:
            - 'focal': FocalLoss
            - 'weighted_bce': WeightedBCELoss
            - 'label_smoothing': LabelSmoothingBCE
            - 'asymmetric': AsymmetricLoss
            - 'combined': CombinedLoss
            - 'dice': DiceLoss
            - 'dice_bce': DiceBCELoss
            - 'bce': Standard BCEWithLogitsLoss
        pos_weight: Optional class weights
        **kwargs: Additional loss-specific parameters

    Returns:
        Configured loss function module

    Example:
        >>> weights = compute_class_weights(class_counts)
        >>> criterion = create_loss_function('combined', pos_weight=weights)
    """
    loss_name = loss_name.lower()

    if loss_name == 'focal':
        try:
            from .focal_loss import FocalLoss
        except ImportError:
            from focal_loss import FocalLoss
        return FocalLoss(pos_weight=pos_weight, **kwargs)

    elif loss_name == 'weighted_bce':
        try:
            from .weighted_loss import WeightedBCELoss
        except ImportError:
            from weighted_loss import WeightedBCELoss
        return WeightedBCELoss(pos_weight=pos_weight, **kwargs)

    elif loss_name == 'label_smoothing':
        try:
            from .smoothing_loss import LabelSmoothingBCE
        except ImportError:
            from smoothing_loss import LabelSmoothingBCE
        return LabelSmoothingBCE(pos_weight=pos_weight, **kwargs)

    elif loss_name == 'asymmetric' or loss_name == 'asl':
        try:
            from .asymmetric_loss import AsymmetricLoss
        except ImportError:
            from asymmetric_loss import AsymmetricLoss
        return AsymmetricLoss(**kwargs)

    elif loss_name == 'combined':
        try:
            from .combined_loss import CombinedLoss
        except ImportError:
            from combined_loss import CombinedLoss
        return CombinedLoss(pos_weight=pos_weight, **kwargs)

    elif loss_name == 'dice':
        try:
            from .dice_loss import DiceLoss
        except ImportError:
            from dice_loss import DiceLoss
        return DiceLoss(**kwargs)

    elif loss_name == 'dice_bce':
        try:
            from .dice_loss import DiceBCELoss
        except ImportError:
            from dice_loss import DiceBCELoss
        return DiceBCELoss(pos_weight=pos_weight, **kwargs)

    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight, **kwargs)

    else:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Options: 'focal', 'weighted_bce', 'label_smoothing', "
            f"'asymmetric', 'combined', 'dice', 'dice_bce', 'bce'"
        )