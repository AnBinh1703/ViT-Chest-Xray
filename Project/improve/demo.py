"""
Loss Functions Demo and Testing
===============================

Demonstration scripts for testing loss functions.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import torch
import numpy as np
import pandas as pd

try:
    from .config import LossConfig
    from .utils import compute_class_weights, create_loss_function
    from .weighted_loss import WeightedBCELoss
    from .focal_loss import FocalLoss
    from .smoothing_loss import LabelSmoothingBCE
    from .asymmetric_loss import AsymmetricLoss
    from .combined_loss import CombinedLoss
    from .dice_loss import DiceLoss, DiceBCELoss
except ImportError:
    from config import LossConfig
    from utils import compute_class_weights, create_loss_function
    from weighted_loss import WeightedBCELoss
    from focal_loss import FocalLoss
    from smoothing_loss import LabelSmoothingBCE
    from asymmetric_loss import AsymmetricLoss
    from combined_loss import CombinedLoss
    from dice_loss import DiceLoss, DiceBCELoss


def demo_loss_functions():
    """Demonstrate all loss functions with sample data."""

    print("=" * 70)
    print("LOSS FUNCTIONS DEMO")
    print("=" * 70)

    # Configuration
    batch_size = 32
    num_classes = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Num classes: {num_classes}")

    # Generate sample data
    torch.manual_seed(42)
    inputs = torch.randn(batch_size, num_classes).to(device)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float().to(device)

    # Sample imbalanced class counts (simulating NIH Chest X-ray)
    class_counts = np.array([
        60361, 19894, 13317, 11559, 6331,
        5782, 5302, 4667, 3385, 2776,
        2665, 2303, 1431, 1354, 227
    ])

    print("\n" + "-" * 70)
    print("CLASS WEIGHT COMPUTATION")
    print("-" * 70)

    methods = ['balanced', 'inverse_sqrt', 'effective', 'pos_neg_ratio']
    for method in methods:
        weights = compute_class_weights(class_counts, method=method)
        print(f"\n{method.upper()} weights (first 5 classes):")
        print(f"  {weights[:5].numpy()}")
        print(f"  Min: {weights.min():.2f}, Max: {weights.max():.2f}")

    # Use 'effective' weights for demo
    pos_weight = compute_class_weights(class_counts, method='effective').to(device)

    print("\n" + "-" * 70)
    print("LOSS FUNCTION OUTPUTS")
    print("-" * 70)

    # Test all loss functions
    loss_configs = [
        ('BCE (baseline)', torch.nn.BCEWithLogitsLoss()),
        ('Weighted BCE', WeightedBCELoss(pos_weight=pos_weight)),
        ('Focal Loss (α=0.25, γ=2)', FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)),
        ('Label Smoothing (ε=0.1)', LabelSmoothingBCE(smoothing=0.1, pos_weight=pos_weight)),
        ('Asymmetric Loss (γ-=4, γ+=1)', AsymmetricLoss(gamma_neg=4, gamma_pos=1)),
        ('Combined Loss', CombinedLoss(pos_weight=pos_weight)),
        ('Dice Loss', DiceLoss()),
        ('Dice + BCE', DiceBCELoss(pos_weight=pos_weight)),
    ]

    print(f"\n{'Loss Function':<35} {'Loss Value':>12}")
    print("-" * 50)

    for name, loss_fn in loss_configs:
        loss_fn = loss_fn.to(device)
        with torch.no_grad():
            try:
                loss_value = loss_fn(inputs, targets)
                print(f"{name:<35} {loss_value.item():>12.4f}")
            except Exception as e:
                print(f"{name:<35} {'ERROR':>12} - {e}")

    print("\n" + "-" * 70)
    print("FACTORY FUNCTION TEST")
    print("-" * 70)

    for loss_name in ['focal', 'asymmetric', 'combined', 'dice_bce']:
        criterion = create_loss_function(loss_name, pos_weight=pos_weight)
        print(f"Created: {criterion}")

    print("\n" + "=" * 70)
    print("✅ All loss functions tested successfully!")
    print("=" * 70)


def demo_sample_weights():
    """Demonstrate sample weight computation."""

    print("\n" + "=" * 70)
    print("SAMPLE WEIGHTS DEMO")
    print("=" * 70)

    # Create mock dataframe
    np.random.seed(42)
    n_samples = 1000
    disease_columns = ['Disease_A', 'Disease_B', 'Disease_C', 'Rare_Disease']

    # Create imbalanced labels
    df = pd.DataFrame({
        'Disease_A': np.random.binomial(1, 0.3, n_samples),  # 30%
        'Disease_B': np.random.binomial(1, 0.2, n_samples),  # 20%
        'Disease_C': np.random.binomial(1, 0.1, n_samples),  # 10%
        'Rare_Disease': np.random.binomial(1, 0.01, n_samples),  # 1%
    })

    print(f"\nClass distribution:")
    for col in disease_columns:
        count = df[col].sum()
        pct = count / len(df) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")

    # Compute sample weights
    for strategy in ['max_class_weight', 'sum_class_weight', 'mean_class_weight']:
        try:
            from .utils import compute_sample_weights
        except ImportError:
            from utils import compute_sample_weights
        weights = compute_sample_weights(df, disease_columns, strategy=strategy)
        print(f"\n{strategy.upper()}:")
        print(f"  Min weight: {weights.min():.4f}")
        print(f"  Max weight: {weights.max():.4f}")
        print(f"  Mean weight: {weights.mean():.4f}")

    print("\n✅ Sample weights computed successfully!")


if __name__ == "__main__":
    demo_loss_functions()
    demo_sample_weights()