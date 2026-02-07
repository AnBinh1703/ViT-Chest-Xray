"""
Loss Function Configuration
===========================

Default configuration parameters for all loss functions.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

class LossConfig:
    """Default configuration for loss functions."""

    # Focal Loss defaults
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # Label Smoothing defaults
    SMOOTHING_FACTOR = 0.1

    # Asymmetric Loss defaults
    ASL_GAMMA_NEG = 4.0
    ASL_GAMMA_POS = 1.0
    ASL_CLIP = 0.05

    # Combined Loss defaults
    COMBINED_FOCAL_WEIGHT = 0.7
    COMBINED_BCE_WEIGHT = 0.2
    COMBINED_SMOOTH_WEIGHT = 0.1

    # Class weight bounds
    MIN_WEIGHT = 0.5
    MAX_WEIGHT = 100.0

    # Effective number beta
    EFFECTIVE_BETA = 0.9999