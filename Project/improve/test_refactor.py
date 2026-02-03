#!/usr/bin/env python3
"""
Test script for the refactored loss functions package.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Test imports
try:
    from config import LossConfig
    from focal_loss import FocalLoss
    from combined_loss import CombinedLoss
    from utils import compute_class_weights
    print("✅ All imports successful!")
    print(f"LossConfig.FOCAL_ALPHA = {LossConfig.FOCAL_ALPHA}")
    print(f"FocalLoss available: {FocalLoss}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test functionality
try:
    import torch
    import numpy as np

    # Test class weights
    class_counts = [100, 50, 25, 10]
    weights = compute_class_weights(class_counts, method='effective')
    print(f"Class weights computed: {weights}")

    # Test loss function creation
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    print(f"Loss function created: {criterion}")

    print("✅ All functionality tests passed!")

except Exception as e:
    print(f"❌ Functionality test failed: {e}")
    sys.exit(1)