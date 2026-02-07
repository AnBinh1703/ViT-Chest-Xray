"""
CNN Model for Chest X-ray Classification
=========================================

Baseline CNN architecture with 2 convolutional layers.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    """
    Simple CNN classifier for multi-label chest X-ray classification.

    Architecture:
    - Conv2d(3, 32, 3) -> ReLU -> MaxPool2d(2)
    - Conv2d(32, 64, 3) -> ReLU -> MaxPool2d(2)
    - Flatten -> Linear(64*54*54, 512) -> ReLU -> Linear(512, num_classes)
    """

    def __init__(self, num_classes=15, input_size=224):
        """
        Initialize CNN classifier.

        Args:
            num_classes (int): Number of output classes
            input_size (int): Input image size (assumed square)
        """
        super(CNNClassifier, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv Block 1: 3x224x224 -> 32x111x111
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 32x111x111 -> 32x55x55

            # Conv Block 2: 32x55x55 -> 64x54x54
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # 64x54x54 -> 64x27x27
        )

        # Calculate flattened size after convolutions
        # Input: 224x224 -> Conv(3x3, pad=0) -> 222x222 -> Pool(2) -> 111x111
        # -> Conv(3x3, pad=0) -> 109x109 -> Pool(2) -> 54x54
        self.flatten_size = 64 * 54 * 54

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Added dropout for regularization
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Feature maps after convolution layers
        """
        return self.features(x)


def create_cnn_model(num_classes=15, pretrained=False):
    """
    Factory function to create CNN model.

    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights (not applicable for this simple CNN)

    Returns:
        CNNClassifier: Initialized model
    """
    model = CNNClassifier(num_classes=num_classes)
    return model


if __name__ == "__main__":
    # Test the model
    model = create_cnn_model()
    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")