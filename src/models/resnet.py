"""
ResNet Model for Chest X-ray Classification
============================================

ResNet-34 implementation from scratch for multi-label chest X-ray classification.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34.

    Architecture:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+x) -> ReLU
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: nn.Module = None):
        """
        Initialize BasicBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first convolution
            downsample: Downsampling layer for residual connection
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet-50/101/152.

    Architecture:
        x -> Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN -> (+x) -> ReLU
    """
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: nn.Module = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet model for multi-label chest X-ray classification.

    Supports ResNet-18, 34, 50, 101, 152 architectures.
    """

    def __init__(self, block: nn.Module, layers: list, num_classes: int = 15,
                 in_channels: int = 3, dropout: float = 0.5):
        """
        Initialize ResNet.

        Args:
            block: Block type (BasicBlock or Bottleneck)
            layers: Number of blocks in each layer [layer1, layer2, layer3, layer4]
            num_classes: Number of output classes
            in_channels: Number of input channels
            dropout: Dropout rate before final layer
        """
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial conv layer (stem)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block: nn.Module, out_channels: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """Create a ResNet layer with multiple blocks."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature representation before classification head."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def create_resnet18(num_classes: int = 15, **kwargs) -> ResNet:
    """Create ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def create_resnet34(num_classes: int = 15, **kwargs) -> ResNet:
    """Create ResNet-34 model."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def create_resnet50(num_classes: int = 15, **kwargs) -> ResNet:
    """Create ResNet-50 model."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def create_resnet101(num_classes: int = 15, **kwargs) -> ResNet:
    """Create ResNet-101 model."""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)


def create_resnet_model(variant: str = 'resnet34', num_classes: int = 15,
                       pretrained: bool = False, **kwargs) -> ResNet:
    """
    Factory function to create ResNet models.

    Args:
        variant: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (not implemented for scratch models)
        **kwargs: Additional arguments passed to ResNet constructor

    Returns:
        ResNet model
    """
    variants = {
        'resnet18': create_resnet18,
        'resnet34': create_resnet34,
        'resnet50': create_resnet50,
        'resnet101': create_resnet101,
    }

    if variant.lower() not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")

    model = variants[variant.lower()](num_classes=num_classes, **kwargs)

    if pretrained:
        print(f"Note: Pretrained weights for {variant} not available in scratch implementation.")
        print("Use timm library for pretrained models.")

    return model


if __name__ == "__main__":
    # Test the models
    for variant in ['resnet18', 'resnet34', 'resnet50']:
        model = create_resnet_model(variant)
        params = sum(p.numel() for p in model.parameters())
        print(f"{variant}: {params:,} parameters")

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        print(f"  Input: {x.shape} -> Output: {output.shape}")
        print()