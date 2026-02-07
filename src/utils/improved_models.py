"""
Improved Model Architectures for Chest X-ray Classification
===========================================================

This module contains enhanced model architectures with transfer learning,
advanced regularization, and optimized components for medical image analysis.

Features:
- Transfer learning implementations
- Advanced dropout and regularization
- Attention mechanisms
- Multi-scale feature extraction
- Ensemble-ready architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from typing import Dict, List, Optional, Tuple
import math


class AdvancedResNet(nn.Module):
    """
    Enhanced ResNet with transfer learning and medical imaging optimizations.
    
    Features:
    - ImageNet pre-training
    - Dropout regularization
    - Advanced classifier head
    - Multi-scale feature extraction
    """
    
    def __init__(
        self, 
        num_classes: int = 15,
        backbone: str = 'resnet34',
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Load pre-trained backbone
        if backbone == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            backbone_features = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            backbone_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Remove original classifier
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Attention mechanism
        if use_attention:
            self.attention = SpatialAttention(backbone_features)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Advanced classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
        
    def _initialize_classifier(self):
        """Initialize classifier weights using Xavier initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.features(x)
        
        # Apply attention if enabled
        if self.use_attention:
            features = self.attention(features)
        
        # Global pooling
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        
        # Classification
        output = self.classifier(features)
        
        return output
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps for visualization."""
        return self.features(x)


class AdvancedViT(nn.Module):
    """
    Enhanced Vision Transformer for medical image classification.
    
    Features:
    - Pre-trained weights
    - Advanced dropout strategies
    - Medical image optimizations
    - Attention visualization support
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        use_classifier_dropout: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pre-trained ViT
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove head to add custom classifier
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            global_pool='avg'
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.model(dummy_input)
            feature_dim = features.shape[-1]
        
        # Custom classifier head
        if use_classifier_dropout:
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(drop_rate * 1.5),
                nn.Linear(feature_dim, 512),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(drop_rate / 2),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Initialize classifier
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.model(x)
        
        # Classification
        output = self.classifier(features)
        
        return output
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention maps for visualization."""
        attention_maps = []
        
        def hook_fn(module, input, output):
            attention_maps.append(output)
        
        # Register hooks on attention layers
        hooks = []
        for name, module in self.model.named_modules():
            if 'attn.softmax' in name:
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        _ = self.model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps


class EfficientNetAdvanced(nn.Module):
    """
    Enhanced EfficientNet for medical image classification.
    
    Features:
    - Multiple EfficientNet variants
    - Advanced augmentation support
    - Medical imaging optimizations
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        model_name: str = 'efficientnet_b3',
        pretrained: bool = True,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.2,
        use_multiscale: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_multiscale = use_multiscale
        
        # Load pre-trained EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove head
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            global_pool='avg'
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[-1]
        
        # Multi-scale feature extraction
        if use_multiscale:
            self.multiscale_pool = MultiScalePooling(feature_dim)
            classifier_input_dim = feature_dim * 3  # 3 different scales
        else:
            self.multiscale_pool = None
            classifier_input_dim = feature_dim
        
        # Advanced classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(classifier_input_dim),
            nn.Dropout(drop_rate),
            nn.Linear(classifier_input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(drop_rate / 2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(drop_rate / 4),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract backbone features
        features = self.backbone.forward_features(x)
        
        # Multi-scale pooling if enabled
        if self.use_multiscale:
            features = self.multiscale_pool(features)
        else:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
        
        # Classification
        output = self.classifier(features)
        
        return output


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for feature enhancement.
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x)
        return x * attention_weights


class MultiScalePooling(nn.Module):
    """
    Multi-scale global pooling for capturing features at different scales.
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.scales = [1, 2, 4]  # Different pooling scales
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(scale) for scale in self.scales
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        features = []
        
        for pool in self.pools:
            pooled = pool(x)
            pooled = torch.flatten(pooled, 1)
            features.append(pooled)
        
        # Concatenate all scales
        return torch.cat(features, dim=1)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved performance.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        voting_strategy: str = 'soft'  # 'soft' or 'hard'
    ):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.voting_strategy = voting_strategy
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            self.weights = weights
        
        # Normalize weights
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        
        for model in self.models:
            model.eval()  # Ensure models are in eval mode
            with torch.no_grad():
                output = model(x)
                outputs.append(output)
        
        if self.voting_strategy == 'soft':
            # Weighted average of softmax probabilities
            weighted_outputs = []
            for output, weight in zip(outputs, self.weights):
                prob = F.softmax(output, dim=1)
                weighted_outputs.append(prob * weight)
            
            ensemble_prob = torch.stack(weighted_outputs).sum(dim=0)
            return torch.log(ensemble_prob + 1e-8)  # Convert back to log-probabilities
            
        else:  # hard voting
            # Weighted average of logits
            weighted_outputs = []
            for output, weight in zip(outputs, self.weights):
                weighted_outputs.append(output * weight)
            
            return torch.stack(weighted_outputs).sum(dim=0)


class DenseNet121Medical(nn.Module):
    """
    DenseNet-121 optimized for medical imaging.
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        use_attention: bool = True
    ):
        super().__init__()
        
        # Load pre-trained DenseNet
        self.backbone = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Remove classifier
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # Add attention if specified
        if use_attention:
            self.attention = SpatialAttention(num_features)
        else:
            self.attention = None
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone.features(x)
        
        # Apply attention if available
        if self.attention is not None:
            features = self.attention(features)
        
        # Global pooling and classification
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        output = self.classifier(features)
        
        return output


def create_model(
    model_name: str,
    num_classes: int = 15,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different model architectures.
    
    Args:
        model_name: Name of the model ('resnet34', 'vit_base', 'efficientnet_b3', etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights
        **kwargs: Additional model-specific arguments
    
    Returns:
        Configured model instance
    """
    
    model_configs = {
        'resnet34': lambda: AdvancedResNet(
            num_classes=num_classes,
            backbone='resnet34',
            pretrained=pretrained,
            **kwargs
        ),
        'resnet50': lambda: AdvancedResNet(
            num_classes=num_classes,
            backbone='resnet50',
            pretrained=pretrained,
            **kwargs
        ),
        'vit_base': lambda: AdvancedViT(
            num_classes=num_classes,
            model_name='vit_base_patch16_224',
            pretrained=pretrained,
            **kwargs
        ),
        'vit_small': lambda: AdvancedViT(
            num_classes=num_classes,
            model_name='vit_small_patch16_224',
            pretrained=pretrained,
            **kwargs
        ),
        'efficientnet_b3': lambda: EfficientNetAdvanced(
            num_classes=num_classes,
            model_name='efficientnet_b3',
            pretrained=pretrained,
            **kwargs
        ),
        'efficientnet_b4': lambda: EfficientNetAdvanced(
            num_classes=num_classes,
            model_name='efficientnet_b4',
            pretrained=pretrained,
            **kwargs
        ),
        'densenet121': lambda: DenseNet121Medical(
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    }
    
    if model_name not in model_configs:
        available_models = list(model_configs.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
    
    return model_configs[model_name]()


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Returns:
        (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...")
    
    models_to_test = [
        'resnet34', 'vit_base', 'efficientnet_b3', 'densenet121'
    ]
    
    for model_name in models_to_test:
        print(f"\n{model_name}:")
        model = create_model(model_name, num_classes=15)
        total, trainable = count_parameters(model)
        
        print(f"  Total parameters: {total:,}")
        print(f"  Trainable parameters: {trainable:,}")
        
        # Test forward pass
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 224, 224)
            output = model(dummy_input)
            print(f"  Output shape: {output.shape}")
    
    print("\n✅ All models created successfully!")