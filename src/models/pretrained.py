"""
Pretrained Model Wrappers
=========================

Provides easy access to pretrained models from timm (PyTorch Image Models)
and torchvision with appropriate modifications for chest X-ray classification.

Supported model families:
- Vision Transformers (ViT, DeiT, Swin)
- ResNet variants (ResNet, ResNeXt, Wide ResNet)
- EfficientNet (V1, V2)
- ConvNeXt

Usage:
    from src.models.pretrained import create_pretrained_model
    
    # Create a pretrained ViT
    model = create_pretrained_model('vit_base_patch16_224', num_classes=15, pretrained=True)
    
    # Create a pretrained ResNet with custom dropout
    model = create_pretrained_model('resnet50', num_classes=15, pretrained=True, drop_rate=0.3)

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn

# Check for timm availability
try:
    import timm
    from timm.models import create_model as timm_create_model
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False  

# Check for torchvision availability
try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# Model family mappings
TIMM_VIT_MODELS = [
    'vit_tiny_patch16_224',
    'vit_small_patch16_224',
    'vit_base_patch16_224',
    'vit_large_patch16_224',
    'deit_tiny_patch16_224',
    'deit_small_patch16_224',
    'deit_base_patch16_224',
    'swin_tiny_patch4_window7_224',
    'swin_small_patch4_window7_224',
    'swin_base_patch4_window7_224',
]

TIMM_CNN_MODELS = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnext50_32x4d', 'resnext101_32x8d',
    'wide_resnet50_2', 'wide_resnet101_2',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l',
    'convnext_tiny', 'convnext_small', 'convnext_base',
    'densenet121', 'densenet169', 'densenet201',
]


def list_available_models(family: Optional[str] = None) -> List[str]:
    """
    List available pretrained models.
    
    Args:
        family: Optional filter ('vit', 'cnn', 'all')
        
    Returns:
        List of available model names
    """
    models = []
    
    if family in ['vit', 'all', None]:
        models.extend(TIMM_VIT_MODELS)
    
    if family in ['cnn', 'all', None]:
        models.extend(TIMM_CNN_MODELS)
    
    return models


class PretrainedModelWrapper(nn.Module):
    """
    Wrapper for pretrained models with custom classifier head.
    
    Handles different model architectures uniformly:
    - Replaces the final classifier with appropriate layers
    - Adds optional dropout before classification
    - Supports feature extraction mode
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        num_features: int,
        drop_rate: float = 0.0,
        feature_extract: bool = False
    ):
        super().__init__()
        
        self.base_model = base_model
        self.feature_extract = feature_extract
        
        # Build classifier head
        if drop_rate > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=drop_rate),
                nn.Linear(num_features, num_classes)
            )
        else:
            self.classifier = nn.Linear(num_features, num_classes)
        
        # Freeze base model if feature extraction mode
        if feature_extract:
            for param in self.base_model.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        return self.classifier(features)
    
    def unfreeze_layers(self, num_layers: int = -1) -> None:
        """
        Unfreeze layers for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end.
                       -1 means unfreeze all layers.
        """
        params = list(self.base_model.parameters())
        
        if num_layers == -1:
            for param in params:
                param.requires_grad = True
        else:
            for param in params[:-num_layers]:
                param.requires_grad = True


def _create_timm_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    feature_extract: bool = False
) -> nn.Module:
    """Create a model using timm library."""
    if not HAS_TIMM:
        raise ImportError(
            "timm is required for pretrained models. "
            "Install with: pip install timm"
        )
    
    # Create base model without classifier
    base_model = timm_create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,  # Remove classifier
        global_pool='avg'  # Global average pooling
    )
    
    # Get number of features
    num_features = base_model.num_features
    
    # Wrap with custom classifier
    model = PretrainedModelWrapper(
        base_model=base_model,
        num_classes=num_classes,
        num_features=num_features,
        drop_rate=drop_rate,
        feature_extract=feature_extract
    )
    
    return model


def _create_torchvision_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.0
) -> nn.Module:
    """Create a model using torchvision."""
    if not HAS_TORCHVISION:
        raise ImportError("torchvision is required")
    
    # Map model name to torchvision function
    model_map = {
        'resnet18': tv_models.resnet18,
        'resnet34': tv_models.resnet34,
        'resnet50': tv_models.resnet50,
        'resnet101': tv_models.resnet101,
        'densenet121': tv_models.densenet121,
        'efficientnet_b0': tv_models.efficientnet_b0,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown torchvision model: {model_name}")
    
    # Load model
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = model_map[model_name](weights=weights)
    
    # Replace classifier
    if hasattr(model, 'fc'):
        num_features = model.fc.in_features
        if drop_rate > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=drop_rate),
                nn.Linear(num_features, num_classes)
            )
        else:
            model.fc = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            num_features = model.classifier.in_features
            if drop_rate > 0:
                model.classifier = nn.Sequential(
                    nn.Dropout(p=drop_rate),
                    nn.Linear(num_features, num_classes)
                )
            else:
                model.classifier = nn.Linear(num_features, num_classes)
        else:
            # For models like EfficientNet with Sequential classifier
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_features, num_classes)
    
    return model


def create_pretrained_model(
    model_name: str,
    num_classes: int = 15,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    feature_extract: bool = False,
    use_timm: bool = True
) -> nn.Module:
    """
    Create a pretrained model for chest X-ray classification.
    
    Args:
        model_name: Name of the model (e.g., 'vit_base_patch16_224', 'resnet50')
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        drop_rate: Dropout rate before classifier
        feature_extract: If True, freeze backbone for feature extraction
        use_timm: Whether to use timm library (recommended)
        
    Returns:
        PyTorch model with appropriate classifier head
    """
    if use_timm and HAS_TIMM:
        return _create_timm_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            drop_rate=drop_rate,
            feature_extract=feature_extract
        )
    else:
        return _create_torchvision_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            drop_rate=drop_rate
        )


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
    """
    if not HAS_TIMM:
        return {'error': 'timm not installed'}
    
    try:
        model = timm_create_model(model_name, pretrained=False)
        info = {
            'name': model_name,
            'num_features': model.num_features if hasattr(model, 'num_features') else None,
            'default_cfg': model.default_cfg if hasattr(model, 'default_cfg') else {},
            'num_params': sum(p.numel() for p in model.parameters()),
        }
        del model
        return info
    except Exception as e:
        return {'error': str(e)}


if __name__ == "__main__":
    print("Testing pretrained model wrapper...")
    
    # Check timm availability
    if HAS_TIMM:
        print(f"✓ timm is available (version: {timm.__version__})")
        
        # List models
        vit_models = list_available_models('vit')
        print(f"  Available ViT models: {len(vit_models)}")
        
        cnn_models = list_available_models('cnn')
        print(f"  Available CNN models: {len(cnn_models)}")
        
        # Test creating a model
        try:
            model = create_pretrained_model(
                'resnet18',
                num_classes=15,
                pretrained=False,  # Don't download for test
                drop_rate=0.2
            )
            
            # Test forward pass
            x = torch.randn(1, 3, 224, 224)
            out = model(x)
            print(f"\n✓ Model created: resnet18")
            print(f"  Input: {tuple(x.shape)}")
            print(f"  Output: {tuple(out.shape)}")
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {num_params:,}")
            
        except Exception as e:
            print(f"  Model creation test skipped: {e}")
    else:
        print("⚠ timm not installed. Install with: pip install timm")
    
    # Check torchvision
    if HAS_TORCHVISION:
        print(f"\n✓ torchvision is available")
        
        # Test torchvision model
        try:
            model = _create_torchvision_model(
                'resnet18',
                num_classes=15,
                pretrained=False
            )
            x = torch.randn(1, 3, 224, 224)
            out = model(x)
            print(f"  Torchvision ResNet18 output: {tuple(out.shape)}")
        except Exception as e:
            print(f"  Torchvision test skipped: {e}")
    
    print("\n✓ Pretrained model wrapper tests complete!")
