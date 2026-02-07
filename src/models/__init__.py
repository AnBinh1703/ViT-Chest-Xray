"""
Model Architectures Module
==========================

Contains implementations of various model architectures for chest X-ray classification.

Available models:
- CNN: Baseline convolutional neural network
- ResNet: Residual networks (ResNet-18, 34, 50, 101)
- ViT: Vision Transformer models (Small, Base, Large)

Author: ViT-Chest-Xray Project Team
"""

from .cnn import CNNClassifier, create_cnn_model
from .resnet import (
    ResNet, BasicBlock, Bottleneck,
    create_resnet18, create_resnet34, create_resnet50, create_resnet101,
    create_resnet_model
)
from .vit import (
    VisionTransformer, PatchEmbedding, TransformerEncoderBlock, MLP,
    ViTSmall, ViTBase, ViTLarge,
    create_vit_small, create_vit_base, create_vit_large, create_vit_model
)
from .pretrained import (
    create_pretrained_model, list_available_models, get_model_info,
    PretrainedModelWrapper
)

__all__ = [
    # CNN
    'CNNClassifier', 'create_cnn_model',
    # ResNet
    'ResNet', 'BasicBlock', 'Bottleneck',
    'create_resnet18', 'create_resnet34', 'create_resnet50', 'create_resnet101',
    'create_resnet_model',
    # ViT
    'VisionTransformer', 'PatchEmbedding', 'TransformerEncoderBlock', 'MLP',
    'ViTSmall', 'ViTBase', 'ViTLarge',
    'create_vit_small', 'create_vit_base', 'create_vit_large', 'create_vit_model',
    # Pretrained
    'create_pretrained_model', 'list_available_models', 'get_model_info',
    'PretrainedModelWrapper',
]