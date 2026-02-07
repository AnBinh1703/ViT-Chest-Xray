"""
Vision Transformer (ViT) for Chest X-ray Classification
========================================================

Vision Transformer implementation from scratch for multi-label chest X-ray classification.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron block for Transformer.

    Architecture: Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PatchEmbedding(nn.Module):
    """
    Patch Embedding layer for Vision Transformer.

    Splits image into patches and projects them to embedding dimension.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        """
        Initialize PatchEmbedding.

        Args:
            img_size: Input image size
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Use Conv2d as patch projection
        self.proj = nn.Conv2d(in_channels, embed_dim,
                             kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block.

    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> (+x) -> LayerNorm -> MLP -> (+x)
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, attn_dropout: float = 0.0):
        """
        Initialize Transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                         dropout=attn_dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-normalization."""
        # Self-attention with residual connection
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)

        # MLP with residual connection
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for multi-label image classification.

    Based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    (Dosovitskiy et al., 2020)
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 15,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 attn_dropout: float = 0.0, use_cls_token: bool = True,
                 classifier_hidden: Optional[list] = None):
        """
        Initialize Vision Transformer.

        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            attn_dropout: Attention dropout rate
            use_cls_token: Whether to use [CLS] token for classification
            classifier_hidden: Hidden layer sizes for classification head
        """
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Class token (optional)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_tokens = self.num_patches + 1
        else:
            self.cls_token = None
            num_tokens = self.num_patches

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])

        # Final layer norm
        self.ln_final = nn.LayerNorm(embed_dim, eps=1e-6)

        # Classification head
        if classifier_hidden is None:
            classifier_hidden = [2048, 1024]

        if use_cls_token:
            head_input_dim = embed_dim
        else:
            head_input_dim = embed_dim * self.num_patches

        layers = []
        prev_dim = head_input_dim
        for hidden_dim in classifier_hidden:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.head = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize class token
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize other weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize individual module weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output logits of shape (B, num_classes)
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add class token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_final(x)

        # Classification
        if self.use_cls_token:
            x = x[:, 0]  # Use [CLS] token
        else:
            x = x.flatten(1)  # Flatten all patches

        x = self.head(x)
        return x

    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Get attention maps from a specific layer.

        Args:
            x: Input tensor
            layer_idx: Index of transformer layer to get attention from

        Returns:
            Attention weights
        """
        batch_size = x.shape[0]

        # Patch embedding + class token + pos embedding
        x = self.patch_embed(x)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Process through transformer blocks until target layer
        attention_weights = None
        for i, block in enumerate(self.transformer_blocks):
            if i == layer_idx or (layer_idx < 0 and i == len(self.transformer_blocks) + layer_idx):
                # Get attention weights from this layer
                x_norm = block.ln1(x)
                _, attention_weights = block.attn(x_norm, x_norm, x_norm,
                                                  need_weights=True)
                x = x + block.attn(x_norm, x_norm, x_norm)[0]
                x = x + block.mlp(block.ln2(x))
            else:
                x = block(x)

        return attention_weights


class ViTSmall(VisionTransformer):
    """Small ViT configuration (ViT-S/32)."""

    def __init__(self, num_classes: int = 15, **kwargs):
        super().__init__(
            patch_size=32,
            embed_dim=64,
            depth=8,
            num_heads=4,
            mlp_ratio=4.0,
            num_classes=num_classes,
            use_cls_token=False,
            classifier_hidden=[2048, 1024],
            **kwargs
        )


class ViTBase(VisionTransformer):
    """Base ViT configuration (ViT-B/16)."""

    def __init__(self, num_classes: int = 15, **kwargs):
        super().__init__(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=num_classes,
            **kwargs
        )


class ViTLarge(VisionTransformer):
    """Large ViT configuration (ViT-L/16)."""

    def __init__(self, num_classes: int = 15, **kwargs):
        super().__init__(
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            num_classes=num_classes,
            **kwargs
        )


def create_vit_small(num_classes: int = 15, **kwargs) -> ViTSmall:
    """Create small ViT model (similar to original paper implementation)."""
    return ViTSmall(num_classes=num_classes, **kwargs)


def create_vit_base(num_classes: int = 15, **kwargs) -> ViTBase:
    """Create base ViT model."""
    return ViTBase(num_classes=num_classes, **kwargs)


def create_vit_large(num_classes: int = 15, **kwargs) -> ViTLarge:
    """Create large ViT model."""
    return ViTLarge(num_classes=num_classes, **kwargs)


def create_vit_model(variant: str = 'small', num_classes: int = 15,
                    pretrained: bool = False, **kwargs) -> VisionTransformer:
    """
    Factory function to create ViT models.

    Args:
        variant: ViT variant ('small', 'base', 'large')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments

    Returns:
        VisionTransformer model
    """
    variants = {
        'small': create_vit_small,
        'base': create_vit_base,
        'large': create_vit_large,
    }

    if variant.lower() not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")

    model = variants[variant.lower()](num_classes=num_classes, **kwargs)

    if pretrained:
        print(f"Note: Pretrained weights for ViT-{variant} not available in scratch implementation.")
        print("Use timm library for pretrained ViT models: timm.create_model('vit_base_patch16_224', pretrained=True)")

    return model


if __name__ == "__main__":
    # Test the models
    for variant in ['small', 'base']:
        model = create_vit_model(variant)
        params = sum(p.numel() for p in model.parameters())
        print(f"ViT-{variant}: {params:,} parameters")

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        print(f"  Input: {x.shape} -> Output: {output.shape}")
        print()