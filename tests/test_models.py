"""
Unit Tests for Model Architectures
==================================

Tests for CNN, ResNet, and ViT model implementations.

Run with: python -m pytest tests/test_models.py -v

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestCNNModel:
    """Tests for CNN model."""

    def test_cnn_import(self):
        """Test CNN model can be imported."""
        from models.cnn import CNNClassifier, create_cnn_model
        assert CNNClassifier is not None
        assert create_cnn_model is not None

    def test_cnn_creation(self):
        """Test CNN model creation."""
        from models.cnn import create_cnn_model

        model = create_cnn_model(num_classes=15)
        assert model is not None

    def test_cnn_forward_pass(self):
        """Test CNN forward pass."""
        from models.cnn import create_cnn_model

        model = create_cnn_model(num_classes=15)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        assert output.shape == (2, 15)

    def test_cnn_parameter_count(self):
        """Test CNN has expected number of parameters."""
        from models.cnn import create_cnn_model

        model = create_cnn_model(num_classes=15)
        params = sum(p.numel() for p in model.parameters())

        # Should have a reasonable number of parameters
        assert params > 0
        assert params < 100_000_000  # Less than 100M


class TestResNetModel:
    """Tests for ResNet models."""

    def test_resnet_import(self):
        """Test ResNet model can be imported."""
        from models.resnet import ResNet, create_resnet_model
        assert ResNet is not None
        assert create_resnet_model is not None

    @pytest.mark.parametrize("variant", ["resnet18", "resnet34", "resnet50"])
    def test_resnet_creation(self, variant):
        """Test ResNet model creation for different variants."""
        from models.resnet import create_resnet_model

        model = create_resnet_model(variant=variant, num_classes=15)
        assert model is not None

    @pytest.mark.parametrize("variant", ["resnet18", "resnet34"])
    def test_resnet_forward_pass(self, variant):
        """Test ResNet forward pass."""
        from models.resnet import create_resnet_model

        model = create_resnet_model(variant=variant, num_classes=15)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        assert output.shape == (2, 15)

    def test_resnet_feature_extraction(self):
        """Test ResNet feature extraction."""
        from models.resnet import create_resnet_model

        model = create_resnet_model(variant="resnet18", num_classes=15)
        x = torch.randn(2, 3, 224, 224)
        features = model.get_features(x)

        assert features.shape == (2, 512)

    def test_resnet_unknown_variant(self):
        """Test ResNet raises error for unknown variant."""
        from models.resnet import create_resnet_model

        with pytest.raises(ValueError):
            create_resnet_model(variant="resnet999")


class TestViTModel:
    """Tests for Vision Transformer models."""

    def test_vit_import(self):
        """Test ViT model can be imported."""
        from models.vit import VisionTransformer, create_vit_model
        assert VisionTransformer is not None
        assert create_vit_model is not None

    @pytest.mark.parametrize("variant", ["small", "base"])
    def test_vit_creation(self, variant):
        """Test ViT model creation for different variants."""
        from models.vit import create_vit_model

        model = create_vit_model(variant=variant, num_classes=15)
        assert model is not None

    def test_vit_small_forward_pass(self):
        """Test ViT-Small forward pass."""
        from models.vit import create_vit_model

        model = create_vit_model(variant="small", num_classes=15)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        assert output.shape == (2, 15)

    def test_patch_embedding(self):
        """Test patch embedding layer."""
        from models.vit import PatchEmbedding

        patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
        x = torch.randn(2, 3, 224, 224)
        output = patch_embed(x)

        expected_patches = (224 // 16) ** 2  # 196
        assert output.shape == (2, expected_patches, 768)

    def test_vit_unknown_variant(self):
        """Test ViT raises error for unknown variant."""
        from models.vit import create_vit_model

        with pytest.raises(ValueError):
            create_vit_model(variant="huge")


class TestModelGradients:
    """Tests for gradient computation."""

    def test_cnn_gradients(self):
        """Test CNN gradients can be computed."""
        from models.cnn import create_cnn_model

        model = create_cnn_model(num_classes=15)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_resnet_gradients(self):
        """Test ResNet gradients can be computed."""
        from models.resnet import create_resnet_model

        model = create_resnet_model(variant="resnet18", num_classes=15)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])