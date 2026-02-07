"""
Main Configuration File for ViT Chest X-ray Classification Project
===================================================================

This file consolidates all configuration parameters for the project.
It imports base configurations and extends them with project-specific settings.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import base configurations
from utils.config import *
from losses.config import LossConfig

class ProjectConfig:
    """Main project configuration class."""

    # Project metadata
    PROJECT_NAME = "ViT-Chest-Xray"
    VERSION = "2.0.0"
    AUTHOR = "ViT-Chest-Xray Team"

    # Dataset configuration
    DATASET_NAME = "NIH ChestX-ray14"
    NUM_IMAGES = 112120
    NUM_PATIENTS = 30805
    NUM_CLASSES = 15

    # Model configurations
    MODELS = {
        'cnn': {
            'name': 'CNN Baseline',
            'parameters': '~95M',
            'architecture': '2 Conv Layers'
        },
        'resnet34': {
            'name': 'ResNet-34',
            'parameters': '~21M',
            'architecture': 'ResNet-34 from scratch'
        },
        'vit_v1': {
            'name': 'ViT-v1',
            'parameters': '~9M',
            'architecture': 'Vision Transformer v1'
        },
        'vit_v2': {
            'name': 'ViT-v2',
            'parameters': '~9M',
            'architecture': 'Vision Transformer v2'
        },
        'vit_pretrained': {
            'name': 'ViT Pretrained',
            'parameters': '~86M',
            'architecture': 'timm ViT with ImageNet weights'
        }
    }

    # Training configurations
    TRAINING_CONFIGS = {
        'default': {
            'learning_rate': 1e-4,
            'weight_decay': 1e-6,
            'num_epochs': 10,
            'batch_size': 32,
            'optimizer': 'Adam',
            'scheduler': None
        },
        'improved': {
            'learning_rate': 1e-4,
            'weight_decay': 1e-6,
            'num_epochs': 20,
            'batch_size': 16,
            'optimizer': 'AdamW',
            'scheduler': 'cosine'
        }
    }

    # Loss function configurations
    LOSS_CONFIGS = {
        'bce': 'Binary Cross Entropy',
        'focal': 'Focal Loss',
        'weighted_bce': 'Weighted BCE',
        'combined': 'Combined Loss',
        'dice': 'Dice Loss',
        'asymmetric': 'Asymmetric Loss'
    }

    # Data augmentation configurations
    AUGMENTATION_CONFIGS = {
        'basic': ['resize', 'normalize'],
        'advanced': ['resize', 'rotate', 'flip', 'brightness', 'contrast', 'normalize'],
        'medical': ['resize', 'clahe', 'normalize']
    }

    # Evaluation metrics
    METRICS = ['auc_macro', 'auc_micro', 'accuracy', 'precision', 'recall', 'f1']

    # File paths (inherited from base config)
    @property
    def data_root(self):
        return DATA_ROOT

    @property
    def images_dir(self):
        return IMAGES_DIR

    @property
    def labels_csv(self):
        return LABELS_CSV

    @property
    def checkpoints_dir(self):
        return CHECKPOINTS_DIR

    @property
    def processed_data_dir(self):
        return PROCESSED_DATA_DIR

# Global configuration instance
config = ProjectConfig()

# Loss configuration instance
loss_config = LossConfig()

def print_full_config():
    """Print the complete project configuration."""
    print("=" * 60)
    print("ViT Chest X-ray Classification - Full Configuration")
    print("=" * 60)
    print(f"Project: {config.PROJECT_NAME} v{config.VERSION}")
    print(f"Dataset: {config.DATASET_NAME} ({config.NUM_IMAGES} images, {config.NUM_PATIENTS} patients)")
    print(f"Classes: {config.NUM_CLASSES}")
    print()
    print("Paths:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Data Root: {config.data_root}")
    print(f"  Images: {config.images_dir}")
    print(f"  Labels: {config.labels_csv}")
    print(f"  Checkpoints: {config.checkpoints_dir}")
    print(f"  Processed Data: {config.processed_data_dir}")
    print()
    print("Available Models:")
    for model_key, model_info in config.MODELS.items():
        print(f"  {model_key}: {model_info['name']} ({model_info['parameters']} params)")
    print()
    print("Training Configs:")
    for config_key, train_config in config.TRAINING_CONFIGS.items():
        print(f"  {config_key}: {train_config['optimizer']}, {train_config['batch_size']} batch, {train_config['num_epochs']} epochs")
    print("=" * 60)

if __name__ == "__main__":
    print_full_config()