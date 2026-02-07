#!/usr/bin/env python3
"""
ViT Chest X-ray Classification - Main Entry Point
==================================================

Unified command-line interface for all project operations.

Usage:
    # Training
    python main.py train --config configs/vit_small.yaml
    
    # Evaluation
    python main.py evaluate --checkpoint outputs/best_model.pth --data test
    
    # Inference on single image
    python main.py predict --checkpoint outputs/best_model.pth --image path/to/xray.png
    
    # List available models
    python main.py models --list
    
    # Verify installation and setup
    python main.py verify

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_train(args):
    """Run training with specified config."""
    from scripts.train import main as train_main
    import sys
    
    # Rebuild sys.argv for train script
    sys.argv = ['train.py', '--config', args.config]
    if args.resume:
        sys.argv.extend(['--resume', args.resume])
    if args.overrides:
        sys.argv.extend(args.overrides)
    
    train_main()


def cmd_verify(args):
    """Verify installation and print system info."""
    print("=" * 60)
    print("ViT Chest X-ray Classification - System Verification")
    print("=" * 60)
    
    # Check Python version
    import sys
    print(f"\n✓ Python: {sys.version}")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch: NOT INSTALLED")
    
    # Check NumPy
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        print("✗ NumPy: NOT INSTALLED")
    
    # Check other dependencies
    deps = ['pandas', 'sklearn', 'PIL', 'yaml', 'tqdm']
    for dep in deps:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'installed')
            print(f"✓ {dep}: {version}")
        except ImportError:
            print(f"✗ {dep}: NOT INSTALLED")
    
    # Check project structure
    print("\n--- Project Structure ---")
    required_dirs = ['src', 'configs', 'scripts', 'notebooks']
    for d in required_dirs:
        path = PROJECT_ROOT / d
        status = "✓" if path.exists() else "✗"
        print(f"{status} {d}/")
    
    # Check models import
    print("\n--- Model Imports ---")
    try:
        from src.models import create_cnn_model, create_resnet_model, create_vit_model
        print("✓ All model factories importable")
    except ImportError as e:
        print(f"✗ Import error: {e}")
    
    # Check data imports
    print("\n--- Data Imports ---")
    try:
        from src.data import patient_level_split, ChestXrayTransform
        print("✓ Data modules importable")
    except ImportError as e:
        print(f"✗ Import error: {e}")
    
    # Check utils imports
    print("\n--- Utils Imports ---")
    try:
        from src.utils import set_seed, load_config, EarlyStopping
        print("✓ Utils modules importable")
    except ImportError as e:
        print(f"✗ Import error: {e}")
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


def cmd_models(args):
    """List or test available models."""
    from src.models import create_cnn_model, create_resnet_model, create_vit_model
    import torch
    
    print("=" * 60)
    print("Available Models")
    print("=" * 60)
    
    models = {
        'CNN Baseline': lambda: create_cnn_model(num_classes=15),
        'ResNet-18': lambda: create_resnet_model('resnet18', num_classes=15),
        'ResNet-34': lambda: create_resnet_model('resnet34', num_classes=15),
        'ResNet-50': lambda: create_resnet_model('resnet50', num_classes=15),
        'ViT-Small': lambda: create_vit_model('small', num_classes=15),
        'ViT-Base': lambda: create_vit_model('base', num_classes=15),
    }
    
    if args.test:
        print("\nTesting model forward passes...\n")
        x = torch.randn(1, 3, 224, 224)
        
        for name, factory in models.items():
            try:
                model = factory()
                model.eval()
                with torch.no_grad():
                    out = model(x)
                params = sum(p.numel() for p in model.parameters())
                print(f"✓ {name:15} | Params: {params:>12,} | Output: {tuple(out.shape)}")
            except Exception as e:
                print(f"✗ {name:15} | Error: {e}")
    else:
        print("\nBuilt-in models:")
        for name in models.keys():
            print(f"  - {name}")
        
        # Try to list pretrained models
        try:
            from src.models.pretrained import list_available_models
            pretrained = list_available_models()
            print(f"\nPretrained models available: {len(pretrained)}")
            print("  (requires timm library)")
        except ImportError:
            pass
        
        print("\nUse --test to verify models work correctly")


def cmd_evaluate(args):
    """Evaluate a trained model."""
    print("Evaluation command - to be implemented")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data split: {args.data}")


def cmd_predict(args):
    """Run inference on a single image."""
    print("Prediction command - to be implemented")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image: {args.image}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ViT Chest X-ray Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py verify                           # Check setup
    python main.py models --test                    # Test all models
    python main.py train --config configs/vit.yaml  # Train a model
    
For more help on each command:
    python main.py <command> --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, required=True, help='Config file path')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.add_argument('overrides', nargs='*', help='Config overrides')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    eval_parser.add_argument('--data', type=str, default='test', choices=['train', 'val', 'test'])
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict on single image')
    predict_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    predict_parser.add_argument('--image', type=str, required=True, help='Image path')
    
    # Models command
    models_parser = subparsers.add_parser('models', help='List available models')
    models_parser.add_argument('--test', action='store_true', help='Test model creation')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify installation')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Route to appropriate command
    commands = {
        'train': cmd_train,
        'evaluate': cmd_evaluate,
        'predict': cmd_predict,
        'models': cmd_models,
        'verify': cmd_verify,
    }
    
    commands[args.command](args)


if __name__ == '__main__':
    main()
