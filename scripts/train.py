#!/usr/bin/env python3
"""
Training Script for Chest X-ray Classification
===============================================

Command-line interface for training models with YAML configuration.

Usage:
    # Train with config file
    python scripts/train.py --config configs/vit_small.yaml
    
    # Override config values
    python scripts/train.py --config configs/vit_small.yaml training.lr=0.001 training.epochs=50
    
    # Resume training
    python scripts/train.py --config configs/vit_small.yaml --resume checkpoints/last.pth
    
    # Evaluate only
    python scripts/train.py --config configs/vit_small.yaml --eval --checkpoint checkpoints/best.pth

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from src.utils.config_loader import load_config, parse_cli_overrides, save_config
from src.utils.reproducibility import set_seed, get_reproducibility_info, seed_worker, get_dataloader_generator
from src.utils.callbacks import (
    CallbackList, EarlyStopping, ModelCheckpoint, 
    MetricsHistory, ProgressLogger
)
from src.data.transforms import create_transforms
from src.data.splits import patient_level_split, load_saved_split
from src.models.cnn import create_cnn_model
from src.models.resnet import create_resnet_model
from src.models.vit import create_vit_model


def setup_logging(output_dir: Path, level: str = 'INFO') -> None:
    """Configure logging to console and file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )


def create_model(cfg):
    """Create model based on configuration."""
    model_name = cfg.model.name.lower()
    num_classes = cfg.model.num_classes
    
    if model_name == 'cnn':
        model = create_cnn_model(
            num_classes=num_classes,
            in_channels=cfg.model.get('in_channels', 3),
            fc_features=cfg.model.get('fc_features', 512),
            dropout=cfg.model.get('dropout', 0.5)
        )
    
    elif model_name == 'resnet':
        variant = cfg.model.get('variant', 'resnet34')
        model = create_resnet_model(
            model_name=variant,
            num_classes=num_classes,
            pretrained=cfg.model.get('pretrained', False)
        )
    
    elif model_name == 'vit':
        variant = cfg.model.get('variant', 'small')
        model = create_vit_model(
            model_name=variant,
            num_classes=num_classes,
            image_size=cfg.data.image_size,
            patch_size=cfg.model.get('patch_size', 16),
            dropout=cfg.model.get('dropout', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def create_optimizer(model: nn.Module, cfg):
    """Create optimizer based on configuration."""
    optimizer_name = cfg.training.optimizer.lower()
    lr = cfg.training.lr
    weight_decay = cfg.training.get('weight_decay', 0)
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(cfg.training.get('betas', [0.9, 0.999]))
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=cfg.training.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer, cfg, steps_per_epoch: int = None):
    """Create learning rate scheduler based on configuration."""
    scheduler_name = cfg.training.get('scheduler', 'cosine').lower()
    epochs = cfg.training.epochs
    
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=cfg.training.get('min_lr', 1e-6)
        )
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.training.get('step_size', 30),
            gamma=cfg.training.get('gamma', 0.1)
        )
    elif scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=cfg.training.get('scheduler_patience', 5)
        )
    elif scheduler_name == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


def create_loss_function(cfg):
    """Create loss function based on configuration."""
    loss_name = cfg.loss.name.lower()
    
    if loss_name == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    
    elif loss_name == 'focal':
        from src.losses.focal_loss import FocalLoss
        criterion = FocalLoss(
            gamma=cfg.loss.get('gamma', 2.0),
            alpha=cfg.loss.get('alpha', 0.25)
        )
    
    elif loss_name == 'weighted_bce':
        from src.losses.weighted_loss import WeightedBCELoss
        criterion = WeightedBCELoss(
            pos_weight=cfg.loss.get('pos_weight', None)
        )
    
    elif loss_name == 'combined':
        from src.losses.combined_loss import CombinedLoss
        criterion = CombinedLoss(
            focal_weight=cfg.loss.get('focal_weight', 0.5),
            bce_weight=cfg.loss.get('bce_weight', 0.5),
            gamma=cfg.loss.get('gamma', 2.0)
        )
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return criterion


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if cfg.training.get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                cfg.training.gradient_clip
            )
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Progress
        if (batch_idx + 1) % 50 == 0:
            logging.debug(
                f"Epoch {epoch+1} [{batch_idx+1}/{num_batches}] "
                f"Loss: {loss.item():.4f}"
            )
    
    avg_loss = total_loss / num_batches
    return {'train_loss': avg_loss}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        
        # Store predictions
        preds = torch.sigmoid(outputs)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    
    avg_loss = total_loss / len(val_loader)
    
    # Compute AUC
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_preds, average='macro')
    except Exception:
        auc = 0.0
    
    return {
        'val_loss': avg_loss,
        'val_auc': auc
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Chest X-ray Classifier')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode only')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint for evaluation')
    parser.add_argument('overrides', nargs='*', help='Config overrides in key=value format')
    
    args = parser.parse_args()
    
    # Load configuration
    overrides = parse_cli_overrides(args.overrides) if args.overrides else None
    cfg = load_config(args.config, overrides=overrides)
    
    # Setup output directory
    output_dir = Path(cfg.experiment.get('output_dir', 'outputs'))
    experiment_name = cfg.experiment.get('name', 'experiment')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(run_dir, cfg.logging.get('level', 'INFO'))
    
    # Save configuration
    save_config(cfg, run_dir / 'config.yaml')
    
    # Set random seed
    seed = cfg.experiment.get('seed', 42)
    set_seed(seed, deterministic=cfg.experiment.get('deterministic', False))
    
    # Log system info
    logging.info(f"Experiment: {experiment_name}")
    logging.info(f"Output directory: {run_dir}")
    info = get_reproducibility_info()
    logging.info(f"PyTorch: {info['pytorch_version']}, CUDA: {info['cuda_available']}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create model
    logging.info(f"Creating model: {cfg.model.name}")
    model = create_model(cfg)
    model = model.to(device)
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {num_params:,}")
    
    # Create data loaders (placeholder - implement with actual dataset)
    logging.info("Creating data loaders...")
    transforms_dict = create_transforms(
        image_size=cfg.data.image_size,
        augmentation=cfg.data.get('augmentation', 'standard'),
        normalize=cfg.data.get('normalize', 'imagenet')
    )
    
    # NOTE: Actual DataLoader creation requires dataset implementation
    # This is a template showing the expected interface
    logging.info("Data loader creation requires dataset implementation")
    logging.info("See src/data/dataset.py and src/data/splits.py")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)
    criterion = create_loss_function(cfg)
    
    logging.info(f"Optimizer: {cfg.training.optimizer}, LR: {cfg.training.lr}")
    logging.info(f"Loss function: {cfg.loss.name}")
    
    # Setup callbacks
    callbacks = CallbackList([
        EarlyStopping(
            patience=cfg.training.get('patience', 10),
            monitor='val_loss',
            mode='min',
            restore_best=True
        ),
        ModelCheckpoint(
            save_dir=run_dir / 'checkpoints',
            monitor='val_auc',
            mode='max',
            save_best_only=not cfg.training.get('save_every', False),
            save_every=cfg.training.get('save_every', None)
        ),
        MetricsHistory(save_path=run_dir / 'history.json'),
        ProgressLogger(total_epochs=cfg.training.epochs)
    ])
    
    # Training loop
    logging.info(f"Starting training for {cfg.training.epochs} epochs")
    callbacks.on_train_begin()
    
    # NOTE: Actual training loop requires data loaders
    # This template shows the expected structure:
    """
    for epoch in range(cfg.training.epochs):
        callbacks.on_epoch_begin(epoch)
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg
        )
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()
        
        callbacks.on_epoch_end(epoch, train_metrics, val_metrics, model)
        
        if callbacks.should_stop:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    callbacks.on_train_end(model=model)
    """
    
    logging.info("Training script setup complete!")
    logging.info(f"To train, implement DataLoader in main() using:")
    logging.info("  - src/data/dataset.py for ChestXrayDataset")
    logging.info("  - src/data/splits.py for patient_level_split()")
    logging.info("  - src/data/transforms.py for data augmentation")


if __name__ == '__main__':
    main()