"""
Training Utilities for Chest X-ray Classification
=================================================

Contains training loops, evaluation metrics, and model utilities.

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


class Trainer:
    """
    Trainer class for chest X-ray classification models.

    Handles training loops, validation, model saving, and logging.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 criterion: nn.Module, optimizer: optim.Optimizer,
                 scheduler: Optional[Any] = None,
                 checkpoint_dir: str = "models/checkpoints"):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            device: Device to run training on
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': [],
            'train_acc': [], 'val_acc': [],
            'learning_rate': []
        }

        self.best_val_auc = 0.0
        self.best_model_path = None

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (loss, auc, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_outputs = []

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Store for metrics calculation
            all_targets.append(labels.cpu().detach().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().detach().numpy())

        # Calculate metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        all_targets = np.vstack(all_targets)
        all_outputs = np.vstack(all_outputs)

        auc = self._calculate_auc(all_targets, all_outputs)
        acc = self._calculate_accuracy(all_targets, all_outputs)

        return epoch_loss, auc, acc

    def validate_epoch(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (loss, auc, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                all_targets.append(labels.cpu().numpy())
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

        # Calculate metrics
        epoch_loss = running_loss / len(val_loader.dataset)
        all_targets = np.vstack(all_targets)
        all_outputs = np.vstack(all_outputs)

        auc = self._calculate_auc(all_targets, all_outputs)
        acc = self._calculate_accuracy(all_targets, all_outputs)

        return epoch_loss, auc, acc

    def _calculate_auc(self, targets: np.ndarray, outputs: np.ndarray) -> float:
        """Calculate AUC score, handling classes with single labels."""
        try:
            valid_classes = []
            for i in range(targets.shape[1]):
                if len(np.unique(targets[:, i])) > 1:
                    valid_classes.append(i)

            if len(valid_classes) > 0:
                return roc_auc_score(
                    targets[:, valid_classes],
                    outputs[:, valid_classes],
                    average='macro'
                )
            else:
                return 0.0
        except ValueError:
            return 0.0

    def _calculate_accuracy(self, targets: np.ndarray, outputs: np.ndarray) -> float:
        """Calculate accuracy using threshold of 0.5."""
        predictions = (outputs > 0.5).astype(int)
        return accuracy_score(targets, predictions)

    def train(self, train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader,
              num_epochs: int, model_name: str = "model") -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            model_name: Name for saving checkpoints

        Returns:
            Training history dictionary
        """
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)

            # Train
            train_loss, train_auc, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_auc, val_acc = self.validate_epoch(val_loader)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            if self.scheduler:
                self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
                self.scheduler.step()
            else:
                self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Print metrics
            print(".4f")
            print(".4f")

            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_model_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pth")
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"✓ New best model saved! (AUC: {val_auc:.4f})")

        print(f"\nTraining completed. Best validation AUC: {self.best_val_auc:.4f}")
        return self.history

    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint with training state."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'best_val_auc': self.best_val_auc
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_auc = checkpoint['best_val_auc']


def plot_training_history(history: Dict[str, List[float]]) -> None:
    """
    Plot training history.

    Args:
        history: Training history dictionary
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # AUC plot
    axes[0, 1].plot(history['train_auc'], label='Train AUC')
    axes[0, 1].plot(history['val_auc'], label='Validation AUC')
    axes[0, 1].set_title('Training and Validation AUC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Accuracy plot
    axes[1, 0].plot(history['train_acc'], label='Train Accuracy')
    axes[1, 0].plot(history['val_acc'], label='Validation Accuracy')
    axes[1, 0].set_title('Training and Validation Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning rate plot
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_model(model: nn.Module, test_loader: torch.utils.data.DataLoader,
                  device: torch.device, class_names: List[str]) -> Dict[str, Any]:
    """
    Evaluate model on test set with detailed metrics.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        class_names: List of class names

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            all_targets.append(labels.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

    targets = np.vstack(all_targets)
    outputs = np.vstack(all_outputs)
    predictions = (outputs > 0.5).astype(int)

    # Calculate metrics
    results = {}

    # Overall metrics
    results['accuracy'] = accuracy_score(targets, predictions)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None
    )

    results['per_class'] = {}
    for i, class_name in enumerate(class_names):
        results['per_class'][class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }

    # Macro-averaged metrics
    results['macro_precision'] = np.mean(precision)
    results['macro_recall'] = np.mean(recall)
    results['macro_f1'] = np.mean(f1)

    # AUC
    try:
        valid_classes = []
        for i in range(targets.shape[1]):
            if len(np.unique(targets[:, i])) > 1:
                valid_classes.append(i)

        if len(valid_classes) > 0:
            results['auc_macro'] = roc_auc_score(
                targets[:, valid_classes],
                outputs[:, valid_classes],
                average='macro'
            )
            results['auc_micro'] = roc_auc_score(
                targets[:, valid_classes],
                outputs[:, valid_classes],
                average='micro'
            )
        else:
            results['auc_macro'] = 0.0
            results['auc_micro'] = 0.0
    except ValueError:
        results['auc_macro'] = 0.0
        results['auc_micro'] = 0.0

    return results


def create_optimizer_scheduler(model: nn.Module, config: Dict[str, Any]) -> Tuple[optim.Optimizer, Optional[Any]]:
    """
    Create optimizer and scheduler based on configuration.

    Args:
        model: PyTorch model
        config: Training configuration dictionary

    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer_name = config.get('optimizer', 'Adam')
    lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 1e-6)

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                            weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Scheduler
    scheduler_type = config.get('scheduler')
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get('num_epochs', 10))
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.get('step_size', 5),
                                            gamma=config.get('gamma', 0.1))
    else:
        scheduler = None

    return optimizer, scheduler


if __name__ == "__main__":
    # Test trainer initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create a simple test
    from models.cnn import create_cnn_model
    model = create_cnn_model(num_classes=15)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    trainer = Trainer(model, device, criterion, optimizer)
    print("Trainer initialized successfully")