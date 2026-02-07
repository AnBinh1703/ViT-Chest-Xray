"""
Training Callbacks for Deep Learning Experiments
=================================================

Provides modular callbacks for training loop:
- EarlyStopping: Stop training when metric plateaus
- ModelCheckpoint: Save best and periodic checkpoints
- LearningRateScheduler: Adjust learning rate during training
- ProgressLogger: Log training progress
- MetricsLogger: Track and save metrics history

Usage:
    from src.utils.callbacks import EarlyStopping, ModelCheckpoint
    
    callbacks = [
        EarlyStopping(patience=10, monitor='val_auc'),
        ModelCheckpoint(save_dir='checkpoints/', monitor='val_auc'),
    ]
    
    for epoch in range(epochs):
        train_metrics, val_metrics = train_one_epoch(...)
        
        for callback in callbacks:
            callback.on_epoch_end(epoch, train_metrics, val_metrics, model)
            
        if any(cb.should_stop for cb in callbacks):
            break

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from datetime import datetime

import torch
import torch.nn as nn


class Callback(ABC):
    """Base class for training callbacks."""
    
    def __init__(self):
        self.should_stop = False
    
    def on_train_begin(self, **kwargs) -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, **kwargs) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        model: nn.Module,
        **kwargs
    ) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, **kwargs) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, batch: int, loss: float, **kwargs) -> None:
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.
    
    Args:
        patience: Number of epochs with no improvement to wait
        monitor: Metric to monitor (e.g., 'val_loss', 'val_auc')
        mode: 'min' or 'max' (minimize loss, maximize auc)
        min_delta: Minimum change to qualify as an improvement
        restore_best: Restore best weights when stopping
    """
    
    def __init__(
        self,
        patience: int = 10,
        monitor: str = 'val_loss',
        mode: str = 'min',
        min_delta: float = 0.0,
        restore_best: bool = True,
        verbose: bool = True
    ):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.verbose = verbose
        
        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        
        if mode == 'min':
            self.is_better = lambda curr, best: curr < best - min_delta
            self.best_score = float('inf')
        else:
            self.is_better = lambda curr, best: curr > best + min_delta
            self.best_score = float('-inf')
    
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        model: nn.Module,
        **kwargs
    ) -> None:
        # Get current metric value
        metrics = {**train_metrics, **val_metrics}
        current = metrics.get(self.monitor)
        
        if current is None:
            logging.warning(f"EarlyStopping: metric '{self.monitor}' not found")
            return
        
        if self.is_better(current, self.best_score):
            self.best_score = current
            self.best_epoch = epoch
            self.counter = 0
            
            if self.restore_best:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            
            if self.verbose:
                logging.info(f"EarlyStopping: {self.monitor} improved to {current:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"EarlyStopping: {self.monitor} did not improve. "
                    f"Patience: {self.counter}/{self.patience}"
                )
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    logging.info(
                        f"EarlyStopping triggered at epoch {epoch}. "
                        f"Best: {self.best_score:.6f} at epoch {self.best_epoch}"
                    )
    
    def on_train_end(self, model: nn.Module = None, **kwargs) -> None:
        if self.restore_best and self.best_weights is not None and model is not None:
            model.load_state_dict(self.best_weights)
            logging.info(f"Restored best weights from epoch {self.best_epoch}")


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    
    Args:
        save_dir: Directory to save checkpoints
        monitor: Metric to monitor for best model
        mode: 'min' or 'max'
        save_best_only: Only save when monitored metric improves
        save_every: Save every N epochs (regardless of improvement)
        filename_template: Template for checkpoint filename
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = False,
        save_every: Optional[int] = None,
        filename_template: str = 'checkpoint_epoch{epoch:03d}_{monitor:.4f}.pth',
        verbose: bool = True
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_every = save_every
        self.filename_template = filename_template
        self.verbose = verbose
        
        if mode == 'min':
            self.is_better = lambda curr, best: curr < best
            self.best_score = float('inf')
        else:
            self.is_better = lambda curr, best: curr > best
            self.best_score = float('-inf')
        
        self.best_path = None
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> Path:
        """Save a checkpoint."""
        # Build filename
        monitor_value = metrics.get(self.monitor, 0.0)
        filename = self.filename_template.format(
            epoch=epoch,
            monitor=monitor_value
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        path = self.save_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.best_path = best_path
        
        return path
    
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        model: nn.Module,
        **kwargs
    ) -> None:
        metrics = {**train_metrics, **val_metrics}
        current = metrics.get(self.monitor)
        
        is_best = False
        if current is not None and self.is_better(current, self.best_score):
            self.best_score = current
            is_best = True
        
        should_save = False
        if is_best:
            should_save = True
        elif not self.save_best_only:
            if self.save_every and (epoch + 1) % self.save_every == 0:
                should_save = True
        
        if should_save:
            path = self._save_checkpoint(model, epoch, metrics, is_best)
            if self.verbose:
                status = "BEST" if is_best else ""
                logging.info(f"Saved checkpoint: {path.name} {status}")


class LearningRateLogger(Callback):
    """Log learning rate at each epoch."""
    
    def __init__(self, optimizer: torch.optim.Optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.history = []
    
    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.history.append({'epoch': epoch, 'learning_rates': lrs})
        logging.info(f"Learning rate(s): {lrs}")


class MetricsHistory(Callback):
    """
    Track and save metrics history.
    
    Args:
        save_path: Path to save history JSON
    """
    
    def __init__(self, save_path: Optional[Union[str, Path]] = None):
        super().__init__()
        self.save_path = Path(save_path) if save_path else None
        self.history = {
            'train': [],
            'val': [],
            'epochs': []
        }
    
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        **kwargs
    ) -> None:
        self.history['epochs'].append(epoch)
        self.history['train'].append(train_metrics)
        self.history['val'].append(val_metrics)
        
        if self.save_path:
            self._save()
    
    def _save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_best_epoch(self, metric: str, mode: str = 'max') -> int:
        """Get the epoch with best metric value."""
        values = [m.get(metric, float('-inf' if mode == 'max' else 'inf')) 
                  for m in self.history['val']]
        
        if mode == 'max':
            return values.index(max(values))
        return values.index(min(values))


class ProgressLogger(Callback):
    """
    Log training progress with time estimates.
    """
    
    def __init__(self, total_epochs: int, log_every: int = 1):
        super().__init__()
        self.total_epochs = total_epochs
        self.log_every = log_every
        self.start_time = None
        self.epoch_times = []
    
    def on_train_begin(self, **kwargs) -> None:
        self.start_time = time.time()
        logging.info(f"Starting training for {self.total_epochs} epochs")
    
    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        self.epoch_start = time.time()
    
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        **kwargs
    ) -> None:
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        
        if (epoch + 1) % self.log_every == 0:
            # Calculate ETA
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = self.total_epochs - epoch - 1
            eta = avg_epoch_time * remaining_epochs
            
            # Format metrics
            train_str = ', '.join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
            val_str = ', '.join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
            
            logging.info(
                f"Epoch {epoch+1}/{self.total_epochs} "
                f"({epoch_time:.1f}s, ETA: {eta/60:.1f}min)\n"
                f"  Train: {train_str}\n"
                f"  Val: {val_str}"
            )
    
    def on_train_end(self, **kwargs) -> None:
        total_time = time.time() - self.start_time
        logging.info(f"Training completed in {total_time/60:.1f} minutes")


class CallbackList:
    """
    Container for multiple callbacks.
    
    Usage:
        callbacks = CallbackList([
            EarlyStopping(patience=10),
            ModelCheckpoint(save_dir='checkpoints/')
        ])
        
        callbacks.on_epoch_end(epoch, train_metrics, val_metrics, model)
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)
    
    @property
    def should_stop(self) -> bool:
        return any(cb.should_stop for cb in self.callbacks)
    
    def on_train_begin(self, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(**kwargs)
    
    def on_train_end(self, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)
    
    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, **kwargs)
    
    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        model: nn.Module,
        **kwargs
    ) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, train_metrics, val_metrics, model, **kwargs)
    
    def on_batch_begin(self, batch: int, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(batch, **kwargs)
    
    def on_batch_end(self, batch: int, loss: float, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(batch, loss, **kwargs)


if __name__ == "__main__":
    import tempfile
    
    print("Testing callbacks...")
    
    # Create a simple model for testing
    model = nn.Linear(10, 2)
    
    # Test EarlyStopping
    early_stopping = EarlyStopping(patience=2, monitor='val_loss', mode='min')
    
    # Simulate epochs
    for epoch, val_loss in enumerate([1.0, 0.8, 0.9, 0.95, 0.96]):
        early_stopping.on_epoch_end(
            epoch, 
            {'train_loss': 0.5}, 
            {'val_loss': val_loss},
            model
        )
        if early_stopping.should_stop:
            print(f"✓ EarlyStopping triggered at epoch {epoch}")
            break
    
    # Test ModelCheckpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = ModelCheckpoint(
            save_dir=tmpdir,
            monitor='val_auc',
            mode='max',
            save_best_only=True
        )
        
        for epoch, auc in enumerate([0.7, 0.8, 0.75]):
            checkpoint.on_epoch_end(
                epoch,
                {'train_loss': 0.5},
                {'val_auc': auc},
                model
            )
        
        assert checkpoint.best_path is not None
        print(f"✓ ModelCheckpoint saved best model")
    
    # Test CallbackList
    callbacks = CallbackList([
        EarlyStopping(patience=5),
        ProgressLogger(total_epochs=10)
    ])
    callbacks.on_train_begin()
    print("✓ CallbackList works")
    
    print("\n✓ All callback tests passed!")
