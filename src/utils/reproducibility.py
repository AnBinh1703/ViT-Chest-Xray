"""
Reproducibility Utilities for Deep Learning Experiments
========================================================

Ensures deterministic and reproducible training across:
- Random number generators (Python, NumPy, PyTorch)
- CUDA operations
- DataLoader workers

Usage:
    from src.utils.reproducibility import set_seed, get_reproducibility_info
    
    set_seed(42)  # Call at start of training
    info = get_reproducibility_info()  # Log with experiment

IMPORTANT: For full reproducibility, also set:
    - torch.backends.cudnn.benchmark = False
    - torch.use_deterministic_algorithms(True)
    
Note: Deterministic mode may reduce performance by 10-20%

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import os
import sys
import random
import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Try to import Git for version tracking
try:
    import git
    HAS_GIT = True
except ImportError:
    HAS_GIT = False


def set_seed(
    seed: int = 42,
    deterministic: bool = False,
    benchmark: bool = True
) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Enable full deterministic mode (slower)
        benchmark: Enable cuDNN benchmark (faster, but non-deterministic)
    
    Note:
        - deterministic=True and benchmark=False for full reproducibility
        - deterministic=False and benchmark=True for faster training
    """
    # Python RNG
    random.seed(seed)
    
    # NumPy RNG
    np.random.seed(seed)
    
    # PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # cuDNN settings
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark and not deterministic
    
    # Full deterministic algorithms (PyTorch 1.8+)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # Older PyTorch version
            pass


def seed_worker(worker_id: int) -> None:
    """
    Seed function for DataLoader workers.
    
    Use with DataLoader:
        DataLoader(..., worker_init_fn=seed_worker)
    
    Args:
        worker_id: Worker process ID
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader_generator(seed: int = 42) -> torch.Generator:
    """
    Create a generator for DataLoader shuffling.
    
    Use with DataLoader:
        g = get_dataloader_generator(42)
        DataLoader(..., generator=g)
    
    Args:
        seed: Random seed
        
    Returns:
        PyTorch Generator object
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Collect system and environment info for logging.
    
    Returns:
        Dictionary with reproducibility-relevant information
    """
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'num_gpus': torch.cuda.device_count(),
        'platform': sys.platform,
    }
    
    # NumPy version
    info['numpy_version'] = np.__version__
    
    # GPU info
    if torch.cuda.is_available():
        info['gpu_names'] = [
            torch.cuda.get_device_name(i) 
            for i in range(torch.cuda.device_count())
        ]
    
    # Try to get git commit hash
    if HAS_GIT:
        try:
            repo = git.Repo(search_parent_directories=True)
            info['git_hash'] = repo.head.object.hexsha
            info['git_branch'] = repo.active_branch.name
            info['git_dirty'] = repo.is_dirty()
        except (git.InvalidGitRepositoryError, git.NoSuchPathError):
            pass
    
    return info


def save_experiment_config(
    config: Dict[str, Any],
    save_dir: Path,
    seed: int,
    experiment_name: Optional[str] = None
) -> Path:
    """
    Save experiment configuration and reproducibility info.
    
    Args:
        config: Experiment configuration dictionary
        save_dir: Directory to save config
        seed: Random seed used
        experiment_name: Optional experiment name
        
    Returns:
        Path to saved config file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"experiment_{timestamp}"
    
    # Combine config with reproducibility info
    full_config = {
        'experiment_name': experiment_name,
        'seed': seed,
        'config': config,
        'reproducibility': get_reproducibility_info()
    }
    
    # Save as JSON
    config_path = save_dir / f"{experiment_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(full_config, f, indent=2, default=str)
    
    return config_path


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute a hash of the configuration for tracking.
    
    Useful for identifying experiments with identical configs.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MD5 hash string (first 8 chars)
    """
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


class ReproducibilityContext:
    """
    Context manager for reproducible code blocks.
    
    Usage:
        with ReproducibilityContext(seed=42):
            # All random operations here are reproducible
            model = create_model()
            train(model)
    """
    
    def __init__(self, seed: int = 42, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self._original_states = {}
    
    def __enter__(self):
        # Save current states
        self._original_states = {
            'python_rng': random.getstate(),
            'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
        }
        
        # Set seeds
        set_seed(self.seed, deterministic=self.deterministic)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original states
        random.setstate(self._original_states['python_rng'])
        np.random.set_state(self._original_states['numpy_rng'])
        torch.set_rng_state(self._original_states['torch_rng'])
        
        if self._original_states['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(self._original_states['cuda_rng'])
        
        torch.backends.cudnn.deterministic = self._original_states['cudnn_deterministic']
        torch.backends.cudnn.benchmark = self._original_states['cudnn_benchmark']


if __name__ == "__main__":
    # Test reproducibility utilities
    print("Testing reproducibility utilities...")
    
    # Test set_seed
    set_seed(42)
    r1 = torch.rand(5)
    
    set_seed(42)
    r2 = torch.rand(5)
    
    assert torch.equal(r1, r2), "Seed not working!"
    print("✓ set_seed() works correctly")
    
    # Test context manager
    with ReproducibilityContext(seed=123):
        t1 = torch.rand(3)
    
    with ReproducibilityContext(seed=123):
        t2 = torch.rand(3)
    
    assert torch.equal(t1, t2), "Context manager not working!"
    print("✓ ReproducibilityContext works correctly")
    
    # Test info gathering
    info = get_reproducibility_info()
    print(f"\n✓ System info collected: {len(info)} fields")
    print(f"  PyTorch: {info['pytorch_version']}")
    print(f"  CUDA: {info['cuda_available']}")
    
    # Test config hash
    config = {'model': 'vit', 'lr': 1e-4, 'epochs': 100}
    hash_val = compute_config_hash(config)
    print(f"\n✓ Config hash: {hash_val}")
    
    print("\n✓ All reproducibility tests passed!")
