"""
Utilities Module
================

Contains training utilities, configuration helpers, and other utilities.

Components:
- training: Training loops and evaluation
- callbacks: EarlyStopping, ModelCheckpoint, etc.
- reproducibility: Seed management, deterministic training
- config_loader: YAML configuration system

Author: ViT-Chest-Xray Project Team
"""

from .training import (
    Trainer,
    plot_training_history,
    evaluate_model,
    create_optimizer_scheduler
)

from .callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    ModelCheckpoint,
    MetricsHistory,
    ProgressLogger,
    LearningRateLogger
)

from .reproducibility import (
    set_seed,
    seed_worker,
    get_dataloader_generator,
    get_reproducibility_info,
    save_experiment_config,
    compute_config_hash,
    ReproducibilityContext
)

from .config_loader import (
    ConfigDict,
    load_config,
    save_config,
    merge_configs,
    parse_cli_overrides,
    DEFAULT_CONFIG
)

__all__ = [
    # Training
    'Trainer',
    'plot_training_history',
    'evaluate_model',
    'create_optimizer_scheduler',
    # Callbacks
    'Callback',
    'CallbackList',
    'EarlyStopping',
    'ModelCheckpoint',
    'MetricsHistory',
    'ProgressLogger',
    'LearningRateLogger',
    # Reproducibility
    'set_seed',
    'seed_worker',
    'get_dataloader_generator',
    'get_reproducibility_info',
    'save_experiment_config',
    'compute_config_hash',
    'ReproducibilityContext',
    # Config
    'ConfigDict',
    'load_config',
    'save_config',
    'merge_configs',
    'parse_cli_overrides',
    'DEFAULT_CONFIG',
]