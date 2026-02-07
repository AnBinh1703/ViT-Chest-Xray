"""
Configuration Package
====================

Central configuration management for the ViT Chest X-ray project.

Usage:
    from config import config, loss_config
    from config.main_config import ProjectConfig, print_full_config

Author: ViT-Chest-Xray Project Team
"""

from .main_config import config, loss_config, ProjectConfig, print_full_config

__all__ = ['config', 'loss_config', 'ProjectConfig', 'print_full_config']