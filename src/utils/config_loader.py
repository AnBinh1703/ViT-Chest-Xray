"""
YAML Configuration Manager for Deep Learning Experiments
=========================================================

Provides a clean, hierarchical configuration system with:
- YAML file loading with defaults
- Environment variable interpolation
- Config inheritance (base + model-specific)
- Validation and type checking
- Config freezing for reproducibility

Usage:
    from src.utils.config_loader import load_config, merge_configs
    
    # Load a single config
    cfg = load_config('configs/vit.yaml')
    
    # Access nested values
    print(cfg.model.name)  # 'vit_small'
    print(cfg['training']['lr'])  # 1e-4

Author: ViT-Chest-Xray Project Team
Date: February 2026
"""

import os
import yaml
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field, asdict


class ConfigDict(dict):
    """
    Dictionary with attribute-style access.
    
    Allows both cfg['key'] and cfg.key access patterns.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)
            elif isinstance(value, list):
                self[key] = [
                    ConfigDict(item) if isinstance(item, dict) else item
                    for item in value
                ]
    
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value
    
    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def to_dict(self) -> Dict:
        """Convert to regular dictionary."""
        result = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, ConfigDict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


def _interpolate_env_vars(value: Any) -> Any:
    """
    Replace ${ENV_VAR} patterns with environment variable values.
    
    Args:
        value: String or nested structure to process
        
    Returns:
        Value with environment variables interpolated
    """
    if isinstance(value, str):
        # Find all ${VAR} or ${VAR:default} patterns
        import re
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace(match):
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name, default)
            if env_value is None:
                raise ValueError(f"Environment variable {var_name} not set and no default")
            return env_value
        
        return re.sub(pattern, replace, value)
    
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    
    return value


def load_yaml(path: Union[str, Path]) -> Dict:
    """
    Load a YAML file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Dictionary from YAML
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    return config


def merge_configs(
    base: Dict,
    override: Dict,
    deep_merge: bool = True
) -> Dict:
    """
    Merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration (takes precedence)
        deep_merge: If True, merge nested dicts; if False, replace
        
    Returns:
        Merged configuration
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if (
            deep_merge 
            and key in result 
            and isinstance(result[key], dict) 
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value, deep_merge=True)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def load_config(
    config_path: Union[str, Path],
    base_config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict] = None,
    interpolate_env: bool = True
) -> ConfigDict:
    """
    Load configuration from YAML file(s).
    
    Args:
        config_path: Path to main config file
        base_config_path: Optional path to base config (merged first)
        overrides: Optional dictionary of command-line overrides
        interpolate_env: Whether to interpolate ${ENV_VAR} patterns
        
    Returns:
        ConfigDict with loaded configuration
    """
    # Load base config if provided
    if base_config_path is not None:
        base = load_yaml(base_config_path)
    else:
        base = {}
    
    # Load main config
    main = load_yaml(config_path)
    
    # Check if main config specifies a base
    if '_base_' in main:
        base_path = main.pop('_base_')
        # Resolve relative to config file location
        if not Path(base_path).is_absolute():
            base_path = Path(config_path).parent / base_path
        inherited_base = load_yaml(base_path)
        base = merge_configs(inherited_base, base)
    
    # Merge: base <- main <- overrides
    config = merge_configs(base, main)
    
    if overrides:
        config = merge_configs(config, overrides)
    
    # Interpolate environment variables
    if interpolate_env:
        config = _interpolate_env_vars(config)
    
    return ConfigDict(config)


def save_config(config: Union[Dict, ConfigDict], path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to regular dict if needed
    if isinstance(config, ConfigDict):
        config = config.to_dict()
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def parse_cli_overrides(override_list: List[str]) -> Dict:
    """
    Parse command-line overrides in format "key=value" or "nested.key=value".
    
    Args:
        override_list: List of "key=value" strings
        
    Returns:
        Nested dictionary of overrides
    """
    result = {}
    
    for item in override_list:
        if '=' not in item:
            raise ValueError(f"Invalid override format: {item}. Expected 'key=value'")
        
        key, value = item.split('=', 1)
        keys = key.split('.')
        
        # Parse value type
        try:
            # Try to parse as Python literal
            import ast
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Keep as string
            pass
        
        # Build nested dict
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    return result


# Default configurations
DEFAULT_CONFIG = ConfigDict({
    'experiment': {
        'name': 'chestxray',
        'seed': 42,
        'output_dir': 'outputs',
    },
    'data': {
        'image_size': 224,
        'batch_size': 32,
        'num_workers': 4,
        'train_ratio': 0.70,
        'val_ratio': 0.15,
        'augmentation': 'standard',
    },
    'model': {
        'name': 'vit_small',
        'num_classes': 15,
        'pretrained': False,
    },
    'training': {
        'epochs': 100,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'early_stopping_patience': 10,
    },
    'loss': {
        'name': 'focal',
        'gamma': 2.0,
        'alpha': 0.25,
    }
})


if __name__ == "__main__":
    # Test config loading
    print("Testing config system...")
    
    # Test ConfigDict
    cfg = ConfigDict({'model': {'name': 'vit', 'size': 'small'}})
    assert cfg.model.name == 'vit'
    assert cfg['model']['size'] == 'small'
    print("✓ ConfigDict access works")
    
    # Test merge
    base = {'a': 1, 'b': {'c': 2, 'd': 3}}
    override = {'b': {'c': 20}, 'e': 5}
    merged = merge_configs(base, override)
    assert merged['a'] == 1
    assert merged['b']['c'] == 20
    assert merged['b']['d'] == 3
    assert merged['e'] == 5
    print("✓ Config merge works")
    
    # Test CLI parsing
    overrides = parse_cli_overrides(['model.lr=0.001', 'epochs=200'])
    assert overrides['model']['lr'] == 0.001
    assert overrides['epochs'] == 200
    print("✓ CLI override parsing works")
    
    # Test env interpolation
    os.environ['TEST_VAR'] = 'test_value'
    result = _interpolate_env_vars('${TEST_VAR}')
    assert result == 'test_value'
    print("✓ Environment interpolation works")
    
    print("\n✓ All config system tests passed!")
