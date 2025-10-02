"""I/O utilities for configs and data."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file.
    
    Args:
        path: Path to YAML file.
        
    Returns:
        Dictionary with config values.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    return config if config is not None else {}


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    """Save a dictionary to a YAML file.
    
    Args:
        data: Dictionary to save.
        path: Output path for YAML file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_config(config_name: str, config_dir: Path) -> Dict[str, Any]:
    """Load a config file by name from the config directory.
    
    Args:
        config_name: Name of config file (without .yaml extension).
        config_dir: Directory containing config files.
        
    Returns:
        Dictionary with config values.
    """
    config_path = config_dir / f"{config_name}.yaml"
    return load_yaml(config_path)
