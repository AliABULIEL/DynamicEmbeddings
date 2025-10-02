"""I/O utilities for loading/saving configs and data."""

import json
from pathlib import Path
from typing import Any, Optional

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: Path) -> DictConfig:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        DictConfig: Configuration object
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config is invalid YAML
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    return OmegaConf.create(config_dict)


def load_all_configs(config_dir: Path) -> dict[str, DictConfig]:
    """Load all configuration files from a directory.
    
    Args:
        config_dir: Directory containing config YAML files
    
    Returns:
        dict: Mapping of config names to DictConfig objects
    """
    configs = {}
    
    for config_file in config_dir.glob("*.yaml"):
        config_name = config_file.stem
        configs[config_name] = load_config(config_file)
    
    return configs


def save_json(data: Any, output_path: Path, indent: int = 2) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save (must be JSON-serializable)
        output_path: Output file path
        indent: JSON indentation level
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(input_path: Path) -> Any:
    """Load data from JSON file.
    
    Args:
        input_path: Input file path
    
    Returns:
        Any: Loaded data
    
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not input_path.exists():
        raise FileNotFoundError(f"JSON file not found: {input_path}")
    
    with open(input_path) as f:
        return json.load(f)


def merge_configs(base: DictConfig, override: DictConfig) -> DictConfig:
    """Merge two configurations, with override taking precedence.
    
    Args:
        base: Base configuration
        override: Configuration to override base
    
    Returns:
        DictConfig: Merged configuration
    """
    return OmegaConf.merge(base, override)
