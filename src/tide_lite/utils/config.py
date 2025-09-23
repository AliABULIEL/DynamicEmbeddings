"""Unified configuration loader for TIDE-Lite.

This module provides a single, type-validated configuration loader that:
- Loads YAML configuration files
- Merges configurations with proper precedence
- Validates all configuration parameters
- Provides utilities for reproducibility
"""

import json
import logging
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import yaml

from .common import seed_everything

logger = logging.getLogger(__name__)


@dataclass
class TIDEConfig:
    """Complete TIDE-Lite configuration with type validation.
    
    This is the single source of truth for all configuration parameters.
    All fields are documented in configs/defaults.yaml.
    """
    
    # Model architecture
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    hidden_dim: int = 384
    time_dims: int = 32
    time_mlp_hidden: int = 128
    mlp_dropout: float = 0.1
    freeze_encoder: bool = True
    pooling_strategy: Literal["mean", "cls", "max"] = "mean"
    gate_activation: Literal["sigmoid", "tanh"] = "sigmoid"
    
    # Data configuration
    max_seq_len: int = 128
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 2
    cache_dir: str = "./data"
    timeqa_data_dir: str = "./data/timeqa"
    
    # Training hyperparameters
    epochs: int = 10
    lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Loss configuration
    consistency_weight: float = 0.01
    preservation_weight: float = 0.005
    tau_seconds: float = 86400.0
    
    # Hardware & optimization
    device: Optional[str] = None
    use_amp: bool = True
    seed: int = 42
    
    # Checkpointing & evaluation
    eval_every: int = 100
    save_every: int = 500
    out_dir: str = "results/default"
    checkpoint_dir: Optional[str] = None
    
    # Retrieval & indexing
    faiss: Literal["FlatIP", "IVF", "HNSW"] = "FlatIP"
    faiss_n_clusters: int = 100
    faiss_n_probe: int = 10
    
    # Logging & debugging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    dry_run: bool = False
    temporal_enabled: bool = True
    
    # Evaluation tasks
    eval_tasks: List[str] = field(default_factory=lambda: ["stsb", "quora", "temporal"])
    eval_stsb_split: str = "test"
    eval_quora_split: str = "test"
    eval_temporal_split: str = "test"
    
    # Experimental features
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 0
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    ema_decay: float = 0.0
    
    # Dataset paths
    stsb_path: str = "nyu-mll/glue"
    stsb_config: str = "stsb"
    quora_path: str = "quora"
    quora_config: Optional[str] = None
    temporal_dataset: Literal["timeqa", "templama"] = "templama"
    templama_path: str = "./data/templama"
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Set checkpoint_dir if not provided
        if self.checkpoint_dir is None:
            self.checkpoint_dir = f"{self.out_dir}/checkpoints"
        
        # Auto-detect device if not specified
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate numeric ranges
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        
        if not 0 <= self.mlp_dropout < 1:
            raise ValueError(f"mlp_dropout must be in [0, 1), got {self.mlp_dropout}")
        
        if self.consistency_weight < 0:
            raise ValueError(f"consistency_weight must be non-negative, got {self.consistency_weight}")
        
        if self.preservation_weight < 0:
            raise ValueError(f"preservation_weight must be non-negative, got {self.preservation_weight}")
        
        # Validate dimensions
        if self.time_dims % 2 != 0:
            raise ValueError(f"time_dims must be even for sinusoidal encoding, got {self.time_dims}")
        
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        
        if self.time_mlp_hidden <= 0:
            raise ValueError(f"time_mlp_hidden must be positive, got {self.time_mlp_hidden}")
        
        # Create output directory
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file.
        
        Args:
            path: Path to save configuration.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
        
        logger.info(f"Saved configuration to {path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TIDEConfig":
        """Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary.
            
        Returns:
            TIDEConfig instance.
        """
        # Filter out unknown keys with warning
        import inspect
        valid_fields = set(inspect.signature(cls).parameters.keys())
        
        unknown_keys = set(config_dict.keys()) - valid_fields
        if unknown_keys:
            logger.warning(f"Ignoring unknown configuration keys: {sorted(unknown_keys)}")
        
        valid_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**valid_config)


class ConfigLoader:
    """Unified configuration loader with YAML support and override merging."""
    
    @staticmethod
    def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML file.
            
        Returns:
            Configuration dictionary.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML parsing fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.debug(f"Loaded configuration from {path}")
        return config or {}
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries.
        
        Later configs override earlier ones.
        
        Args:
            *configs: Configuration dictionaries to merge.
            
        Returns:
            Merged configuration dictionary.
        """
        result = {}
        for config in configs:
            if config:
                result.update(config)
        return result
    
    @classmethod
    def load(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        cli_args: Optional[Dict[str, Any]] = None,
    ) -> TIDEConfig:
        """Load and merge configuration from multiple sources.
        
        Precedence (highest to lowest):
        1. CLI arguments
        2. Programmatic overrides
        3. Config file
        4. Defaults
        
        Args:
            config_path: Path to YAML configuration file.
            overrides: Dictionary of programmatic overrides.
            cli_args: Dictionary of CLI argument overrides.
            
        Returns:
            Validated TIDEConfig instance.
        """
        # Start with defaults (built into dataclass)
        config_dict = {}
        
        # Load from file if provided
        if config_path:
            config_dict = cls.load_yaml(config_path)
        
        # Apply programmatic overrides
        if overrides:
            config_dict = cls.merge_configs(config_dict, overrides)
        
        # Apply CLI overrides (highest precedence)
        if cli_args:
            # Filter out None values from CLI
            cli_overrides = {k: v for k, v in cli_args.items() if v is not None}
            config_dict = cls.merge_configs(config_dict, cli_overrides)
        
        # Create and validate configuration
        config = TIDEConfig.from_dict(config_dict)
        
        # Save configuration for reproducibility
        config_used_path = Path(config.out_dir) / "config_used.json"
        config.save(config_used_path)
        
        return config
    
    @classmethod
    def load_auto(
        cls,
        default_config: str = "configs/defaults.yaml",
        colab_override: bool = False,
    ) -> TIDEConfig:
        """Automatically load configuration with smart defaults.
        
        Args:
            default_config: Path to default configuration.
            colab_override: If True, apply Colab overrides.
            
        Returns:
            Validated TIDEConfig instance.
        """
        # Load defaults
        config = cls.load(config_path=default_config)
        
        # Apply Colab overrides if requested
        if colab_override:
            colab_path = Path(default_config).parent / "colab.yaml"
            if colab_path.exists():
                colab_config = cls.load_yaml(colab_path)
                config = TIDEConfig.from_dict(
                    cls.merge_configs(config.to_dict(), colab_config)
                )
        
        return config


def setup_logging(config: TIDEConfig) -> None:
    """Configure logging based on configuration.
    
    Args:
        config: TIDE configuration.
    """
    log_file = Path(config.out_dir) / "training.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format=format_string,
        handlers=handlers,
        force=True,
    )
    
    # Reduce noise from other libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    logger.info(f"Logging configured at level {config.log_level}")
    logger.info(f"Logging to file: {log_file}")





def initialize_environment(config: TIDEConfig) -> None:
    """Initialize environment based on configuration.
    
    Sets up logging, seeds, and device settings.
    
    Args:
        config: TIDE configuration.
    """
    setup_logging(config)
    seed_everything(config.seed)
    
    # Log configuration summary
    logger.info("="*60)
    logger.info("TIDE-Lite Configuration")
    logger.info("="*60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.lr}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Output: {config.out_dir}")
    logger.info("="*60)


# Convenience function for backward compatibility
def load_config(
    config_file: Optional[str] = None,
    **kwargs: Any
) -> TIDEConfig:
    """Load configuration with backward compatibility.
    
    Args:
        config_file: Path to configuration file.
        **kwargs: Override parameters.
        
    Returns:
        Validated TIDEConfig instance.
    """
    return ConfigLoader.load(config_path=config_file, overrides=kwargs)
