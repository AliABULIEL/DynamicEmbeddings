"""Command-line interface for TIDE-Lite training.

This module provides the argparse-based CLI for training TIDE-Lite models
with support for configuration files and command-line overrides.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..models.tide_lite import TIDELite, TIDELiteConfig
from ..train.trainer import TIDETrainer, TrainingConfig

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Dictionary of configuration parameters.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def merge_configs(
    base_config: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge configuration dictionaries with overrides.
    
    Args:
        base_config: Base configuration dictionary.
        overrides: Override values (from CLI args).
        
    Returns:
        Merged configuration dictionary.
    """
    merged = base_config.copy()
    
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    
    return merged


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup argument parser for training CLI.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="tide-lite-train",
        description="Train TIDE-Lite temporal embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/defaults.yaml"),
        help="Path to YAML configuration file",
    )
    
    # Output paths
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    
    # Model configuration
    parser.add_argument(
        "--encoder-name",
        type=str,
        default=None,
        help="HuggingFace encoder model name",
    )
    
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension of temporal MLP",
    )
    
    # Training configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Peak learning rate",
    )
    
    parser.add_argument(
        "--temporal-weight",
        type=float,
        default=None,
        help="Weight for temporal consistency loss",
    )
    
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable automatic mixed precision",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without actual training",
    )
    
    return parser


def split_configs(config_dict: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split configuration into model and training configs.
    
    Args:
        config_dict: Combined configuration dictionary.
        
    Returns:
        Tuple of (model_config_dict, training_config_dict).
    """
    # Model config fields
    model_fields = {
        "encoder_name", "hidden_dim", "time_encoding_dim", 
        "mlp_hidden_dim", "mlp_dropout", "gate_activation",
        "freeze_encoder", "pooling_strategy"
    }
    
    # Training config fields
    training_fields = {
        "batch_size", "eval_batch_size", "max_seq_length", "num_workers",
        "num_epochs", "learning_rate", "warmup_steps", "weight_decay", 
        "gradient_clip", "temporal_weight", "preservation_weight", 
        "tau_seconds", "use_amp", "save_every_n_steps", "eval_every_n_steps",
        "output_dir", "checkpoint_dir", "seed", "log_level", "dry_run"
    }
    
    model_config = {k: v for k, v in config_dict.items() if k in model_fields}
    training_config = {k: v for k, v in config_dict.items() if k in training_fields}
    
    return model_config, training_config


def main() -> int:
    """Main entry point for training CLI.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load base config from file if exists
        base_config = {}
        if args.config.exists():
            base_config = load_yaml_config(args.config)
        else:
            print(f"Warning: Config file {args.config} not found, using defaults")
        
        # Convert args to config dict (only non-None values)
        cli_overrides = {}
        for key in ["output_dir", "encoder_name", "mlp_hidden_dim", "batch_size", 
                    "num_epochs", "learning_rate", "temporal_weight", "dry_run"]:
            value = getattr(args, key.replace("-", "_"), None)
            if value is not None:
                cli_overrides[key.replace("-", "_")] = value
        
        if args.use_amp:
            cli_overrides["use_amp"] = True
        
        # Merge configurations
        final_config = merge_configs(base_config, cli_overrides)
        
        # Generate output directory if not specified
        if "output_dir" not in final_config:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_config["output_dir"] = f"results/run_{timestamp}"
        
        # Split into model and training configs
        model_config_dict, training_config_dict = split_configs(final_config)
        
        # Create config objects with defaults
        model_config = TIDELiteConfig(**model_config_dict)
        training_config = TrainingConfig(**training_config_dict)
        
        # Print training plan
        print("\n" + "=" * 60)
        print("TIDE-LITE TRAINING PLAN")
        print("=" * 60)
        print(f"\n📊 Model: {model_config.encoder_name}")
        print(f"   Extra params: ~{model_config.mlp_hidden_dim * 600} (estimated)")
        print(f"\n🔄 Training: {training_config.num_epochs} epochs")
        print(f"   Batch size: {training_config.batch_size}")
        print(f"   Learning rate: {training_config.learning_rate}")
        print(f"\n💾 Output: {training_config.output_dir}")
        
        if training_config.dry_run:
            print("\n⚠️  DRY RUN MODE - No actual training will occur")
        
        print("\n" + "=" * 60 + "\n")
        
        # Initialize model
        print("Initializing TIDE-Lite model...")
        model = TIDELite(model_config)
        
        # Print parameter summary
        param_summary = model.get_parameter_summary()
        print(f"  • Total parameters: {param_summary['total_params']:,}")
        print(f"  • Trainable parameters: {param_summary['trainable_params']:,}")
        print(f"  • Extra TIDE parameters: {param_summary['extra_params']:,}")
        
        # Initialize trainer
        print("\nInitializing trainer...")
        trainer = TIDETrainer(model, training_config)
        
        # Run training or dry run
        if training_config.dry_run:
            print("\nExecuting dry run...")
            summary = trainer.dry_run_summary()
            print("\nDry run complete. Check output directory for summary.")
        else:
            print("\nStarting training...")
            print("Press Ctrl+C to interrupt\n")
            metrics = trainer.train()
            print("\n✅ Training complete!")
            print(f"Final validation Spearman: {metrics['final_val_spearman']:.4f}")
            print(f"Results saved to: {training_config.output_dir}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
