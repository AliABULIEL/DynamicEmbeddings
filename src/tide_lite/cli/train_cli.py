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
            # Handle nested keys (e.g., "model.hidden_dim")
            if "." in key:
                parts = key.split(".")
                current = merged
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
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
        epilog="""
Examples:
  # Train with default configuration
  tide-lite-train
  
  # Train with custom config file
  tide-lite-train --config configs/custom.yaml
  
  # Override specific parameters
  tide-lite-train --batch-size 64 --learning-rate 1e-4 --num-epochs 5
  
  # Dry run to see training plan
  tide-lite-train --dry-run
  
  # Custom output directory
  tide-lite-train --output-dir results/experiment1
""",
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/defaults.yaml"),
        help="Path to YAML configuration file (default: configs/defaults.yaml)",
    )
    
    # Output paths
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/run-<timestamp>)",
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for checkpoints (default: <output-dir>/checkpoints)",
    )
    
    # Model configuration
    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--encoder-name",
        type=str,
        default=None,
        help="HuggingFace encoder model name",
    )
    
    model_group.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension of encoder",
    )
    
    model_group.add_argument(
        "--time-encoding-dim",
        type=int,
        default=None,
        help="Dimension of temporal encoding",
    )
    
    model_group.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension of temporal MLP",
    )
    
    model_group.add_argument(
        "--mlp-dropout",
        type=float,
        default=None,
        help="Dropout probability in temporal MLP",
    )
    
    model_group.add_argument(
        "--no-freeze-encoder",
        action="store_true",
        help="Don't freeze encoder weights (allow fine-tuning)",
    )
    
    # Training configuration
    train_group = parser.add_argument_group("training")
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )
    
    train_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Evaluation batch size",
    )
    
    train_group.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Peak learning rate",
    )
    
    train_group.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Number of warmup steps",
    )
    
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="AdamW weight decay",
    )
    
    train_group.add_argument(
        "--gradient-clip",
        type=float,
        default=None,
        help="Gradient clipping norm",
    )
    
    # Loss configuration
    loss_group = parser.add_argument_group("loss")
    loss_group.add_argument(
        "--temporal-weight",
        type=float,
        default=None,
        help="Weight for temporal consistency loss",
    )
    
    loss_group.add_argument(
        "--preservation-weight",
        type=float,
        default=None,
        help="Weight for base embedding preservation",
    )
    
    loss_group.add_argument(
        "--tau-seconds",
        type=float,
        default=None,
        help="Time constant for temporal consistency (seconds)",
    )
    
    # Hardware configuration
    hw_group = parser.add_argument_group("hardware")
    hw_group.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    
    hw_group.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker threads",
    )
    
    # Checkpointing
    ckpt_group = parser.add_argument_group("checkpointing")
    ckpt_group.add_argument(
        "--save-every-n-steps",
        type=int,
        default=None,
        help="Checkpoint frequency (steps)",
    )
    
    ckpt_group.add_argument(
        "--eval-every-n-steps",
        type=int,
        default=None,
        help="Evaluation frequency (steps)",
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging verbosity",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without actual training",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="TIDE-Lite v0.1.0",
    )
    
    return parser


def args_to_config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert parsed arguments to configuration dictionary.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Configuration dictionary with non-None values.
    """
    config_dict = {}
    
    # Direct mappings
    simple_mappings = {
        "output_dir": "output_dir",
        "checkpoint_dir": "checkpoint_dir",
        "encoder_name": "encoder_name",
        "hidden_dim": "hidden_dim",
        "time_encoding_dim": "time_encoding_dim",
        "mlp_hidden_dim": "mlp_hidden_dim",
        "mlp_dropout": "mlp_dropout",
        "batch_size": "batch_size",
        "eval_batch_size": "eval_batch_size",
        "num_epochs": "num_epochs",
        "learning_rate": "learning_rate",
        "warmup_steps": "warmup_steps",
        "weight_decay": "weight_decay",
        "gradient_clip": "gradient_clip",
        "temporal_weight": "temporal_weight",
        "preservation_weight": "preservation_weight",
        "tau_seconds": "tau_seconds",
        "num_workers": "num_workers",
        "save_every_n_steps": "save_every_n_steps",
        "eval_every_n_steps": "eval_every_n_steps",
        "seed": "seed",
        "log_level": "log_level",
        "dry_run": "dry_run",
    }
    
    for arg_name, config_name in simple_mappings.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config_dict[config_name] = value
    
    # Boolean inversions
    if args.no_freeze_encoder:
        config_dict["freeze_encoder"] = False
    
    if args.no_amp:
        config_dict["use_amp"] = False
    
    return config_dict


def generate_run_id() -> str:
    """Generate unique run identifier based on timestamp.
    
    Returns:
        Run ID string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}"


def print_training_plan(config: TrainingConfig) -> None:
    """Print training execution plan.
    
    Args:
        config: Training configuration.
    """
    print("\n" + "=" * 60)
    print("TIDE-LITE TRAINING PLAN")
    print("=" * 60)
    
    print("\n📊 Model Configuration:")
    print(f"  • Encoder: {config.encoder_name}")
    print(f"  • Hidden dim: {config.hidden_dim}")
    print(f"  • Time encoding dim: {config.time_encoding_dim}")
    print(f"  • MLP hidden dim: {config.mlp_hidden_dim}")
    print(f"  • Frozen encoder: {config.freeze_encoder}")
    
    print("\n🔄 Training Configuration:")
    print(f"  • Epochs: {config.num_epochs}")
    print(f"  • Batch size: {config.batch_size}")
    print(f"  • Learning rate: {config.learning_rate}")
    print(f"  • Warmup steps: {config.warmup_steps}")
    print(f"  • Mixed precision: {config.use_amp}")
    
    print("\n⚖️ Loss Configuration:")
    print(f"  • Temporal weight: {config.temporal_weight}")
    print(f"  • Preservation weight: {config.preservation_weight}")
    print(f"  • Tau (days): {config.tau_seconds / 86400:.1f}")
    
    print("\n💾 Output Configuration:")
    print(f"  • Output dir: {config.output_dir}")
    print(f"  • Checkpoint dir: {config.checkpoint_dir}")
    print(f"  • Save frequency: every {config.save_every_n_steps} steps")
    print(f"  • Eval frequency: every {config.eval_every_n_steps} steps")
    
    if config.dry_run:
        print("\n⚠️  DRY RUN MODE - No actual training will occur")
    
    print("\n" + "=" * 60 + "\n")


def main() -> int:
    """Main entry point for training CLI.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load base configuration
        if args.config.exists():
            base_config = load_yaml_config(args.config)
        else:
            print(f"Warning: Config file {args.config} not found, using defaults")
            base_config = {}
        
        # Convert args to config dict
        cli_overrides = args_to_config_dict(args)
        
        # Generate output directory if not specified
        if "output_dir" not in cli_overrides and "output_dir" not in base_config:
            cli_overrides["output_dir"] = f"results/{generate_run_id()}"
        
        # Merge configurations
        final_config = merge_configs(base_config, cli_overrides)
        
        # Create configuration objects
        training_config = TrainingConfig(**final_config)
        
        # Extract model config parameters
        model_config_params = {
            "encoder_name": training_config.encoder_name,
            "hidden_dim": training_config.hidden_dim,
            "time_encoding_dim": training_config.time_encoding_dim,
            "mlp_hidden_dim": training_config.mlp_hidden_dim,
            "mlp_dropout": training_config.mlp_dropout,
            "freeze_encoder": training_config.freeze_encoder,
            "pooling_strategy": training_config.pooling_strategy,
            "gate_activation": training_config.gate_activation,
        }
        model_config = TIDELiteConfig(**model_config_params)
        
        # Print training plan
        print_training_plan(training_config)
        
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
        logger.exception("Training failed with exception")
        return 1


if __name__ == "__main__":
    sys.exit(main())
