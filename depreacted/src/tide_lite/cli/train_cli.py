"""Command-line interface for TIDE-Lite training.

This module provides the CLI for training TIDE-Lite models using the unified
configuration system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from ..models.tide_lite import TIDELite, TIDELiteConfig
from ..train.trainer import TIDELiteTrainer, TrainingConfig  
from ..utils.config import ConfigLoader, TIDEConfig, initialize_environment

logger = logging.getLogger(__name__)


def create_model_config(tide_config: TIDEConfig) -> TIDELiteConfig:
    """Convert TIDEConfig to TIDELiteConfig for model initialization.
    
    Args:
        tide_config: Unified TIDE configuration.
        
    Returns:
        Model-specific configuration.
    """
    return TIDELiteConfig(
        encoder_name=tide_config.model_name,
        hidden_dim=tide_config.hidden_dim,
        time_encoding_dim=tide_config.time_dims,
        mlp_hidden_dim=tide_config.time_mlp_hidden,
        mlp_dropout=tide_config.mlp_dropout,
        freeze_encoder=tide_config.freeze_encoder,
        pooling_strategy=tide_config.pooling_strategy,
        gate_activation=tide_config.gate_activation,
    )


def create_training_config(tide_config: TIDEConfig) -> TrainingConfig:
    """Convert TIDEConfig to TrainingConfig for trainer.
    
    Args:
        tide_config: Unified TIDE configuration.
        
    Returns:
        Training-specific configuration.
    """
    return TrainingConfig(
        encoder_name=tide_config.model_name,
        time_encoding_dim=tide_config.time_dims,
        mlp_hidden_dim=tide_config.time_mlp_hidden,
        mlp_dropout=tide_config.mlp_dropout,
        gate_activation=tide_config.gate_activation,
        batch_size=tide_config.batch_size,
        eval_batch_size=tide_config.eval_batch_size,
        max_seq_length=tide_config.max_seq_len,
        num_workers=tide_config.num_workers,
        num_epochs=tide_config.epochs,
        learning_rate=tide_config.lr,
        warmup_steps=tide_config.warmup_steps,
        weight_decay=tide_config.weight_decay,
        gradient_clip=tide_config.gradient_clip,
        temporal_weight=tide_config.consistency_weight,
        preservation_weight=tide_config.preservation_weight,
        tau_seconds=tide_config.tau_seconds,
        use_amp=tide_config.use_amp,
        save_every=tide_config.save_every,
        eval_every=tide_config.eval_every,
        output_dir=tide_config.out_dir,
        cache_dir=tide_config.cache_dir,
        seed=tide_config.seed,
        dry_run=tide_config.dry_run,
        skip_temporal=not tide_config.temporal_enabled,
    )


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup argument parser for training CLI.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="tide-train",
        description="Train TIDE-Lite temporal embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/defaults.yaml"),
        help="Path to YAML configuration file (default: configs/defaults.yaml)",
    )
    
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Use Colab configuration overrides",
    )
    
    # Common overrides
    parser.add_argument("--model-name", type=str, help="Override model name")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--out-dir", type=str, help="Override output directory")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Override device")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without training")
    parser.add_argument("--seed", type=int, help="Override random seed")
    
    return parser


def main() -> int:
    """Main entry point for training CLI.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # Prepare CLI overrides
        cli_overrides: Dict[str, Any] = {}
        
        # Map CLI arguments to config keys
        if args.model_name is not None:
            cli_overrides["model_name"] = args.model_name
        if args.batch_size is not None:
            cli_overrides["batch_size"] = args.batch_size
        if args.epochs is not None:
            cli_overrides["epochs"] = args.epochs
        if args.lr is not None:
            cli_overrides["lr"] = args.lr
        if args.out_dir is not None:
            cli_overrides["out_dir"] = args.out_dir
        if args.device is not None:
            cli_overrides["device"] = args.device
        if args.dry_run:
            cli_overrides["dry_run"] = True
        if args.seed is not None:
            cli_overrides["seed"] = args.seed
        
        # Load configuration with overrides
        if args.colab:
            # Load defaults first, then apply colab overrides
            config = ConfigLoader.load_auto(
                default_config=str(args.config),
                colab_override=True
            )
            # Apply CLI overrides on top
            if cli_overrides:
                config_dict = ConfigLoader.merge_configs(config.to_dict(), cli_overrides)
                config = TIDEConfig.from_dict(config_dict)
        else:
            config = ConfigLoader.load(
                config_path=args.config,
                cli_args=cli_overrides
            )
        
        # Initialize environment (logging, seeds, etc.)
        initialize_environment(config)
        
        # Dry run information
        if config.dry_run:
            logger.info("=" * 60)
            logger.info("DRY RUN MODE - No training will be performed")
            logger.info("=" * 60)
        
        # Create model configuration
        model_config = create_model_config(config)
        
        # Initialize model
        logger.info("Initializing TIDE-Lite model...")
        model = TIDELite(model_config)
        
        # Log parameter summary
        extra_params = model.count_extra_parameters()
        logger.info(f"Extra TIDE parameters: {extra_params:,}")
        
        # Create training configuration
        training_config = create_training_config(config)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = TIDELiteTrainer(training_config, model)
        
        # Run training or dry run
        if config.dry_run:
            logger.info("Executing dry run...")
            trainer.train()  # Will exit early in dry-run mode
            logger.info("Dry run complete.")
        else:
            logger.info("Starting training...")
            logger.info("Press Ctrl+C to interrupt")
            trainer.train()
            logger.info("Training complete!")
            logger.info(f"Results saved to: {config.out_dir}")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
