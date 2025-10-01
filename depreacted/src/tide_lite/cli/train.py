"""Command-line interface for training TIDE-Lite.

This module provides the argparse-based CLI for training TIDE-Lite models
with configurable hyperparameters and dry-run support.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from tide_lite.train.trainer import TrainingConfig, TIDELiteTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train TIDE-Lite model on STS-B dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--encoder-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base encoder model from HuggingFace",
    )
    model_group.add_argument(
        "--time-encoding-dim",
        type=int,
        default=32,
        help="Dimension of temporal encoding (must be even)",
    )
    model_group.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension of temporal MLP",
    )
    model_group.add_argument(
        "--mlp-dropout",
        type=float,
        default=0.1,
        help="Dropout rate in temporal MLP",
    )
    model_group.add_argument(
        "--gate-activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "tanh"],
        help="Activation function for temporal gating",
    )
    
    # Training arguments
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    train_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Evaluation batch size",
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Peak learning rate",
    )
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW",
    )
    train_group.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    
    # Loss arguments
    loss_group = parser.add_argument_group("Loss Configuration")
    loss_group.add_argument(
        "--temporal-weight",
        type=float,
        default=0.1,
        help="Weight for temporal consistency loss",
    )
    loss_group.add_argument(
        "--preservation-weight",
        type=float,
        default=0.05,
        help="Weight for preservation loss",
    )
    loss_group.add_argument(
        "--tau-seconds",
        type=float,
        default=86400.0,
        help="Time constant for temporal consistency (seconds)",
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization",
    )
    data_group.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers",
    )
    data_group.add_argument(
        "--cache-dir",
        type=str,
        default="./data",
        help="Cache directory for datasets",
    )
    
    # Optimization arguments
    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision training",
    )
    opt_group.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Disable automatic mixed precision",
    )
    opt_group.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (0 to disable)",
    )
    
    # Checkpointing arguments
    checkpoint_group = parser.add_argument_group("Checkpointing")
    checkpoint_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (auto-generated if not specified)",
    )
    checkpoint_group.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save checkpoint every N steps (0 for epoch only)",
    )
    checkpoint_group.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Evaluate every N steps (0 for epoch only)",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print training plan without execution (default)",
    )
    parser.add_argument(
        "--run",
        dest="dry_run",
        action="store_false",
        help="Actually run training (disable dry-run)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    return parser.parse_args()


def main():
    """Main entry point for training."""
    args = parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Print system info
    logger.info("=" * 60)
    logger.info("TIDE-Lite Training CLI")
    logger.info("=" * 60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("")
    
    # Create training configuration from arguments
    config = TrainingConfig(
        # Model
        encoder_name=args.encoder_name,
        time_encoding_dim=args.time_encoding_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
        gate_activation=args.gate_activation,
        # Training
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        # Loss
        temporal_weight=args.temporal_weight,
        preservation_weight=args.preservation_weight,
        tau_seconds=args.tau_seconds,
        # Data
        max_seq_length=args.max_seq_length,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        # Optimization
        use_amp=args.use_amp,
        gradient_clip=args.gradient_clip,
        # Checkpointing
        output_dir=args.output_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        # Other
        seed=args.seed,
        dry_run=args.dry_run,
    )
    
    # Print configuration
    logger.info("Configuration:")
    logger.info("-" * 40)
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 40)
    logger.info("")
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE ENABLED")
        logger.info("This will print the training plan without executing.")
        logger.info("To actually train, use: --run")
        logger.info("")
    
    # Create trainer and run
    trainer = TIDELiteTrainer(config)
    trainer.train()
    
    if not args.dry_run:
        logger.info(f"\n✅ Training complete! Results saved to: {config.output_dir}")
    else:
        logger.info("\n✅ Dry run complete! Use --run to execute training.")


if __name__ == "__main__":
    main()
