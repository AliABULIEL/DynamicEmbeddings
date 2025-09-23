"""Command-line interface for STS-B evaluation.

This module provides the argparse-based CLI for evaluating models
on the STS-B benchmark with Spearman correlation and bootstrap CI.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from tide_lite.eval.eval_stsb import evaluate_stsb

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
        description="Evaluate models on STS-B benchmark with bootstrap confidence intervals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "model",
        type=str,
        help="Model identifier or path to saved model",
    )
    model_group.add_argument(
        "--model-type",
        type=str,
        default="tide_lite",
        choices=["tide_lite", "minilm", "e5-base", "bge-base"],
        help="Type of model to evaluate",
    )
    model_group.add_argument(
        "--use-timestamps",
        action="store_true",
        default=False,
        help="Use temporal modulation for TIDE-Lite (if applicable)",
    )
    
    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation Configuration")
    eval_group.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on",
    )
    eval_group.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    eval_group.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization",
    )
    
    # Bootstrap arguments
    bootstrap_group = parser.add_argument_group("Bootstrap Configuration")
    bootstrap_group.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for confidence intervals",
    )
    bootstrap_group.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap CI",
    )
    bootstrap_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap sampling",
    )
    
    # Output arguments
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    output_group.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Don't save results to JSON file",
    )
    
    # Execution mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print evaluation plan without execution (default)",
    )
    parser.add_argument(
        "--run",
        dest="dry_run",
        action="store_false",
        help="Actually run evaluation (disable dry-run)",
    )
    
    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", None],
        help="Device to use (auto-detect if not specified)",
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
    """Main entry point for STS-B evaluation."""
    args = parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Print header
    logger.info("=" * 60)
    logger.info("STS-B Evaluation CLI")
    logger.info("=" * 60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("")
    
    # Print configuration
    logger.info("Configuration:")
    logger.info("-" * 40)
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Type: {args.model_type}")
    logger.info(f"  Split: {args.split}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max sequence length: {args.max_seq_length}")
    logger.info(f"  Use timestamps: {args.use_timestamps}")
    logger.info(f"  Bootstrap iterations: {args.n_bootstrap}")
    logger.info(f"  Confidence level: {args.confidence}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Save results: {not args.no_save}")
    logger.info("-" * 40)
    logger.info("")
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE ENABLED")
        logger.info("This will print the evaluation plan without executing.")
        logger.info("To actually evaluate, use: --run")
        logger.info("")
    
    # Run evaluation
    results = evaluate_stsb(
        model_id_or_path=args.model,
        model_type=args.model_type,
        split=args.split,
        use_timestamps=args.use_timestamps,
        n_bootstrap=args.n_bootstrap,
        batch_size=args.batch_size,
        output_dir=args.output_dir if not args.no_save else None,
        dry_run=args.dry_run,
    )
    
    if not args.dry_run:
        # Print summary
        if "metrics" in results:
            metrics = results["metrics"]
            logger.info("")
            logger.info("📊 Evaluation Summary:")
            logger.info(f"  Spearman ρ: {metrics['spearman_rho']:.4f}")
            logger.info(f"    95% CI: [{metrics['spearman_ci_lower']:.4f}, {metrics['spearman_ci_upper']:.4f}]")
            logger.info(f"  Pearson r: {metrics['pearson_r']:.4f}")
            logger.info(f"    95% CI: [{metrics['pearson_ci_lower']:.4f}, {metrics['pearson_ci_upper']:.4f}]")
            logger.info(f"  MSE: {metrics['mse']:.4f}")
            logger.info(f"  Samples: {metrics['num_samples']}")
        
        logger.info("")
        logger.info("✅ Evaluation complete!")
        
        if not args.no_save:
            model_name = Path(args.model).name if "/" in args.model else args.model
            model_name = model_name.replace("/", "_")
            output_file = f"{args.output_dir}/metrics_stsb_{model_name}.json"
            logger.info(f"Results saved to: {output_file}")
    else:
        logger.info("")
        logger.info("✅ Dry run complete! Use --run to execute evaluation.")


if __name__ == "__main__":
    main()
