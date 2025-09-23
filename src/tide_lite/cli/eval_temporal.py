"""Command-line interface for temporal evaluation.

This module provides the CLI for evaluating models on temporal understanding
tasks using TimeQA/TempLAMA datasets with temporal accuracy and consistency metrics.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..eval.eval_temporal import TemporalEvaluator
from ..models.tide_lite import TIDELite
from ..models.baselines import load_baseline

logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup argument parser for temporal evaluation.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="eval-temporal",
        description="Evaluate temporal understanding with TimeQA/TempLAMA datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (default) - show evaluation plan
  python -m tide_lite.cli.eval_temporal --model path/to/model
  
  # Actually run evaluation
  python -m tide_lite.cli.eval_temporal --model path/to/model --run
  
  # Evaluate baseline model
  python -m tide_lite.cli.eval_temporal --model minilm --type baseline --run
  
  # Use custom time window (7 days)
  python -m tide_lite.cli.eval_temporal --model model.pt --time-window-days 7 --run
  
  # Limit samples for quick testing
  python -m tide_lite.cli.eval_temporal --model model.pt --max-samples 100 --run
        """,
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier or path to checkpoint",
    )
    
    parser.add_argument(
        "--type",
        type=str,
        default="tide_lite",
        choices=["tide_lite", "baseline"],
        help="Model type (default: tide_lite)",
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="timeqa",
        choices=["timeqa", "templama"],
        help="Dataset to use (default: timeqa)",
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for dataset cache (default: ./data)",
    )
    
    # Temporal settings
    parser.add_argument(
        "--time-window-days",
        type=float,
        default=30.0,
        help="Time window for temporal accuracy in days (default: 30)",
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="k value for Temporal Accuracy@k metric (default: 5)",
    )
    
    # Data configuration
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None for all)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)",
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file",
    )
    
    # Execution mode
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run evaluation (default is dry run)",
    )
    
    parser.add_argument(
        "--dry-run",
        dest="run",
        action="store_false",
        help="Show evaluation plan without running (default)",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (auto-detect if not specified)",
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    return parser


def evaluate_temporal(
    model_id_or_path: str,
    model_type: str = "tide_lite",
    dataset: str = "timeqa",
    data_dir: str = "./data",
    time_window_days: float = 30.0,
    top_k: int = 5,
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    output_dir: Optional[str] = None,
    save_results: bool = True,
    dry_run: bool = False,
    device: Optional[str] = None,
) -> dict:
    """Evaluate a model on temporal understanding tasks.
    
    Args:
        model_id_or_path: Model identifier or path.
        model_type: Type of model ('tide_lite' or 'baseline').
        dataset: Dataset to use ('timeqa' or 'templama').
        data_dir: Directory for dataset cache.
        time_window_days: Time window for temporal accuracy.
        top_k: k value for Temporal Accuracy@k.
        max_samples: Maximum samples to evaluate.
        batch_size: Batch size for evaluation.
        output_dir: Directory to save results.
        save_results: Whether to save results to JSON.
        dry_run: If True, just print plan without execution.
        device: Device to use (auto-detect if None).
        
    Returns:
        Dictionary with evaluation results.
    """
    if dry_run:
        logger.info("[DRY RUN] Would evaluate model on temporal understanding:")
        logger.info(f"  Model: {model_id_or_path}")
        logger.info(f"  Type: {model_type}")
        logger.info(f"  Dataset: {dataset}")
        logger.info(f"  Time window: {time_window_days} days")
        logger.info(f"  Top-k: {top_k}")
        logger.info(f"  Max samples: {max_samples or 'all'}")
        logger.info(f"  Batch size: {batch_size}")
        if save_results:
            model_name = Path(model_id_or_path).name if "/" in model_id_or_path else model_id_or_path
            output_file = f"results/metrics_temporal_{model_name}.json"
            logger.info(f"  Would save to: {output_file}")
        return {
            "dry_run": True,
            "model": model_id_or_path,
            "dataset": dataset,
            "time_window_days": time_window_days,
        }
    
    # Load model
    logger.info(f"Loading model: {model_id_or_path}")
    
    if model_type == "tide_lite":
        if Path(model_id_or_path).exists():
            model = TIDELite.from_pretrained(model_id_or_path)
            logger.info(f"Loaded TIDE-Lite from {model_id_or_path}")
        else:
            from ..models.tide_lite import TIDELiteConfig
            config = TIDELiteConfig(encoder_name=model_id_or_path)
            model = TIDELite(config)
            logger.info(f"Initialized TIDE-Lite with encoder {model_id_or_path}")
    else:
        model = load_baseline(model_type)
        logger.info(f"Loaded baseline model: {model_type}")
    
    # Setup device
    if device is None:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        import torch
        device = torch.device(device)
    
    # Create evaluator
    evaluator = TemporalEvaluator(
        model=model,
        device=device,
        time_window_seconds=time_window_days * 86400.0,
    )
    
    # Evaluate
    logger.info(f"Evaluating on {dataset} dataset")
    metrics = evaluator.evaluate(
        batch_size=batch_size,
        max_samples=max_samples,
    )
    
    # Prepare results
    results = {
        "model": model_id_or_path,
        "model_type": model_type,
        "dataset": dataset,
        "metrics": {
            "temporal_accuracy_at_1": metrics.temporal_accuracy_at_1,
            "temporal_accuracy_at_5": metrics.temporal_accuracy_at_5,
            "temporal_consistency_score": metrics.temporal_consistency_score,
            "time_window_precision": metrics.time_window_precision,
            "time_drift_mae_days": metrics.time_drift_mae,
            "avg_inference_time_ms": metrics.avg_inference_time_ms,
        },
        "config": {
            "time_window_days": time_window_days,
            "top_k": top_k,
            "batch_size": batch_size,
            "max_samples": max_samples,
            "num_samples_evaluated": metrics.num_samples,
        },
    }
    
    # Print results
    logger.info("=" * 60)
    logger.info("Temporal Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Model: {model_id_or_path}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Samples: {metrics.num_samples}")
    logger.info("-" * 40)
    logger.info(f"Temporal Accuracy@1: {metrics.temporal_accuracy_at_1:.4f}")
    logger.info(f"Temporal Accuracy@5: {metrics.temporal_accuracy_at_5:.4f}")
    logger.info(f"Temporal Consistency Score: {metrics.temporal_consistency_score:.4f}")
    logger.info(f"Time Window Precision: {metrics.time_window_precision:.4f}")
    logger.info(f"Time Drift MAE: {metrics.time_drift_mae:.1f} days")
    logger.info("-" * 40)
    logger.info(f"Avg Inference Time: {metrics.avg_inference_time_ms:.2f} ms")
    logger.info("=" * 60)
    
    # Save results
    if save_results and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename based on model
        model_name = Path(model_id_or_path).name if "/" in model_id_or_path else model_id_or_path
        model_name = model_name.replace("/", "_")
        output_file = output_path / f"metrics_temporal_{model_name}.json"
        
        import json
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
    
    return results


def main() -> int:
    """Main entry point for temporal evaluation.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    try:
        # Evaluate
        results = evaluate_temporal(
            model_id_or_path=args.model,
            model_type=args.type,
            dataset=args.dataset,
            data_dir=args.data_dir,
            time_window_days=args.time_window_days,
            top_k=args.top_k,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            output_dir=args.output_dir if not args.no_save else None,
            save_results=not args.no_save,
            dry_run=not args.run,
            device=args.device,
        )
        
        if not args.run:
            logger.info("Dry run complete. Use --run to actually execute evaluation.")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
