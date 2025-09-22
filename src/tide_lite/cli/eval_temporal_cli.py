"""Command-line interface for temporal evaluation."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from ..eval.eval_temporal import TemporalEvaluator
from ..eval.eval_stsb import load_model_for_evaluation
from ..models.baselines import load_minilm_baseline
from ..utils.config import setup_logging

logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup argument parser for temporal evaluation CLI.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="tide-lite-eval-temporal",
        description="Evaluate temporal understanding and consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate trained TIDE-Lite model
  tide-lite-eval-temporal --model-path results/run_123/checkpoints/final.pt
  
  # Evaluate with custom time window (7 days)
  tide-lite-eval-temporal --model-path model.pt --time-window-days 7
  
  # Compare with baseline
  tide-lite-eval-temporal --model-path model.pt --compare-baseline
  
  # Limit samples for quick testing
  tide-lite-eval-temporal --model-path model.pt --max-samples 1000
  
  # Dry run to see evaluation plan
  tide-lite-eval-temporal --model-path model.pt --dry-run
""",
    )
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-path",
        type=Path,
        help="Path to trained model checkpoint or directory",
    )
    model_group.add_argument(
        "--baseline",
        type=str,
        choices=["minilm", "e5-base", "bge-base"],
        help="Use a baseline model instead of trained TIDE-Lite",
    )
    
    # Temporal settings
    parser.add_argument(
        "--time-window-days",
        type=float,
        default=30.0,
        help="Time window for temporal accuracy in days (default: 30)",
    )
    
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
    
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also evaluate baseline for comparison",
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation"),
        help="Directory to save evaluation results (default: results/evaluation)",
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for output files (auto-generated if not specified)",
    )
    
    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps"],
        default=None,
        help="Device for computation (auto-detect if not specified)",
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers (default: 2)",
    )
    
    # Misc
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print evaluation plan without executing",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    return parser


def print_evaluation_plan(args: argparse.Namespace) -> None:
    """Print evaluation plan for dry run.
    
    Args:
        args: Parsed command-line arguments.
    """
    print("\n" + "=" * 60)
    print("TEMPORAL EVALUATION PLAN")
    print("=" * 60)
    
    print("\n📊 Model Configuration:")
    if args.model_path:
        print(f"  • Model: TIDE-Lite from {args.model_path}")
    else:
        print(f"  • Model: Baseline {args.baseline}")
    
    print("\n⏰ Temporal Settings:")
    print(f"  • Time window: {args.time_window_days} days")
    print(f"  • Max samples: {args.max_samples or 'all'}")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Compare baseline: {args.compare_baseline}")
    
    print("\n💾 Output Configuration:")
    print(f"  • Output dir: {args.output_dir}")
    print(f"  • Model name: {args.model_name or 'auto-generated'}")
    
    print("\n🖥️ Hardware:")
    print(f"  • Device: {args.device or 'auto-detect'}")
    print(f"  • Workers: {args.num_workers}")
    
    print("\n📈 Expected Metrics:")
    print("  • Temporal Accuracy@1 and @5")
    print("  • Temporal Consistency Score (correlation)")
    print("  • Time Window Precision")
    print("  • Time Drift (MAE in days)")
    print("  • Inference latency")
    
    print("\n⚠️ Note: Using TimeQA-lite surrogate dataset")
    print("  For production, replace with actual TimeQA dataset")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No actual evaluation will occur")
    
    print("\n" + "=" * 60 + "\n")


def main() -> int:
    """Main entry point for temporal evaluation CLI.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Print evaluation plan
    print_evaluation_plan(args)
    
    if args.dry_run:
        logger.info("Dry run complete. Exiting.")
        return 0
    
    try:
        # Setup device
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {device}")
        
        # Load model
        if args.model_path:
            logger.info(f"Loading TIDE-Lite model from {args.model_path}")
            model = load_model_for_evaluation(args.model_path, device)
            model_name = args.model_name or f"tide_lite_{args.model_path.stem}"
        else:
            logger.info(f"Loading baseline model: {args.baseline}")
            if args.baseline == "minilm":
                model = load_minilm_baseline()
            elif args.baseline == "e5-base":
                from ..models.baselines import load_e5_base_baseline
                model = load_e5_base_baseline()
            else:  # bge-base
                from ..models.baselines import load_bge_base_baseline
                model = load_bge_base_baseline()
            model_name = args.model_name or f"baseline_{args.baseline}"
        
        # Create evaluator
        evaluator = TemporalEvaluator(
            model=model,
            device=device,
            time_window_seconds=args.time_window_days * 86400.0,
        )
        
        # Evaluate
        logger.info("Starting temporal evaluation")
        metrics = evaluator.evaluate(
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
        
        # Save results
        metrics_file = evaluator.save_results(
            metrics,
            output_dir=args.output_dir,
            model_name=model_name,
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("TEMPORAL EVALUATION RESULTS")
        print("=" * 60)
        print(f"\n📊 Metrics for {model_name}:")
        print(f"  • Temporal Accuracy@1: {metrics.temporal_accuracy_at_1:.4f}")
        print(f"  • Temporal Accuracy@5: {metrics.temporal_accuracy_at_5:.4f}")
        print(f"  • Temporal Consistency: {metrics.temporal_consistency_score:.4f}")
        print(f"  • Time Window Precision: {metrics.time_window_precision:.4f}")
        print(f"  • Time Drift MAE: {metrics.time_drift_mae:.1f} days")
        
        print(f"\n⏱️ Performance:")
        print(f"  • Avg inference time: {metrics.avg_inference_time_ms:.2f} ms")
        print(f"  • Total time: {metrics.total_eval_time_s:.2f} seconds")
        print(f"  • Samples evaluated: {metrics.num_samples}")
        
        print(f"\n💾 Results saved to: {metrics_file}")
        
        # Compare with baseline if requested
        if args.compare_baseline and args.model_path:
            print("\n" + "-" * 60)
            print("Comparing with baseline...")
            
            baseline_model = load_minilm_baseline()
            comparison = evaluator.compare_temporal_awareness(baseline_model)
            
            # Save comparison
            comparison_file = args.output_dir / f"comparison_temporal_{model_name}.json"
            with open(comparison_file, "w") as f:
                json.dump(
                    {
                        "tide_lite": {
                            "accuracy_at_1": comparison["tide_lite"].temporal_accuracy_at_1,
                            "consistency": comparison["tide_lite"].temporal_consistency_score,
                            "drift_mae": comparison["tide_lite"].time_drift_mae,
                        },
                        "baseline": {
                            "accuracy_at_1": comparison["baseline"].temporal_accuracy_at_1,
                            "consistency": comparison["baseline"].temporal_consistency_score,
                            "drift_mae": comparison["baseline"].time_drift_mae,
                        },
                        "improvements": {
                            "accuracy_at_1": (
                                comparison["tide_lite"].temporal_accuracy_at_1 - 
                                comparison["baseline"].temporal_accuracy_at_1
                            ),
                            "consistency": (
                                comparison["tide_lite"].temporal_consistency_score - 
                                comparison["baseline"].temporal_consistency_score
                            ),
                            "drift_reduction": (
                                comparison["baseline"].time_drift_mae - 
                                comparison["tide_lite"].time_drift_mae
                            ),
                        },
                    },
                    f,
                    indent=2,
                )
            
            print(f"Comparison saved to: {comparison_file}")
        
        print("\n✅ Evaluation complete!")
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        logger.exception("Evaluation failed with exception")
        return 1


if __name__ == "__main__":
    sys.exit(main())
