"""Command-line interface for Quora retrieval evaluation."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from ..eval.retrieval_quora import QuoraRetrievalEvaluator
from ..eval.eval_stsb import load_model_for_evaluation
from ..models.baselines import load_minilm_baseline
from ..utils.config import setup_logging

logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup argument parser for Quora retrieval evaluation CLI.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="tide-lite-eval-quora",
        description="Evaluate models on Quora duplicate questions retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate trained TIDE-Lite model
  tide-lite-eval-quora --model-path results/run_123/checkpoints/final.pt
  
  # Evaluate with IVFFlat index for speed
  tide-lite-eval-quora --model-path model.pt --index-type IVFFlat
  
  # Evaluate baseline model
  tide-lite-eval-quora --baseline e5-base
  
  # Limit corpus/query size for quick testing
  tide-lite-eval-quora --model-path model.pt --max-corpus 10000 --max-queries 1000
  
  # Dry run to see evaluation plan
  tide-lite-eval-quora --model-path model.pt --dry-run
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
    
    # Retrieval settings
    parser.add_argument(
        "--index-type",
        type=str,
        default="Flat",
        choices=["Flat", "IVFFlat"],
        help="FAISS index type (default: Flat for exact search)",
    )
    
    parser.add_argument(
        "--max-corpus",
        type=int,
        default=None,
        help="Maximum corpus size (None for all)",
    )
    
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries (None for all)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for encoding (default: 128)",
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
    print("QUORA RETRIEVAL EVALUATION PLAN")
    print("=" * 60)
    
    print("\n📊 Model Configuration:")
    if args.model_path:
        print(f"  • Model: TIDE-Lite from {args.model_path}")
    else:
        print(f"  • Model: Baseline {args.baseline}")
    
    print("\n🔍 Retrieval Settings:")
    print(f"  • Index type: {args.index_type}")
    print(f"  • Max corpus: {args.max_corpus or 'all'}")
    print(f"  • Max queries: {args.max_queries or 'all'}")
    print(f"  • Batch size: {args.batch_size}")
    
    print("\n💾 Output Configuration:")
    print(f"  • Output dir: {args.output_dir}")
    print(f"  • Model name: {args.model_name or 'auto-generated'}")
    
    print("\n🖥️ Hardware:")
    print(f"  • Device: {args.device or 'auto-detect'}")
    print(f"  • Workers: {args.num_workers}")
    
    print("\n📈 Expected Metrics:")
    print("  • nDCG@10 (Normalized Discounted Cumulative Gain)")
    print("  • Recall@10, Recall@5, Recall@1")
    print("  • Mean Reciprocal Rank (MRR)")
    print("  • Query latency statistics")
    print("  • Index build time")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No actual evaluation will occur")
    
    print("\n" + "=" * 60 + "\n")


def main() -> int:
    """Main entry point for Quora retrieval evaluation CLI.
    
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
        evaluator = QuoraRetrievalEvaluator(
            model=model,
            index_type=args.index_type,
            device=device,
            use_temporal=False,  # No temporal info in Quora
        )
        
        # Evaluate
        logger.info("Starting Quora retrieval evaluation")
        metrics = evaluator.evaluate(
            max_corpus_size=args.max_corpus,
            max_queries=args.max_queries,
        )
        
        # Save results
        metrics_file = evaluator.save_results(
            metrics,
            output_dir=args.output_dir,
            model_name=model_name,
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("QUORA RETRIEVAL EVALUATION RESULTS")
        print("=" * 60)
        print(f"\n📊 Metrics for {model_name}:")
        print(f"  • nDCG@10: {metrics.ndcg_at_10:.4f}")
        print(f"  • Recall@10: {metrics.recall_at_10:.4f}")
        print(f"  • Recall@5: {metrics.recall_at_5:.4f}")
        print(f"  • Recall@1: {metrics.recall_at_1:.4f}")
        print(f"  • Mean Reciprocal Rank: {metrics.mean_reciprocal_rank:.4f}")
        
        print(f"\n⏱️ Performance:")
        print(f"  • Avg query time: {metrics.avg_query_time_ms:.2f} ms")
        print(f"  • Index build time: {metrics.index_build_time_s:.2f} seconds")
        print(f"  • Total time: {metrics.total_eval_time_s:.2f} seconds")
        
        print(f"\n📈 Dataset:")
        print(f"  • Corpus size: {metrics.corpus_size:,}")
        print(f"  • Queries evaluated: {metrics.num_queries:,}")
        
        print(f"\n💾 Results saved to: {metrics_file}")
        
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
