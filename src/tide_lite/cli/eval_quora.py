"""Command-line interface for Quora retrieval evaluation.

This module provides the CLI for evaluating models on the Quora duplicate
questions retrieval task using FAISS for efficient similarity search.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..eval.retrieval_quora import evaluate_quora_retrieval

logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup argument parser for Quora retrieval evaluation.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="eval-quora",
        description="Evaluate models on Quora duplicate questions retrieval task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (default) - show evaluation plan
  python -m tide_lite.cli.eval_quora --model path/to/model
  
  # Actually run evaluation
  python -m tide_lite.cli.eval_quora --model path/to/model --run
  
  # Evaluate baseline model
  python -m tide_lite.cli.eval_quora --model minilm --type baseline --run
  
  # Use IVF index for faster search
  python -m tide_lite.cli.eval_quora --model model.pt --index-type IVF --run
  
  # Limit corpus/queries for quick testing
  python -m tide_lite.cli.eval_quora --model model.pt --max-corpus 1000 --max-queries 100 --run
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
    
    # FAISS configuration (must be same across all models)
    parser.add_argument(
        "--index-type",
        type=str,
        default="Flat",
        choices=["Flat", "IVF"],
        help="FAISS index type (default: Flat for exact search)",
    )
    
    parser.add_argument(
        "--faiss-nlist",
        type=int,
        default=100,
        help="Number of clusters for IVF index (default: 100)",
    )
    
    parser.add_argument(
        "--faiss-nprobe",
        type=int,
        default=10,
        help="Number of probes for IVF search (default: 10)",
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for FAISS if available",
    )
    
    # Data configuration
    parser.add_argument(
        "--max-corpus",
        type=int,
        default=10000,
        help="Maximum corpus size (default: 10000)",
    )
    
    parser.add_argument(
        "--max-queries",
        type=int,
        default=1000,
        help="Maximum number of queries (default: 1000)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for encoding (default: 128)",
    )
    
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum sequence length - must be same for all models (default: 128)",
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


def main() -> int:
    """Main entry point for Quora retrieval evaluation.
    
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
        results = evaluate_quora_retrieval(
            model_id_or_path=args.model,
            model_type=args.type,
            max_corpus_size=args.max_corpus,
            max_queries=args.max_queries,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            faiss_index_type=args.index_type,
            output_dir=args.output_dir if not args.no_save else None,
            dry_run=not args.run,
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
