#!/usr/bin/env python
"""Evaluation script for TIDE-Lite and baseline models.

Usage:
    python scripts/evaluate.py --model tide-lite --checkpoint path/to/model
    python scripts/evaluate.py --model baseline --encoder all-MiniLM-L6-v2
    python scripts/evaluate.py --task stsb --output-dir results/
"""

import sys
import os
from pathlib import Path
import argparse
import json
import logging
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.tide_lite.models.tide_lite import TIDELite
from src.tide_lite.models.baselines import BaselineEncoder
from src.tide_lite.eval.eval_stsb import STSBEvaluator
from src.tide_lite.eval.retrieval_quora import QuoraRetrievalEvaluator

logger = logging.getLogger(__name__)


def check_faiss_availability():
    """Check if FAISS is available and which version."""
    try:
        import faiss
        try:
            # Check if GPU version is available
            if hasattr(faiss, 'StandardGpuResources'):
                logger.info("FAISS-GPU detected")
                return "faiss-gpu"
        except:
            pass
        logger.info("FAISS-CPU detected")
        return "faiss-cpu"
    except ImportError:
        logger.error("FAISS not found. Please install faiss-cpu: pip install faiss-cpu")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate TIDE-Lite models")
    
    # Model selection
    parser.add_argument("--model", choices=["tide-lite", "baseline"], 
                       default="tide-lite", help="Model type to evaluate")
    parser.add_argument("--checkpoint", type=str, help="Path to TIDE-Lite checkpoint")
    parser.add_argument("--encoder", type=str, 
                       default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Encoder name for baseline models")
    
    # Task selection
    parser.add_argument("--task", choices=["stsb", "quora", "all"], 
                       default="all", help="Evaluation task")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results/",
                       help="Output directory for metrics")
    
    # Options
    parser.add_argument("--max-samples", type=int, help="Limit samples for quick testing")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Check FAISS
    faiss_version = check_faiss_availability()
    if faiss_version is None and args.task in ["quora", "all"]:
        logger.error("FAISS required for retrieval evaluation")
        return 1
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    if args.model == "tide-lite":
        if not args.checkpoint:
            logger.error("--checkpoint required for TIDE-Lite evaluation")
            return 1
        model = TIDELite.from_pretrained(args.checkpoint)
    else:
        model = BaselineEncoder(args.encoder)
    
    model = model.to(device)
    model.eval()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = {}
    
    # Evaluate on STS-B
    if args.task in ["stsb", "all"]:
        logger.info("Evaluating on STS-B...")
        stsb_evaluator = STSBEvaluator(model, device=device)
        stsb_metrics = stsb_evaluator.evaluate(max_samples=args.max_samples)
        
        # Save STS-B metrics
        model_name = args.model if args.model == "tide-lite" else args.encoder.split("/")[-1]
        stsb_file = stsb_evaluator.save_results(stsb_metrics, output_dir, model_name)
        
        all_metrics["stsb"] = {
            "spearman": stsb_metrics.spearman,
            "pearson": stsb_metrics.pearson,
        }
        logger.info(f"STS-B Spearman: {stsb_metrics.spearman:.4f}")
    
    # Evaluate on Quora
    if args.task in ["quora", "all"]:
        logger.info("Evaluating on Quora retrieval...")
        quora_evaluator = QuoraRetrievalEvaluator(model, device=device)
        
        # Use smaller corpus for quick testing
        max_corpus = args.max_samples if args.max_samples else 10000
        max_queries = min(1000, max_corpus // 10) if args.max_samples else 1000
        
        quora_metrics = quora_evaluator.evaluate(
            max_corpus_size=max_corpus,
            max_queries=max_queries
        )
        
        # Save Quora metrics
        model_name = args.model if args.model == "tide-lite" else args.encoder.split("/")[-1]
        quora_file = quora_evaluator.save_results(quora_metrics, output_dir, model_name)
        
        all_metrics["quora"] = {
            "ndcg_at_10": quora_metrics.ndcg_at_10,
            "recall_at_10": quora_metrics.recall_at_10,
            "mrr": quora_metrics.mean_reciprocal_rank,
        }
        logger.info(f"Quora nDCG@10: {quora_metrics.ndcg_at_10:.4f}")
    
    # Save combined metrics
    metrics_file = output_dir / "metrics_all.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"All metrics saved to {metrics_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for task, metrics in all_metrics.items():
        print(f"\n{task.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
