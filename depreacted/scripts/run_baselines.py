#!/usr/bin/env python
"""Script to run baseline model evaluations.

Usage:
    python scripts/run_baselines.py --output-dir results/baselines/
"""

import sys
import os
from pathlib import Path
import argparse
import json
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.tide_lite.models.baselines import BaselineEncoder
from src.tide_lite.eval.eval_stsb import STSBEvaluator
from src.tide_lite.eval.retrieval_quora import QuoraRetrievalEvaluator


BASELINE_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/e5-base-v2",
    "BAAI/bge-base-en-v1.5",
]


def evaluate_baseline(model_name: str, device: torch.device, max_samples: int = None):
    """Evaluate a single baseline model."""
    print(f"\nEvaluating: {model_name}")
    print("-" * 60)
    
    try:
        # Initialize model
        model = BaselineEncoder(model_name)
        model = model.to(device)
        model.eval()
        
        results = {
            "model": model_name,
            "device": str(device),
        }
        
        # STS-B evaluation
        print("  • Evaluating on STS-B...")
        stsb_evaluator = STSBEvaluator(model, device=device)
        stsb_metrics = stsb_evaluator.evaluate(max_samples=max_samples)
        
        results["stsb_spearman"] = stsb_metrics.spearman
        results["stsb_pearson"] = stsb_metrics.pearson
        print(f"    Spearman: {stsb_metrics.spearman:.4f}")
        
        # Quora evaluation (smaller subset for speed)
        if max_samples is None or max_samples > 100:
            print("  • Evaluating on Quora retrieval...")
            quora_evaluator = QuoraRetrievalEvaluator(model, device=device)
            
            corpus_size = min(5000, max_samples) if max_samples else 5000
            query_size = min(500, corpus_size // 10)
            
            quora_metrics = quora_evaluator.evaluate(
                max_corpus_size=corpus_size,
                max_queries=query_size
            )
            
            results["quora_ndcg_10"] = quora_metrics.ndcg_at_10
            results["quora_recall_10"] = quora_metrics.recall_at_10
            print(f"    nDCG@10: {quora_metrics.ndcg_at_10:.4f}")
        
        return results
        
    except Exception as e:
        print(f"  ⚠️ Failed to evaluate {model_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models")
    parser.add_argument("--output-dir", type=str, default="results/baselines",
                       help="Output directory for results")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Specific models to evaluate")
    parser.add_argument("--max-samples", type=int, 
                       help="Maximum samples for quick testing")
    parser.add_argument("--device", type=str,
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Select models
    models_to_eval = args.models if args.models else BASELINE_MODELS
    
    # Skip large models on CPU
    if device.type == "cpu" and args.max_samples is None:
        print("⚠️ Running on CPU - limiting to smaller models")
        models_to_eval = [m for m in models_to_eval if "base" not in m or "MiniLM" in m]
    
    # Evaluate each baseline
    all_results = []
    for model_name in models_to_eval:
        result = evaluate_baseline(model_name, device, args.max_samples)
        if result:
            all_results.append(result)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_file = output_dir / "metrics_all.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save as CSV
    csv_file = output_dir / "metrics_all.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(csv_file, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION SUMMARY")
    print("=" * 60)
    
    for result in all_results:
        model = result["model"].split("/")[-1]
        stsb = result.get("stsb_spearman", 0)
        quora = result.get("quora_ndcg_10", 0)
        print(f"{model:30} | STS-B: {stsb:.4f} | Quora: {quora:.4f}")
    
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  • {json_file}")
    print(f"  • {csv_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
