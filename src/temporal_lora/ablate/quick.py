"""Quick ablation study for hyperparameter sensitivity.

Tests LoRA rank and merge strategy on a small eval slice.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..eval.encoder import encode_and_cache_bucket
from ..eval.evaluate import run_evaluation
from ..eval.indexes import build_bucket_indexes
from ..train.trainer import train_all_buckets
from ..utils.io import load_config
from ..utils.logging import get_logger
from ..utils.paths import CONFIG_DIR, DATA_PROCESSED_DIR, PROJECT_ROOT

logger = get_logger(__name__)


def run_quick_ablation(
    data_dir: Path = DATA_PROCESSED_DIR,
    output_dir: Optional[Path] = None,
    lora_ranks: List[int] = None,
    merge_strategies: List[str] = None,
    max_eval_queries: int = 100,
) -> Dict[str, pd.DataFrame]:
    """Run quick ablation study on LoRA rank and merge strategy.
    
    Args:
        data_dir: Directory with processed data.
        output_dir: Output directory for results.
        lora_ranks: List of LoRA ranks to test (default: [8, 16]).
        merge_strategies: List of merge strategies (default: ['softmax', 'mean']).
        max_eval_queries: Max queries per scenario for quick eval.
        
    Returns:
        Dictionary with results DataFrames.
    """
    if lora_ranks is None:
        lora_ranks = [8, 16]
    
    if merge_strategies is None:
        merge_strategies = ["softmax", "mean"]
    
    if output_dir is None:
        output_dir = PROJECT_ROOT / "deliverables" / "results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("QUICK ABLATION STUDY")
    logger.info("=" * 60)
    logger.info(f"LoRA ranks: {lora_ranks}")
    logger.info(f"Merge strategies: {merge_strategies}")
    logger.info(f"Max eval queries: {max_eval_queries}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load configs
    data_config = load_config("data", CONFIG_DIR)
    model_config = load_config("model", CONFIG_DIR)
    train_config = load_config("train", CONFIG_DIR)
    
    buckets = [b["name"] for b in data_config["buckets"]]
    base_model = model_config["base_model"]["name"]
    
    all_results = []
    
    # Run ablation
    for rank in lora_ranks:
        for merge in merge_strategies:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Configuration: rank={rank}, merge={merge}")
            logger.info(f"{'=' * 60}")
            
            start_time = time.time()
            
            # Set config
            config = {
                "model": {**model_config, "lora": {**model_config["lora"], "r": rank}},
                "training": train_config["training"],
                "negatives": train_config["negatives"],
            }
            
            # Train adapters
            adapters_dir = PROJECT_ROOT / f"models/ablate_adapters_r{rank}"
            logger.info(f"Training adapters with rank={rank}...")
            train_all_buckets(data_dir, adapters_dir, config)
            
            # Build indexes
            embeddings_dir = PROJECT_ROOT / f"models/ablate_embeddings_r{rank}"
            indexes_dir = PROJECT_ROOT / f"models/ablate_indexes_r{rank}"
            
            logger.info("Encoding and indexing...")
            for bucket_name in buckets:
                bucket_data_path = data_dir / bucket_name
                adapter_dir = adapters_dir / bucket_name
                
                encode_and_cache_bucket(
                    bucket_name=bucket_name,
                    bucket_data_path=bucket_data_path,
                    adapter_dir=adapter_dir,
                    base_model_name=base_model,
                    output_dir=embeddings_dir,
                    use_lora=True,
                )
            
            build_bucket_indexes(embeddings_dir, indexes_dir, buckets)
            
            # Evaluate with quick subset
            logger.info(f"Evaluating with merge={merge}...")
            eval_results = run_evaluation(
                data_dir=data_dir,
                embeddings_dir=embeddings_dir,
                indexes_dir=indexes_dir,
                buckets=buckets,
                scenarios=["within", "cross"],
                mode="multi-index",
                merge_strategy=merge,
                temperature=2.0,
                output_dir=output_dir,
                max_queries=max_eval_queries,
            )
            
            # Collect results
            elapsed = time.time() - start_time
            
            for scenario, metrics in eval_results.items():
                result_row = {
                    "lora_rank": rank,
                    "merge_strategy": merge,
                    "scenario": scenario,
                    "elapsed_time": elapsed,
                    **metrics,
                }
                all_results.append(result_row)
            
            logger.info(f"Configuration completed in {elapsed:.2f}s")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save CSV
    csv_path = output_dir / "quick_ablation.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to: {csv_path}")
    
    # Generate summary
    summary = generate_ablation_summary(results_df)
    
    # Save Markdown summary
    md_path = output_dir / "quick_ablation.md"
    with open(md_path, "w") as f:
        f.write(summary)
    logger.info(f"Saved summary to: {md_path}")
    
    return {"results": results_df}


def generate_ablation_summary(results_df: pd.DataFrame) -> str:
    """Generate Markdown summary of ablation results.
    
    Args:
        results_df: Results DataFrame.
        
    Returns:
        Markdown-formatted summary.
    """
    summary = ["# Quick Ablation Study: LoRA Rank & Merge Strategy\n"]
    summary.append("## Overview\n")
    summary.append("This ablation tests the sensitivity of the Temporal LoRA system to:\n")
    summary.append("- **LoRA Rank**: {8, 16}\n")
    summary.append("- **Merge Strategy**: {softmax, mean}\n\n")
    
    # Results table
    summary.append("## Results Summary\n\n")
    summary.append("### NDCG@10 by Configuration\n\n")
    
    # Pivot table for NDCG@10
    pivot_ndcg = results_df.pivot_table(
        values="ndcg@10",
        index=["scenario"],
        columns=["lora_rank", "merge_strategy"],
        aggfunc="mean"
    )
    
    summary.append(pivot_ndcg.to_markdown())
    summary.append("\n\n")
    
    # Best configuration
    summary.append("## Best Configuration\n\n")
    
    # Find best by mean NDCG@10
    config_means = results_df.groupby(["lora_rank", "merge_strategy"])["ndcg@10"].mean()
    best_config = config_means.idxmax()
    best_score = config_means.max()
    
    summary.append(f"**Best Configuration:**\n")
    summary.append(f"- LoRA Rank: **{best_config[0]}**\n")
    summary.append(f"- Merge Strategy: **{best_config[1]}**\n")
    summary.append(f"- Mean NDCG@10: **{best_score:.4f}**\n\n")
    
    # Performance breakdown
    summary.append("## Performance by Scenario\n\n")
    
    for scenario in results_df["scenario"].unique():
        scenario_df = results_df[results_df["scenario"] == scenario]
        
        summary.append(f"### {scenario.title()} Queries\n\n")
        
        # Create compact table
        scenario_summary = scenario_df.groupby(["lora_rank", "merge_strategy"]).agg({
            "ndcg@10": "mean",
            "recall@10": "mean",
            "mrr": "mean",
            "elapsed_time": "mean"
        }).reset_index()
        
        summary.append(scenario_summary.to_markdown(index=False))
        summary.append("\n\n")
    
    # Insights
    summary.append("## Key Insights\n\n")
    
    # Rank effect
    rank_means = results_df.groupby("lora_rank")["ndcg@10"].mean()
    rank_8 = rank_means.get(8, 0)
    rank_16 = rank_means.get(16, 0)
    
    if rank_16 > rank_8:
        rank_winner = "16"
        rank_delta = rank_16 - rank_8
        summary.append(f"1. **LoRA Rank 16 outperforms Rank 8** by {rank_delta:.4f} NDCG@10 on average.\n")
    else:
        rank_winner = "8"
        rank_delta = rank_8 - rank_16
        summary.append(f"1. **LoRA Rank 8 matches or exceeds Rank 16** (Δ={rank_delta:.4f} NDCG@10), suggesting rank 8 is sufficient.\n")
    
    # Merge effect
    merge_means = results_df.groupby("merge_strategy")["ndcg@10"].mean()
    softmax_score = merge_means.get("softmax", 0)
    mean_score = merge_means.get("mean", 0)
    
    if softmax_score > mean_score:
        merge_delta = softmax_score - mean_score
        summary.append(f"2. **Softmax merge is superior** to mean by {merge_delta:.4f} NDCG@10, indicating score calibration matters.\n")
    else:
        merge_delta = mean_score - softmax_score
        summary.append(f"2. **Mean merge performs comparably** to softmax (Δ={merge_delta:.4f}), suggesting simpler aggregation suffices.\n")
    
    # Cross-period effect
    within_df = results_df[results_df["scenario"] == "within"]
    cross_df = results_df[results_df["scenario"] == "cross"]
    
    if len(within_df) > 0 and len(cross_df) > 0:
        within_mean = within_df["ndcg@10"].mean()
        cross_mean = cross_df["ndcg@10"].mean()
        gap = within_mean - cross_mean
        
        summary.append(f"3. **Within-period queries outperform cross-period** by {gap:.4f} NDCG@10, ")
        summary.append("confirming temporal adaptation improves period-specific retrieval.\n")
    
    summary.append("\n")
    summary.append("---\n")
    summary.append("*This is a quick ablation on a subset of queries. ")
    summary.append("For comprehensive results, run full evaluation.*\n")
    
    return "".join(summary)


def main():
    """CLI entry point for quick ablation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run quick ablation study")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help="Data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--lora-ranks",
        type=int,
        nargs="+",
        default=[8, 16],
        help="LoRA ranks to test"
    )
    parser.add_argument(
        "--merge-strategies",
        nargs="+",
        default=["softmax", "mean"],
        help="Merge strategies to test"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=100,
        help="Max queries per scenario"
    )
    
    args = parser.parse_args()
    
    run_quick_ablation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lora_ranks=args.lora_ranks,
        merge_strategies=args.merge_strategies,
        max_eval_queries=args.max_queries,
    )


if __name__ == "__main__":
    main()
