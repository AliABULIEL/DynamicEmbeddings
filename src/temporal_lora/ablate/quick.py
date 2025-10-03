"""Quick ablation study for LoRA hyperparameters."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer

from ..train.trainer import train_single_bucket
from ..eval.encoder import encode_and_cache_bucket
from ..eval.indexes import build_faiss_index, query_index
from ..eval.metrics import evaluate_rankings_batch
from ..utils.logging import get_logger

logger = get_logger(__name__)


def run_ablation_experiment(
    bucket_name: str,
    data_dir: Path,
    base_model_name: str,
    lora_r: int,
    lora_target_modules: List[str],
    epochs: int = 1,
    max_eval_samples: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """Run single ablation experiment.
    
    Args:
        bucket_name: Name of the time bucket.
        data_dir: Directory with processed data.
        base_model_name: Base model identifier.
        lora_r: LoRA rank.
        lora_target_modules: Target modules for LoRA.
        epochs: Training epochs.
        max_eval_samples: Maximum samples for evaluation.
        seed: Random seed.
        
    Returns:
        Dictionary with results.
    """
    import tempfile
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        adapter_dir = tmpdir / "adapter"
        embeddings_dir = tmpdir / "embeddings"
        
        # Training config
        config = {
            "model": {
                "base_model": {"name": base_model_name},
                "lora": {
                    "r": lora_r,
                    "lora_alpha": lora_r * 2,
                    "lora_dropout": 0.1,
                    "bias": "none",
                    "target_modules": lora_target_modules,
                },
            },
            "training": {
                "epochs": epochs,
                "batch_size": 16,
                "learning_rate": 2e-4,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "fp16": False,  # For quick ablation
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0,
                "seed": seed,
            },
            "negatives": {
                "cross_period_negatives": False,
            },
        }
        
        # Train
        logger.info(f"Training with r={lora_r}, modules={lora_target_modules}")
        start_time = time.time()
        
        try:
            metrics = train_single_bucket(
                bucket_name=bucket_name,
                bucket_data_path=data_dir / bucket_name,
                output_dir=adapter_dir,
                config=config,
                mode="lora",
                use_hard_negatives=False,
            )
            train_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "rank": lora_r,
                "target_modules": "+".join(lora_target_modules),
                "status": "failed",
                "error": str(e),
            }
        
        # Encode test set
        logger.info("Encoding test set...")
        encode_and_cache_bucket(
            bucket_name=bucket_name,
            bucket_data_path=data_dir / bucket_name,
            adapter_dir=adapter_dir,
            base_model_name=base_model_name,
            output_dir=embeddings_dir,
            use_lora=True,
        )
        
        # Build index and evaluate
        from ..eval.encoder import load_embeddings
        
        test_dir = embeddings_dir / bucket_name / "test"
        embeddings, doc_ids = load_embeddings(test_dir)
        
        # Limit evaluation samples
        if len(embeddings) > max_eval_samples:
            embeddings = embeddings[:max_eval_samples]
            doc_ids = doc_ids[:max_eval_samples]
        
        # Build index
        index = build_faiss_index(embeddings)
        
        # Query (self-retrieval)
        scores, indices = query_index(index, embeddings, k=10)
        
        # Evaluate
        retrieved_ids = [[doc_ids[idx] for idx in query_indices] for query_indices in indices]
        ground_truth = [[doc_id] for doc_id in doc_ids]
        
        eval_metrics = evaluate_rankings_batch(retrieved_ids, ground_truth)
        
        # Compile results
        results = {
            "rank": lora_r,
            "target_modules": "+".join(lora_target_modules),
            "trainable_params": metrics.get("trainable_params", 0),
            "trainable_percent": metrics.get("trainable_percent", 0.0),
            "train_time_seconds": train_time,
            "ndcg@10": eval_metrics["ndcg@10"],
            "recall@10": eval_metrics["recall@10"],
            "recall@100": eval_metrics["recall@100"],
            "mrr": eval_metrics["mrr"],
            "status": "success",
        }
        
        return results


def run_quick_ablation(
    data_dir: Path,
    base_model_name: str,
    bucket_name: str,
    output_path: Path,
    ranks: List[int] = [8, 16, 32],
    target_module_sets: Optional[List[List[str]]] = None,
    epochs: int = 1,
    max_eval_samples: int = 500,
) -> pd.DataFrame:
    """Run quick ablation study.
    
    Args:
        data_dir: Directory with processed data.
        base_model_name: Base model identifier.
        bucket_name: Bucket to use for ablation.
        output_path: Path to save results CSV.
        ranks: LoRA ranks to test.
        target_module_sets: List of target module sets to test.
        epochs: Training epochs.
        max_eval_samples: Max samples for evaluation.
        
    Returns:
        DataFrame with ablation results.
    """
    logger.info("\n" + "="*80)
    logger.info("QUICK ABLATION STUDY")
    logger.info("="*80)
    logger.info(f"Bucket: {bucket_name}")
    logger.info(f"Ranks: {ranks}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Eval samples: {max_eval_samples}")
    
    # Default target module sets
    if target_module_sets is None:
        target_module_sets = [
            ["q_proj", "k_proj", "v_proj"],  # QKV
            ["q_proj", "k_proj", "v_proj", "out_proj"],  # QKVO
        ]
    
    logger.info(f"Target module sets: {len(target_module_sets)}")
    
    # Run experiments
    all_results = []
    
    for rank in ranks:
        for target_modules in target_module_sets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment: r={rank}, modules={target_modules}")
            logger.info(f"{'='*60}")
            
            results = run_ablation_experiment(
                bucket_name=bucket_name,
                data_dir=data_dir,
                base_model_name=base_model_name,
                lora_r=rank,
                lora_target_modules=target_modules,
                epochs=epochs,
                max_eval_samples=max_eval_samples,
            )
            
            all_results.append(results)
            
            # Print intermediate result
            if results["status"] == "success":
                logger.info(f"✓ NDCG@10: {results['ndcg@10']:.4f}")
                logger.info(f"✓ Trainable: {results['trainable_percent']:.2f}%")
                logger.info(f"✓ Time: {results['train_time_seconds']:.1f}s")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Ablation results saved to: {output_path}")
    
    # Create markdown summary
    create_ablation_summary(df, output_path.parent / "ablation_summary.md")
    
    # Print summary table
    print("\n" + "="*80)
    print("ABLATION SUMMARY")
    print("="*80)
    
    if len(df) > 0 and "status" in df.columns:
        success_df = df[df["status"] == "success"]
        
        if len(success_df) > 0:
            display_cols = ["rank", "target_modules", "trainable_percent", "ndcg@10", "train_time_seconds"]
            print(success_df[display_cols].to_string(index=False))
            
            # Best configuration
            best_idx = success_df["ndcg@10"].idxmax()
            best_config = success_df.loc[best_idx]
            
            print("\n" + "="*80)
            print("BEST CONFIGURATION")
            print("="*80)
            print(f"Rank: {best_config['rank']}")
            print(f"Target Modules: {best_config['target_modules']}")
            print(f"NDCG@10: {best_config['ndcg@10']:.4f}")
            print(f"Trainable %: {best_config['trainable_percent']:.2f}%")
            print(f"Train Time: {best_config['train_time_seconds']:.1f}s")
            print("="*80)
    
    return df


def create_ablation_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Create markdown summary of ablation results.
    
    Args:
        df: DataFrame with ablation results.
        output_path: Path to save markdown file.
    """
    if len(df) == 0:
        return
    
    success_df = df[df["status"] == "success"] if "status" in df.columns else df
    
    if len(success_df) == 0:
        logger.warning("No successful ablation experiments to summarize")
        return
    
    # Generate markdown
    md = "# Quick Ablation Study Results\n\n"
    
    md += "## Overview\n\n"
    md += f"Total experiments: {len(df)}\n"
    md += f"Successful: {len(success_df)}\n\n"
    
    md += "## Results Table\n\n"
    md += "| Rank | Target Modules | Trainable % | NDCG@10 | Recall@10 | MRR | Train Time (s) |\n"
    md += "|------|---------------|-------------|---------|-----------|-----|----------------|\n"
    
    for _, row in success_df.iterrows():
        md += f"| {row['rank']} "
        md += f"| {row['target_modules']} "
        md += f"| {row['trainable_percent']:.2f}% "
        md += f"| {row['ndcg@10']:.4f} "
        md += f"| {row['recall@10']:.4f} "
        md += f"| {row['mrr']:.4f} "
        md += f"| {row['train_time_seconds']:.1f} |\n"
    
    # Best configuration
    best_idx = success_df["ndcg@10"].idxmax()
    best_config = success_df.loc[best_idx]
    
    md += "\n## Best Configuration\n\n"
    md += f"- **Rank**: {best_config['rank']}\n"
    md += f"- **Target Modules**: {best_config['target_modules']}\n"
    md += f"- **NDCG@10**: {best_config['ndcg@10']:.4f}\n"
    md += f"- **Trainable Parameters**: {best_config['trainable_percent']:.2f}%\n"
    md += f"- **Training Time**: {best_config['train_time_seconds']:.1f} seconds\n\n"
    
    # Insights
    md += "## Key Insights\n\n"
    
    # Rank comparison
    rank_comparison = success_df.groupby("rank")["ndcg@10"].mean().to_dict()
    md += "### Rank Comparison (Average NDCG@10)\n\n"
    for rank, score in sorted(rank_comparison.items()):
        md += f"- **r={rank}**: {score:.4f}\n"
    
    md += "\n"
    
    # Module comparison
    module_comparison = success_df.groupby("target_modules")["ndcg@10"].mean().to_dict()
    md += "### Target Module Comparison (Average NDCG@10)\n\n"
    for modules, score in sorted(module_comparison.items(), key=lambda x: x[1], reverse=True):
        md += f"- **{modules}**: {score:.4f}\n"
    
    # Save
    with open(output_path, "w") as f:
        f.write(md)
    
    logger.info(f"✓ Ablation summary saved to: {output_path}")
