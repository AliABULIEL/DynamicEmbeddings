"""Evaluation orchestration with cross-mode matrices and temperature sweep."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .encoder import encode_and_cache_bucket, load_embeddings
from .indexes import build_bucket_indexes, load_faiss_index, multi_index_search, query_index
from .metrics import evaluate_rankings_batch, bootstrap_confidence_interval, permutation_test
from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_ground_truth(
    query_ids: List[str],
    all_ids: List[str],
    same_period_only: bool = False,
) -> List[List[str]]:
    """Create ground truth relevant documents for queries.
    
    For this task, each query (paper) is considered relevant to itself.
    This is a simplified relevance assumption for demonstration.
    
    Args:
        query_ids: List of query document IDs.
        all_ids: List of all candidate document IDs.
        same_period_only: If True, only same paper ID is relevant.
        
    Returns:
        List of relevant ID lists for each query.
    """
    ground_truth = []
    for query_id in query_ids:
        # Simple: paper is relevant to itself
        relevant = [query_id]
        ground_truth.append(relevant)
    
    return ground_truth


def evaluate_bucket_pair(
    query_bucket: str,
    doc_bucket: str,
    embeddings_dir: Path,
    indexes_dir: Path,
    top_k: int = 100,
) -> Dict[str, float]:
    """Evaluate retrieval for a query-bucket × doc-bucket pair.
    
    Args:
        query_bucket: Bucket for queries.
        doc_bucket: Bucket for documents.
        embeddings_dir: Directory with cached embeddings.
        indexes_dir: Directory with FAISS indexes.
        top_k: Number of results to retrieve.
        
    Returns:
        Dictionary with metric scores.
    """
    # Load test embeddings from query bucket
    test_dir = embeddings_dir / query_bucket / "test"
    if not test_dir.exists():
        logger.warning(f"Test data not found for {query_bucket}")
        return {
            "ndcg@10": 0.0,
            "recall@10": 0.0,
            "recall@100": 0.0,
            "mrr": 0.0,
        }
    
    query_embeddings, query_ids = load_embeddings(test_dir)
    
    # Load index from doc bucket
    index_path = indexes_dir / f"{doc_bucket}.faiss"
    if not index_path.exists():
        logger.warning(f"Index not found: {index_path}")
        return {
            "ndcg@10": 0.0,
            "recall@10": 0.0,
            "recall@100": 0.0,
            "mrr": 0.0,
        }
    
    index = load_faiss_index(index_path)
    
    # Load index IDs
    ids_path = indexes_dir / f"{doc_bucket}_ids.txt"
    with open(ids_path, "r") as f:
        index_ids = [line.strip() for line in f]
    
    # Query
    scores, indices = query_index(index, query_embeddings, k=top_k)
    
    # Map to document IDs
    retrieved_ids = []
    for query_indices in indices:
        retrieved_ids.append([index_ids[idx] for idx in query_indices])
    
    # Create ground truth
    ground_truth = create_ground_truth(query_ids, index_ids)
    
    # Evaluate
    metrics = evaluate_rankings_batch(retrieved_ids, ground_truth)
    
    return metrics


def evaluate_cross_bucket_matrix(
    buckets: List[str],
    embeddings_dir: Path,
    indexes_dir: Path,
    top_k: int = 100,
) -> Dict[str, pd.DataFrame]:
    """Evaluate all query-bucket × doc-bucket combinations.
    
    Creates a matrix where rows are query buckets and columns are doc buckets.
    
    Args:
        buckets: List of bucket names.
        embeddings_dir: Directory with cached embeddings.
        indexes_dir: Directory with FAISS indexes.
        top_k: Number of results to retrieve.
        
    Returns:
        Dictionary mapping metric_name -> DataFrame (query_bucket × doc_bucket).
    """
    logger.info(f"Evaluating cross-bucket matrix for {len(buckets)} buckets")
    
    metrics_names = ["ndcg@10", "recall@10", "recall@100", "mrr"]
    results = {metric: {} for metric in metrics_names}
    
    for query_bucket in buckets:
        logger.info(f"Query bucket: {query_bucket}")
        
        for doc_bucket in buckets:
            logger.info(f"  → Doc bucket: {doc_bucket}")
            
            # Evaluate this pair
            metrics = evaluate_bucket_pair(
                query_bucket=query_bucket,
                doc_bucket=doc_bucket,
                embeddings_dir=embeddings_dir,
                indexes_dir=indexes_dir,
                top_k=top_k,
            )
            
            # Store each metric
            for metric_name in metrics_names:
                if query_bucket not in results[metric_name]:
                    results[metric_name][query_bucket] = {}
                results[metric_name][query_bucket][doc_bucket] = metrics[metric_name]
    
    # Convert to DataFrames
    dfs = {}
    for metric_name in metrics_names:
        df = pd.DataFrame(results[metric_name])
        df = df.T  # Transpose so query buckets are rows
        dfs[metric_name] = df
    
    return dfs


def evaluate_multi_index_with_temperature(
    buckets: List[str],
    embeddings_dir: Path,
    indexes_dir: Path,
    merge_strategy: str = "softmax",
    temperatures: List[float] = [1.5, 2.0, 3.0],
    top_k: int = 100,
) -> pd.DataFrame:
    """Evaluate multi-index retrieval with temperature sweep.
    
    Args:
        buckets: List of bucket names.
        embeddings_dir: Directory with cached embeddings.
        indexes_dir: Directory with FAISS indexes.
        merge_strategy: Merge strategy (softmax, mean, max, rrf).
        temperatures: List of temperatures to try (only for softmax).
        top_k: Number of results to retrieve.
        
    Returns:
        DataFrame with results for each temperature.
    """
    logger.info(f"Multi-index temperature sweep: {merge_strategy}, temps={temperatures}")
    
    # Load all indexes
    indexes = {}
    index_ids_map = {}
    
    for bucket_name in buckets:
        index_path = indexes_dir / f"{bucket_name}.faiss"
        if not index_path.exists():
            logger.warning(f"Index not found: {index_path}")
            continue
        
        indexes[bucket_name] = load_faiss_index(index_path)
        
        ids_path = indexes_dir / f"{bucket_name}_ids.txt"
        with open(ids_path, "r") as f:
            index_ids_map[bucket_name] = [line.strip() for line in f]
    
    # Collect all results
    all_results = []
    
    for temperature in temperatures:
        logger.info(f"Temperature: {temperature}")
        
        for bucket_name in buckets:
            test_dir = embeddings_dir / bucket_name / "test"
            if not test_dir.exists():
                continue
            
            query_embeddings, query_ids = load_embeddings(test_dir)
            
            # Multi-index search
            retrieved_ids, _ = multi_index_search(
                indexes,
                index_ids_map,
                query_embeddings,
                k=top_k,
                merge_strategy=merge_strategy,
                temperature=temperature,
            )
            
            # Create ground truth
            all_ids = []
            for bucket_ids in index_ids_map.values():
                all_ids.extend(bucket_ids)
            ground_truth = create_ground_truth(query_ids, all_ids)
            
            # Evaluate
            metrics = evaluate_rankings_batch(retrieved_ids, ground_truth)
            
            # Store result
            result = {
                "temperature": temperature,
                "query_bucket": bucket_name,
                "merge_strategy": merge_strategy,
                **metrics,
            }
            all_results.append(result)
    
    return pd.DataFrame(all_results)


def evaluate_mode(
    mode: str,
    buckets: List[str],
    embeddings_dir: Path,
    indexes_dir: Path,
    output_dir: Path,
    top_k: int = 100,
) -> Dict[str, pd.DataFrame]:
    """Evaluate a single training mode.
    
    Args:
        mode: Training mode (baseline_frozen, lora, full_ft, seq_ft).
        buckets: List of bucket names.
        embeddings_dir: Directory with embeddings for this mode.
        indexes_dir: Directory with indexes for this mode.
        output_dir: Output directory for results.
        top_k: Number of results to retrieve.
        
    Returns:
        Dictionary of metric DataFrames.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating mode: {mode}")
    logger.info(f"{'='*60}")
    
    # Check if embeddings exist
    if not embeddings_dir.exists():
        logger.warning(f"Embeddings directory not found: {embeddings_dir}")
        return {}
    
    # Check if indexes exist
    if not indexes_dir.exists():
        logger.warning(f"Indexes directory not found: {indexes_dir}")
        return {}
    
    # Evaluate cross-bucket matrix
    metric_dfs = evaluate_cross_bucket_matrix(
        buckets=buckets,
        embeddings_dir=embeddings_dir,
        indexes_dir=indexes_dir,
        top_k=top_k,
    )
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for metric_name, df in metric_dfs.items():
        csv_path = output_dir / f"{mode}_{metric_name.replace('@', '_at_')}.csv"
        df.to_csv(csv_path)
        logger.info(f"✓ Saved {metric_name} matrix: {csv_path}")
    
    return metric_dfs


def run_temperature_sweep(
    mode: str,
    buckets: List[str],
    embeddings_dir: Path,
    indexes_dir: Path,
    output_dir: Path,
    temperatures: List[float] = [1.5, 2.0, 3.0],
) -> pd.DataFrame:
    """Run temperature sweep for multi-index merge.
    
    Args:
        mode: Training mode.
        buckets: List of bucket names.
        embeddings_dir: Directory with embeddings.
        indexes_dir: Directory with indexes.
        output_dir: Output directory.
        temperatures: List of temperatures to try.
        
    Returns:
        DataFrame with sweep results.
    """
    logger.info(f"Running temperature sweep for {mode}")
    
    # Check if paths exist
    if not embeddings_dir.exists() or not indexes_dir.exists():
        logger.warning(f"Missing data for {mode}, skipping temperature sweep")
        return pd.DataFrame()
    
    # Sweep
    results_df = evaluate_multi_index_with_temperature(
        buckets=buckets,
        embeddings_dir=embeddings_dir,
        indexes_dir=indexes_dir,
        merge_strategy="softmax",
        temperatures=temperatures,
    )
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{mode}_temperature_sweep.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved temperature sweep: {csv_path}")
    
    # Find best temperature per metric
    if len(results_df) > 0:
        metrics = ["ndcg@10", "recall@10", "recall@100", "mrr"]
        best_temps = {}
        
        for metric in metrics:
            # Average across query buckets
            avg_by_temp = results_df.groupby("temperature")[metric].mean()
            best_temp = avg_by_temp.idxmax()
            best_score = avg_by_temp.max()
            best_temps[metric] = {
                "temperature": best_temp,
                "score": best_score,
            }
        
        # Print summary
        print(f"\nBest temperatures for {mode}:")
        for metric, info in best_temps.items():
            print(f"  {metric}: T={info['temperature']:.1f} → {info['score']:.4f}")
    
    return results_df


def compare_modes(
    baseline_results: Dict[str, pd.DataFrame],
    lora_results: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Compare baseline and LoRA results, compute deltas.
    
    Args:
        baseline_results: Baseline metric DataFrames.
        lora_results: LoRA metric DataFrames.
        output_dir: Output directory for delta matrices.
    """
    logger.info("Computing delta matrices (LoRA - Baseline)")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for metric_name in baseline_results.keys():
        if metric_name not in lora_results:
            continue
        
        baseline_df = baseline_results[metric_name]
        lora_df = lora_results[metric_name]
        
        # Compute delta
        delta_df = lora_df - baseline_df
        
        # Save
        csv_path = output_dir / f"delta_{metric_name.replace('@', '_at_')}.csv"
        delta_df.to_csv(csv_path)
        logger.info(f"✓ Saved delta matrix: {csv_path}")
        
        # Print summary stats
        print(f"\nDelta statistics for {metric_name}:")
        print(f"  Mean: {delta_df.values.mean():.4f}")
        print(f"  Median: {np.median(delta_df.values):.4f}")
        print(f"  Max: {delta_df.values.max():.4f}")
        print(f"  Min: {delta_df.values.min():.4f}")


def run_full_evaluation(
    data_dir: Path,
    cache_dir: Path,
    modes: List[str],
    buckets: List[str],
    output_dir: Path,
    run_temperature_sweep_flag: bool = True,
    temperatures: List[float] = [1.5, 2.0, 3.0],
) -> None:
    """Run full evaluation across all modes.
    
    Args:
        data_dir: Directory with processed data.
        cache_dir: Cache directory (contains embeddings and indexes).
        modes: List of training modes to evaluate.
        buckets: List of bucket names.
        output_dir: Output directory for results.
        run_temperature_sweep_flag: Whether to run temperature sweep.
        temperatures: List of temperatures for sweep.
    """
    logger.info("\n" + "="*80)
    logger.info("FULL EVALUATION ACROSS MODES")
    logger.info("="*80)
    
    all_mode_results = {}
    
    # Evaluate each mode
    for mode in modes:
        embeddings_dir = cache_dir / "embeddings" / mode
        indexes_dir = cache_dir / "indexes" / mode
        mode_output_dir = output_dir / mode
        
        # Evaluate cross-bucket matrices
        metric_dfs = evaluate_mode(
            mode=mode,
            buckets=buckets,
            embeddings_dir=embeddings_dir,
            indexes_dir=indexes_dir,
            output_dir=mode_output_dir,
        )
        
        all_mode_results[mode] = metric_dfs
        
        # Temperature sweep
        if run_temperature_sweep_flag and mode != "baseline_frozen":
            run_temperature_sweep(
                mode=mode,
                buckets=buckets,
                embeddings_dir=embeddings_dir,
                indexes_dir=indexes_dir,
                output_dir=mode_output_dir,
                temperatures=temperatures,
            )
    
    # Compare baseline vs LoRA
    if "baseline_frozen" in all_mode_results and "lora" in all_mode_results:
        compare_modes(
            baseline_results=all_mode_results["baseline_frozen"],
            lora_results=all_mode_results["lora"],
            output_dir=output_dir / "comparisons",
        )
    
    logger.info("\n" + "="*80)
    logger.info("✓ Full evaluation complete!")
    logger.info("="*80)


def load_evaluation_results(
    mode: str,
    results_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Load evaluation results for a mode.
    
    Args:
        mode: Training mode.
        results_dir: Results directory.
        
    Returns:
        Dictionary of metric DataFrames.
    """
    mode_dir = results_dir / mode
    
    if not mode_dir.exists():
        raise FileNotFoundError(f"Results not found: {mode_dir}")
    
    metrics = {}
    metric_names = ["ndcg_at_10", "recall_at_10", "recall_at_100", "mrr"]
    
    for metric_name in metric_names:
        csv_path = mode_dir / f"{mode}_{metric_name}.csv"
        if csv_path.exists():
            metrics[metric_name.replace("_at_", "@")] = pd.read_csv(csv_path, index_col=0)
    
    return metrics
