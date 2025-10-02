"""Evaluation orchestration with scenarios and modes."""

import json
from pathlib import Path
from typing import Dict, List, Optional

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


def evaluate_within_period(
    bucket_name: str,
    embeddings_dir: Path,
    indexes_dir: Path,
    top_k: int = 100,
) -> Dict[str, float]:
    """Evaluate retrieval within same time period.
    
    Args:
        bucket_name: Name of the bucket.
        embeddings_dir: Directory with cached embeddings.
        indexes_dir: Directory with FAISS indexes.
        top_k: Number of results to retrieve.
        
    Returns:
        Dictionary with metric scores.
    """
    logger.info(f"Evaluating within-period for bucket: {bucket_name}")
    
    # Load test embeddings
    test_dir = embeddings_dir / bucket_name / "test"
    if not test_dir.exists():
        logger.warning(f"Test data not found for {bucket_name}")
        return {}
    
    query_embeddings, query_ids = load_embeddings(test_dir)
    
    # Load index
    index_path = indexes_dir / f"{bucket_name}.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    
    index = load_faiss_index(index_path)
    
    # Load index IDs
    ids_path = indexes_dir / f"{bucket_name}_ids.txt"
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


def evaluate_cross_period(
    query_bucket: str,
    doc_bucket: str,
    embeddings_dir: Path,
    indexes_dir: Path,
    top_k: int = 100,
) -> Dict[str, float]:
    """Evaluate cross-period retrieval (query from one period, docs from another).
    
    Args:
        query_bucket: Bucket for queries.
        doc_bucket: Bucket for documents.
        embeddings_dir: Directory with cached embeddings.
        indexes_dir: Directory with FAISS indexes.
        top_k: Number of results to retrieve.
        
    Returns:
        Dictionary with metric scores.
    """
    logger.info(f"Evaluating cross-period: {query_bucket} -> {doc_bucket}")
    
    # Load test embeddings from query bucket
    test_dir = embeddings_dir / query_bucket / "test"
    if not test_dir.exists():
        logger.warning(f"Test data not found for {query_bucket}")
        return {}
    
    query_embeddings, query_ids = load_embeddings(test_dir)
    
    # Load index from doc bucket
    index_path = indexes_dir / f"{doc_bucket}.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    
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
    
    # Create ground truth (query paper should be found if it exists in doc bucket)
    ground_truth = create_ground_truth(query_ids, index_ids)
    
    # Evaluate
    metrics = evaluate_rankings_batch(retrieved_ids, ground_truth)
    
    return metrics


def evaluate_multi_index(
    buckets: List[str],
    embeddings_dir: Path,
    indexes_dir: Path,
    merge_strategy: str = "softmax",
    temperature: float = 2.0,
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Evaluate multi-index retrieval with merge strategy.
    
    Args:
        buckets: List of bucket names.
        embeddings_dir: Directory with cached embeddings.
        indexes_dir: Directory with FAISS indexes.
        merge_strategy: How to merge (softmax, mean, max, rrf).
        temperature: Temperature for softmax.
        top_k: Number of results to retrieve.
        
    Returns:
        Dictionary mapping bucket -> metrics.
    """
    logger.info(f"Evaluating multi-index with merge={merge_strategy}, temp={temperature}")
    
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
    
    # Evaluate for each bucket's test set
    results = {}
    
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
        results[bucket_name] = metrics
    
    return results


def run_evaluation(
    data_dir: Path,
    embeddings_dir: Path,
    indexes_dir: Path,
    buckets: List[str],
    scenarios: List[str],
    mode: str = "multi-index",
    merge_strategy: str = "softmax",
    temperature: float = 2.0,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Run full evaluation across scenarios.
    
    Args:
        data_dir: Directory with processed data.
        embeddings_dir: Directory with cached embeddings.
        indexes_dir: Directory with FAISS indexes.
        buckets: List of bucket names.
        scenarios: List of scenarios (within, cross, all).
        mode: Retrieval mode (time-select or multi-index).
        merge_strategy: Merge strategy for multi-index.
        temperature: Temperature for softmax.
        output_dir: Output directory for results CSV.
        
    Returns:
        Dictionary with all results.
    """
    all_results = {}
    
    for scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Scenario: {scenario}")
        logger.info(f"{'='*60}")
        
        if scenario == "within":
            # Evaluate within each period
            for bucket in buckets:
                metrics = evaluate_within_period(bucket, embeddings_dir, indexes_dir)
                all_results[f"{scenario}_{bucket}"] = metrics
        
        elif scenario == "cross":
            # Evaluate cross-period
            for query_bucket in buckets:
                for doc_bucket in buckets:
                    if query_bucket != doc_bucket:
                        metrics = evaluate_cross_period(
                            query_bucket, doc_bucket, embeddings_dir, indexes_dir
                        )
                        all_results[f"{scenario}_{query_bucket}_to_{doc_bucket}"] = metrics
        
        elif scenario == "all":
            # Evaluate multi-index or time-select
            if mode == "multi-index":
                bucket_metrics = evaluate_multi_index(
                    buckets, embeddings_dir, indexes_dir, merge_strategy, temperature
                )
                for bucket, metrics in bucket_metrics.items():
                    all_results[f"{scenario}_{mode}_{bucket}"] = metrics
            else:
                # Time-select: just within-period
                for bucket in buckets:
                    metrics = evaluate_within_period(bucket, embeddings_dir, indexes_dir)
                    all_results[f"{scenario}_{mode}_{bucket}"] = metrics
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to: {results_path}")
        
        # Convert to CSV
        df = pd.DataFrame(all_results).T
        csv_path = output_dir / "evaluation_results.csv"
        df.to_csv(csv_path)
        logger.info(f"CSV saved to: {csv_path}")
    
    return all_results
