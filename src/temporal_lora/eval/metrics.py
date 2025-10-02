"""Evaluation metrics for retrieval tasks."""

from typing import Dict, List

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


def ndcg_at_k(relevance: np.ndarray, k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain @ k.
    
    Args:
        relevance: Binary relevance array (n_results,). 1 = relevant, 0 = not.
        k: Cutoff position.
        
    Returns:
        NDCG@k score.
    """
    if len(relevance) == 0:
        return 0.0
    
    # Actual DCG
    relevance_k = relevance[:k]
    discounts = np.log2(np.arange(2, len(relevance_k) + 2))
    dcg = np.sum(relevance_k / discounts)
    
    # Ideal DCG
    ideal_relevance = np.sort(relevance)[::-1][:k]
    ideal_discounts = np.log2(np.arange(2, len(ideal_relevance) + 2))
    idcg = np.sum(ideal_relevance / ideal_discounts)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def recall_at_k(relevance: np.ndarray, k: int = 10) -> float:
    """Compute Recall @ k.
    
    Args:
        relevance: Binary relevance array (n_results,).
        k: Cutoff position.
        
    Returns:
        Recall@k score.
    """
    if len(relevance) == 0 or np.sum(relevance) == 0:
        return 0.0
    
    relevance_k = relevance[:k]
    return np.sum(relevance_k) / np.sum(relevance)


def mean_reciprocal_rank(relevance: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank.
    
    Args:
        relevance: Binary relevance array (n_results,).
        
    Returns:
        MRR score.
    """
    if len(relevance) == 0:
        return 0.0
    
    # Find first relevant position
    relevant_positions = np.where(relevance > 0)[0]
    if len(relevant_positions) == 0:
        return 0.0
    
    first_relevant = relevant_positions[0] + 1  # 1-indexed
    return 1.0 / first_relevant


def compute_relevance(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> np.ndarray:
    """Compute binary relevance for retrieved documents.
    
    Args:
        retrieved_ids: List of retrieved document IDs.
        relevant_ids: List of ground-truth relevant document IDs.
        
    Returns:
        Binary relevance array (len(retrieved_ids),).
    """
    relevant_set = set(relevant_ids)
    relevance = np.array([1 if doc_id in relevant_set else 0 for doc_id in retrieved_ids])
    return relevance


def evaluate_ranking(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> Dict[str, float]:
    """Evaluate a single ranking.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ranked).
        relevant_ids: List of ground-truth relevant document IDs.
        
    Returns:
        Dictionary with metric scores.
    """
    relevance = compute_relevance(retrieved_ids, relevant_ids)
    
    metrics = {
        "ndcg@10": ndcg_at_k(relevance, k=10),
        "recall@10": recall_at_k(relevance, k=10),
        "recall@100": recall_at_k(relevance, k=100),
        "mrr": mean_reciprocal_rank(relevance),
    }
    
    return metrics


def evaluate_rankings_batch(
    all_retrieved_ids: List[List[str]],
    all_relevant_ids: List[List[str]],
) -> Dict[str, float]:
    """Evaluate multiple rankings and compute average metrics.
    
    Args:
        all_retrieved_ids: List of retrieved ID lists (one per query).
        all_relevant_ids: List of relevant ID lists (one per query).
        
    Returns:
        Dictionary with averaged metric scores.
    """
    if len(all_retrieved_ids) != len(all_relevant_ids):
        raise ValueError("Number of queries mismatch")
    
    n_queries = len(all_retrieved_ids)
    if n_queries == 0:
        return {"ndcg@10": 0.0, "recall@10": 0.0, "recall@100": 0.0, "mrr": 0.0}
    
    # Compute metrics for each query
    all_metrics = []
    for retrieved, relevant in zip(all_retrieved_ids, all_relevant_ids):
        query_metrics = evaluate_ranking(retrieved, relevant)
        all_metrics.append(query_metrics)
    
    # Average metrics
    avg_metrics = {}
    metric_names = ["ndcg@10", "recall@10", "recall@100", "mrr"]
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        avg_metrics[metric_name] = np.mean(values)
    
    return avg_metrics


def bootstrap_confidence_interval(
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for metric scores.
    
    Args:
        scores: Array of metric scores (n_queries,).
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        seed: Random seed.
        
    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    rng = np.random.RandomState(seed)
    n = len(scores)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    
    return lower, upper


def permutation_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> float:
    """Perform permutation test to compare two sets of scores.
    
    Args:
        scores_a: Scores from system A (n_queries,).
        scores_b: Scores from system B (n_queries,).
        n_permutations: Number of permutations.
        seed: Random seed.
        
    Returns:
        P-value.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have same length")
    
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    
    # Observed difference
    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    
    # Permutation distribution
    perm_diffs = []
    combined = np.concatenate([scores_a, scores_b])
    
    for _ in range(n_permutations):
        # Shuffle combined scores
        rng.shuffle(combined)
        perm_a = combined[:n]
        perm_b = combined[n:]
        perm_diff = np.mean(perm_a) - np.mean(perm_b)
        perm_diffs.append(perm_diff)
    
    perm_diffs = np.array(perm_diffs)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    
    return p_value
