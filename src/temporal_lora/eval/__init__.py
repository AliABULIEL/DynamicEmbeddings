"""Evaluation components for retrieval."""

from .encoder import (
    encode_batch,
    encode_and_cache_bucket,
    save_embeddings,
    load_embeddings,
)
from .indexes import (
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
    query_index,
    build_bucket_indexes,
    multi_index_search,
)
from .metrics import (
    ndcg_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    evaluate_ranking,
    evaluate_rankings_batch,
    bootstrap_confidence_interval,
    permutation_test,
)
from .evaluate import (
    evaluate_bucket_pair,
    evaluate_cross_bucket_matrix,
    evaluate_multi_index_with_temperature,
    evaluate_mode,
    run_temperature_sweep,
    run_full_evaluation,
    compare_modes,
)

__all__ = [
    # encoder
    "encode_batch",
    "encode_and_cache_bucket",
    "save_embeddings",
    "load_embeddings",
    # indexes
    "build_faiss_index",
    "save_faiss_index",
    "load_faiss_index",
    "query_index",
    "build_bucket_indexes",
    "multi_index_search",
    # metrics
    "ndcg_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "evaluate_ranking",
    "evaluate_rankings_batch",
    "bootstrap_confidence_interval",
    "permutation_test",
    # evaluate
    "evaluate_bucket_pair",
    "evaluate_cross_bucket_matrix",
    "evaluate_multi_index_with_temperature",
    "evaluate_mode",
    "run_temperature_sweep",
    "run_full_evaluation",
    "compare_modes",
]
