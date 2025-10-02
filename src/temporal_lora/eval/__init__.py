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
    evaluate_within_period,
    evaluate_cross_period,
    evaluate_multi_index,
    run_evaluation,
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
    "evaluate_within_period",
    "evaluate_cross_period",
    "evaluate_multi_index",
    "run_evaluation",
]
