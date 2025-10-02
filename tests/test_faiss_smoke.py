"""Smoke tests for FAISS functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_build_faiss_index():
    """Test FAISS index building."""
    from temporal_lora.eval.indexes import build_faiss_index
    
    # Create synthetic embeddings
    n_docs = 100
    dim = 384
    embeddings = np.random.randn(n_docs, dim).astype(np.float32)
    
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Build index
    index = build_faiss_index(embeddings)
    
    assert index.ntotal == n_docs
    assert index.d == dim


def test_save_load_index():
    """Test saving and loading FAISS index."""
    from temporal_lora.eval.indexes import build_faiss_index, save_faiss_index, load_faiss_index
    
    # Create index
    embeddings = np.random.randn(50, 384).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = build_faiss_index(embeddings)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        index_path = tmpdir / "test.faiss"
        
        # Save
        save_faiss_index(index, index_path)
        assert index_path.exists()
        
        # Load
        loaded_index = load_faiss_index(index_path)
        assert loaded_index.ntotal == index.ntotal
        assert loaded_index.d == index.d


def test_query_roundtrip():
    """Test that querying returns correct documents."""
    from temporal_lora.eval.indexes import build_faiss_index, query_index
    
    # Create embeddings where each is orthogonal (for clean test)
    n_docs = 10
    dim = 384
    embeddings = np.eye(n_docs, dim, dtype=np.float32)
    
    # Build index
    index = build_faiss_index(embeddings)
    
    # Query with first embedding (should return itself as top result)
    query = embeddings[0:1]
    scores, indices = query_index(index, query, k=5)
    
    # First result should be document 0 with high score
    assert indices[0, 0] == 0
    assert scores[0, 0] > 0.9  # High similarity to itself


def test_encoder_cache():
    """Test encoding and caching embeddings."""
    from temporal_lora.eval.encoder import save_embeddings, load_embeddings
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic embeddings
        embeddings = np.random.randn(20, 384).astype(np.float32)
        ids = [f"doc_{i}" for i in range(20)]
        
        # Save
        save_embeddings(embeddings, ids, tmpdir)
        
        # Load
        loaded_embeddings, loaded_ids = load_embeddings(tmpdir)
        
        assert loaded_embeddings.shape == embeddings.shape
        assert loaded_ids == ids
        assert np.allclose(loaded_embeddings, embeddings)


def test_metrics_ndcg():
    """Test NDCG computation."""
    from temporal_lora.eval.metrics import ndcg_at_k
    
    # Perfect ranking
    relevance = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    ndcg = ndcg_at_k(relevance, k=10)
    assert ndcg == 1.0
    
    # Worst ranking
    relevance = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    ndcg = ndcg_at_k(relevance, k=10)
    assert ndcg < 1.0
    
    # No relevant docs
    relevance = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ndcg = ndcg_at_k(relevance, k=10)
    assert ndcg == 0.0


def test_metrics_recall():
    """Test Recall computation."""
    from temporal_lora.eval.metrics import recall_at_k
    
    # Perfect recall@5
    relevance = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    recall = recall_at_k(relevance, k=5)
    assert recall == 1.0
    
    # Partial recall@5
    relevance = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 0])
    recall = recall_at_k(relevance, k=5)
    assert recall == 0.4  # 2 out of 5 relevant docs found


def test_metrics_mrr():
    """Test MRR computation."""
    from temporal_lora.eval.metrics import mean_reciprocal_rank
    
    # First position
    relevance = np.array([1, 0, 0, 0, 0])
    mrr = mean_reciprocal_rank(relevance)
    assert mrr == 1.0
    
    # Third position
    relevance = np.array([0, 0, 1, 0, 0])
    mrr = mean_reciprocal_rank(relevance)
    assert mrr == pytest.approx(1.0 / 3.0)
    
    # No relevant
    relevance = np.array([0, 0, 0, 0, 0])
    mrr = mean_reciprocal_rank(relevance)
    assert mrr == 0.0


def test_multi_index_search():
    """Test multi-index search with merge strategies."""
    from temporal_lora.eval.indexes import build_faiss_index, multi_index_search
    
    # Create two indexes with different documents
    embeddings_1 = np.random.randn(20, 384).astype(np.float32)
    embeddings_1 = embeddings_1 / np.linalg.norm(embeddings_1, axis=1, keepdims=True)
    embeddings_2 = np.random.randn(30, 384).astype(np.float32)
    embeddings_2 = embeddings_2 / np.linalg.norm(embeddings_2, axis=1, keepdims=True)
    
    index_1 = build_faiss_index(embeddings_1)
    index_2 = build_faiss_index(embeddings_2)
    
    indexes = {"bucket1": index_1, "bucket2": index_2}
    index_ids = {
        "bucket1": [f"doc1_{i}" for i in range(20)],
        "bucket2": [f"doc2_{i}" for i in range(30)],
    }
    
    # Query
    query = np.random.randn(1, 384).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    # Test different merge strategies
    for merge_strategy in ["softmax", "mean", "max", "rrf"]:
        doc_ids, scores = multi_index_search(
            indexes, index_ids, query, k=10, merge_strategy=merge_strategy
        )
        
        assert len(doc_ids) == 1  # One query
        assert len(doc_ids[0]) == 10  # Top 10 results
        assert len(scores) == 1
        assert len(scores[0]) == 10
        
        # Check IDs are from both buckets or either bucket
        all_ids = doc_ids[0]
        assert all(id.startswith("doc1_") or id.startswith("doc2_") for id in all_ids)
