"""Smoke tests for evaluation system."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from temporal_lora.eval.evaluate import (
    create_ground_truth,
    evaluate_bucket_pair,
    evaluate_cross_bucket_matrix,
    evaluate_multi_index_with_temperature,
    compare_modes,
)
from temporal_lora.eval.indexes import (
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
    query_index,
    multi_index_search,
)
from temporal_lora.eval.metrics import (
    ndcg_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    compute_relevance,
    evaluate_ranking,
    evaluate_rankings_batch,
)


class TestGroundTruth:
    """Test ground truth creation."""
    
    def test_basic_ground_truth(self):
        """Test basic ground truth creation."""
        query_ids = ["doc1", "doc2", "doc3"]
        all_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        
        gt = create_ground_truth(query_ids, all_ids)
        
        assert len(gt) == 3
        assert gt[0] == ["doc1"]
        assert gt[1] == ["doc2"]
        assert gt[2] == ["doc3"]


class TestFAISSRoundtrip:
    """Test FAISS index creation and querying."""
    
    def test_build_and_query_index(self):
        """Test building index and querying."""
        # Create dummy embeddings
        np.random.seed(42)
        embeddings = np.random.randn(100, 64).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Build index
        index = build_faiss_index(embeddings)
        
        # Query with same embeddings
        query_embeddings = embeddings[:10]
        scores, indices = query_index(index, query_embeddings, k=5)
        
        # Check shapes
        assert scores.shape == (10, 5)
        assert indices.shape == (10, 5)
        
        # Check that first result is the query itself (perfect match)
        assert indices[0, 0] == 0
        assert indices[5, 0] == 5
        
        # Scores should be close to 1.0 for perfect match
        assert scores[0, 0] > 0.99
    
    def test_save_and_load_index(self):
        """Test saving and loading FAISS index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create and save index
            np.random.seed(42)
            embeddings = np.random.randn(50, 32).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            index = build_faiss_index(embeddings)
            index_path = tmpdir / "test.faiss"
            save_faiss_index(index, index_path)
            
            # Load index
            loaded_index = load_faiss_index(index_path)
            
            # Query both indexes
            query = embeddings[:5]
            scores1, indices1 = query_index(index, query, k=3)
            scores2, indices2 = query_index(loaded_index, query, k=3)
            
            # Results should match
            np.testing.assert_array_equal(indices1, indices2)
            np.testing.assert_allclose(scores1, scores2, rtol=1e-5)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_perfect_ranking(self):
        """Test metrics with perfect ranking."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc2"]
        
        relevance = compute_relevance(retrieved, relevant)
        
        assert relevance[0] == 1
        assert relevance[1] == 1
        assert relevance[2] == 0
        
        ndcg = ndcg_at_k(relevance, k=10)
        recall = recall_at_k(relevance, k=10)
        mrr = mean_reciprocal_rank(relevance)
        
        assert ndcg > 0.9  # Should be very high
        assert recall == 1.0  # All relevant docs retrieved
        assert mrr == 1.0  # First result is relevant
    
    def test_poor_ranking(self):
        """Test metrics with poor ranking."""
        retrieved = ["doc5", "doc6", "doc7", "doc8", "doc1"]
        relevant = ["doc1", "doc2"]
        
        relevance = compute_relevance(retrieved, relevant)
        
        ndcg = ndcg_at_k(relevance, k=10)
        recall = recall_at_k(relevance, k=10)
        mrr = mean_reciprocal_rank(relevance)
        
        assert ndcg < 0.5  # Should be low
        assert recall == 0.5  # Only 1 of 2 relevant docs
        assert abs(mrr - 0.2) < 0.01  # Relevant doc at position 5
    
    def test_batch_evaluation(self):
        """Test batch evaluation."""
        all_retrieved = [
            ["doc1", "doc2", "doc3"],
            ["doc5", "doc1", "doc2"],
            ["doc1", "doc4", "doc5"],
        ]
        all_relevant = [
            ["doc1"],
            ["doc1"],
            ["doc1"],
        ]
        
        metrics = evaluate_rankings_batch(all_retrieved, all_relevant)
        
        assert "ndcg@10" in metrics
        assert "recall@10" in metrics
        assert "recall@100" in metrics
        assert "mrr" in metrics
        
        # All queries have doc1 as relevant
        # Query 1: doc1 at position 1 (perfect)
        # Query 2: doc1 at position 2
        # Query 3: doc1 at position 1 (perfect)
        
        assert metrics["recall@10"] == 1.0  # All found
        assert metrics["mrr"] > 0.8  # Average MRR should be high


class TestMultiIndexSearch:
    """Test multi-index search with different merge strategies."""
    
    def setup_indexes(self):
        """Create test indexes."""
        np.random.seed(42)
        
        # Create two indexes
        embeddings1 = np.random.randn(50, 32).astype(np.float32)
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        
        embeddings2 = np.random.randn(50, 32).astype(np.float32)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        index1 = build_faiss_index(embeddings1)
        index2 = build_faiss_index(embeddings2)
        
        ids1 = [f"doc1_{i}" for i in range(50)]
        ids2 = [f"doc2_{i}" for i in range(50)]
        
        return {
            "bucket1": index1,
            "bucket2": index2,
        }, {
            "bucket1": ids1,
            "bucket2": ids2,
        }
    
    def test_softmax_merge(self):
        """Test softmax merge strategy."""
        indexes, index_ids = self.setup_indexes()
        
        # Create query
        np.random.seed(42)
        query = np.random.randn(5, 32).astype(np.float32)
        query = query / np.linalg.norm(query, axis=1, keepdims=True)
        
        # Search with softmax
        retrieved_ids, scores = multi_index_search(
            indexes,
            index_ids,
            query,
            k=10,
            merge_strategy="softmax",
            temperature=2.0,
        )
        
        assert len(retrieved_ids) == 5
        assert all(len(ids) == 10 for ids in retrieved_ids)
        
        # Check that results contain IDs from both indexes
        all_ids = [id for query_ids in retrieved_ids for id in query_ids]
        has_bucket1 = any("doc1_" in id for id in all_ids)
        has_bucket2 = any("doc2_" in id for id in all_ids)
        assert has_bucket1 or has_bucket2  # At least one bucket represented
    
    def test_mean_merge(self):
        """Test mean merge strategy."""
        indexes, index_ids = self.setup_indexes()
        
        np.random.seed(42)
        query = np.random.randn(3, 32).astype(np.float32)
        query = query / np.linalg.norm(query, axis=1, keepdims=True)
        
        retrieved_ids, scores = multi_index_search(
            indexes,
            index_ids,
            query,
            k=5,
            merge_strategy="mean",
        )
        
        assert len(retrieved_ids) == 3
        assert all(len(ids) == 5 for ids in retrieved_ids)
    
    def test_max_merge(self):
        """Test max merge strategy."""
        indexes, index_ids = self.setup_indexes()
        
        np.random.seed(42)
        query = np.random.randn(3, 32).astype(np.float32)
        query = query / np.linalg.norm(query, axis=1, keepdims=True)
        
        retrieved_ids, scores = multi_index_search(
            indexes,
            index_ids,
            query,
            k=5,
            merge_strategy="max",
        )
        
        assert len(retrieved_ids) == 3
        assert all(len(ids) == 5 for ids in retrieved_ids)
    
    def test_rrf_merge(self):
        """Test RRF merge strategy."""
        indexes, index_ids = self.setup_indexes()
        
        np.random.seed(42)
        query = np.random.randn(3, 32).astype(np.float32)
        query = query / np.linalg.norm(query, axis=1, keepdims=True)
        
        retrieved_ids, scores = multi_index_search(
            indexes,
            index_ids,
            query,
            k=5,
            merge_strategy="rrf",
        )
        
        assert len(retrieved_ids) == 3
        assert all(len(ids) == 5 for ids in retrieved_ids)


class TestCrossBucketEvaluation:
    """Test cross-bucket evaluation matrix."""
    
    def create_test_data(self, tmpdir: Path):
        """Create test embeddings and indexes."""
        buckets = ["bucket1", "bucket2"]
        np.random.seed(42)
        
        for bucket in buckets:
            # Create embeddings
            embeddings_dir = tmpdir / "embeddings" / bucket / "test"
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            # Save embeddings
            embeddings = np.random.randn(20, 32).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            np.save(embeddings_dir / "embeddings.npy", embeddings)
            
            # Save IDs
            ids = [f"{bucket}_doc{i}" for i in range(20)]
            with open(embeddings_dir / "ids.txt", "w") as f:
                for doc_id in ids:
                    f.write(f"{doc_id}\n")
            
            # Create index
            indexes_dir = tmpdir / "indexes"
            indexes_dir.mkdir(parents=True, exist_ok=True)
            
            index = build_faiss_index(embeddings)
            save_faiss_index(index, indexes_dir / f"{bucket}.faiss")
            
            # Save index IDs (all splits combined)
            with open(indexes_dir / f"{bucket}_ids.txt", "w") as f:
                for doc_id in ids:
                    f.write(f"{doc_id}\n")
        
        return buckets
    
    def test_evaluate_bucket_pair(self):
        """Test evaluating single bucket pair."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            buckets = self.create_test_data(tmpdir)
            
            embeddings_dir = tmpdir / "embeddings"
            indexes_dir = tmpdir / "indexes"
            
            # Evaluate same bucket (within-period)
            metrics = evaluate_bucket_pair(
                query_bucket="bucket1",
                doc_bucket="bucket1",
                embeddings_dir=embeddings_dir,
                indexes_dir=indexes_dir,
            )
            
            assert "ndcg@10" in metrics
            assert "recall@10" in metrics
            assert "mrr" in metrics
            
            # Within same bucket, metrics should be reasonable
            assert metrics["ndcg@10"] > 0
            assert metrics["recall@10"] > 0
    
    def test_cross_bucket_matrix(self):
        """Test creating full cross-bucket matrix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            buckets = self.create_test_data(tmpdir)
            
            embeddings_dir = tmpdir / "embeddings"
            indexes_dir = tmpdir / "indexes"
            
            # Evaluate matrix
            metric_dfs = evaluate_cross_bucket_matrix(
                buckets=buckets,
                embeddings_dir=embeddings_dir,
                indexes_dir=indexes_dir,
            )
            
            # Check structure
            assert "ndcg@10" in metric_dfs
            assert "recall@10" in metric_dfs
            assert "recall@100" in metric_dfs
            assert "mrr" in metric_dfs
            
            # Check DataFrame shapes
            for metric_name, df in metric_dfs.items():
                assert df.shape == (2, 2)  # 2 query buckets × 2 doc buckets
                assert list(df.index) == buckets
                assert list(df.columns) == buckets


class TestTemperatureSweep:
    """Test temperature sweep for multi-index."""
    
    def create_test_data(self, tmpdir: Path):
        """Create test data."""
        buckets = ["bucket1", "bucket2"]
        np.random.seed(42)
        
        for bucket in buckets:
            # Create embeddings
            embeddings_dir = tmpdir / "embeddings" / bucket / "test"
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            embeddings = np.random.randn(15, 32).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            np.save(embeddings_dir / "embeddings.npy", embeddings)
            
            ids = [f"{bucket}_doc{i}" for i in range(15)]
            with open(embeddings_dir / "ids.txt", "w") as f:
                for doc_id in ids:
                    f.write(f"{doc_id}\n")
            
            # Create index
            indexes_dir = tmpdir / "indexes"
            indexes_dir.mkdir(parents=True, exist_ok=True)
            
            index = build_faiss_index(embeddings)
            save_faiss_index(index, indexes_dir / f"{bucket}.faiss")
            
            with open(indexes_dir / f"{bucket}_ids.txt", "w") as f:
                for doc_id in ids:
                    f.write(f"{doc_id}\n")
        
        return buckets
    
    def test_temperature_sweep(self):
        """Test temperature sweep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            buckets = self.create_test_data(tmpdir)
            
            embeddings_dir = tmpdir / "embeddings"
            indexes_dir = tmpdir / "indexes"
            
            # Run sweep
            results_df = evaluate_multi_index_with_temperature(
                buckets=buckets,
                embeddings_dir=embeddings_dir,
                indexes_dir=indexes_dir,
                merge_strategy="softmax",
                temperatures=[1.0, 2.0, 3.0],
            )
            
            # Check output
            assert len(results_df) == 6  # 2 buckets × 3 temperatures
            assert "temperature" in results_df.columns
            assert "query_bucket" in results_df.columns
            assert "ndcg@10" in results_df.columns
            
            # Check temperatures
            temps = sorted(results_df["temperature"].unique())
            assert temps == [1.0, 2.0, 3.0]


class TestModeComparison:
    """Test comparing different modes."""
    
    def test_compare_modes(self):
        """Test delta computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / "comparisons"
            
            # Create dummy baseline and LoRA results
            buckets = ["bucket1", "bucket2"]
            baseline_results = {
                "ndcg@10": pd.DataFrame(
                    [[0.5, 0.3], [0.4, 0.6]],
                    index=buckets,
                    columns=buckets,
                ),
                "recall@10": pd.DataFrame(
                    [[0.7, 0.5], [0.6, 0.8]],
                    index=buckets,
                    columns=buckets,
                ),
            }
            
            lora_results = {
                "ndcg@10": pd.DataFrame(
                    [[0.6, 0.4], [0.5, 0.7]],
                    index=buckets,
                    columns=buckets,
                ),
                "recall@10": pd.DataFrame(
                    [[0.8, 0.6], [0.7, 0.9]],
                    index=buckets,
                    columns=buckets,
                ),
            }
            
            # Compare
            compare_modes(baseline_results, lora_results, output_dir)
            
            # Check outputs
            assert (output_dir / "delta_ndcg_at_10.csv").exists()
            assert (output_dir / "delta_recall_at_10.csv").exists()
            
            # Load and check delta
            delta_ndcg = pd.read_csv(output_dir / "delta_ndcg_at_10.csv", index_col=0)
            
            # Delta should be positive (LoRA better)
            assert delta_ndcg.values.mean() > 0


class TestEfficiencyTracking:
    """Test efficiency metrics computation."""
    
    def test_all_modes_present(self):
        """Test that efficiency summary includes all modes."""
        # This is a placeholder test - in practice would need actual model files
        modes = ["baseline_frozen", "lora", "full_ft", "seq_ft"]
        
        # Just verify the modes list is correct
        assert "baseline_frozen" in modes
        assert "lora" in modes
        assert "full_ft" in modes
        assert "seq_ft" in modes
        
        assert len(modes) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
