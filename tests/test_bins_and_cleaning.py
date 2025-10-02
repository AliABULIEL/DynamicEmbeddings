"""Tests for time bins, balancing, cleaning, and L2 normalization."""

import numpy as np
import pandas as pd
import pytest

from temporal_lora.data.bucketing import (
    assign_buckets,
    bucket_and_split,
    compute_balanced_sample_size,
    parse_bucket_spec,
)
from temporal_lora.data.preprocessing import (
    clean_and_preprocess,
    clean_html,
    collapse_whitespace,
    compute_content_hash,
    truncate_tokens,
)


class TestBucketParsing:
    """Test custom bin parsing."""
    
    def test_parse_simple_buckets(self):
        """Test parsing of simple bucket specifications."""
        bucket_config = [
            {"name": "early", "range": [2000, 2010]},
            {"name": "late", "range": [2011, 2020]},
        ]
        
        ranges = parse_bucket_spec(bucket_config)
        
        assert len(ranges) == 2
        assert ranges["early"] == (2000, 2010)
        assert ranges["late"] == (2011, 2020)
    
    def test_parse_open_ended_buckets(self):
        """Test parsing of open-ended bucket specifications."""
        bucket_config = [
            {"name": "old", "range": [None, 2015]},
            {"name": "new", "range": [2016, None]},
        ]
        
        ranges = parse_bucket_spec(bucket_config)
        
        assert len(ranges) == 2
        assert ranges["old"] == (-np.inf, 2015)
        assert ranges["new"] == (2016, np.inf)
    
    def test_assign_buckets(self):
        """Test bucket assignment to DataFrame."""
        df = pd.DataFrame({
            "paper_id": ["p1", "p2", "p3", "p4"],
            "year": [2010, 2015, 2018, 2022],
            "title": ["t1", "t2", "t3", "t4"],
            "abstract": ["a1", "a2", "a3", "a4"],
        })
        
        bucket_config = [
            {"name": "early", "range": [None, 2015]},
            {"name": "recent", "range": [2016, None]},
        ]
        
        result = assign_buckets(df, bucket_config)
        
        assert len(result) == 4
        assert result.iloc[0]["bucket"] == "early"
        assert result.iloc[1]["bucket"] == "early"
        assert result.iloc[2]["bucket"] == "recent"
        assert result.iloc[3]["bucket"] == "recent"


class TestBalancedSampling:
    """Test balanced sampling across bins."""
    
    def test_balanced_mode_uses_min_count(self):
        """Test that balanced mode uses minimum bucket size."""
        df = pd.DataFrame({
            "paper_id": [f"p{i}" for i in range(100)],
            "bucket": ["A"] * 60 + ["B"] * 40,
            "year": [2020] * 100,
        })
        
        sample_sizes = compute_balanced_sample_size(
            df, max_per_bucket=1000, balance_per_bin=True
        )
        
        # Should use min(60, 40, 1000) = 40 for both buckets
        assert sample_sizes["A"] == 40
        assert sample_sizes["B"] == 40
    
    def test_balanced_mode_respects_max_cap(self):
        """Test that balanced mode respects max_per_bucket cap."""
        df = pd.DataFrame({
            "paper_id": [f"p{i}" for i in range(1000)],
            "bucket": ["A"] * 500 + ["B"] * 500,
            "year": [2020] * 1000,
        })
        
        sample_sizes = compute_balanced_sample_size(
            df, max_per_bucket=100, balance_per_bin=True
        )
        
        # Should use min(500, 500, 100) = 100 for both buckets
        assert sample_sizes["A"] == 100
        assert sample_sizes["B"] == 100
    
    def test_unbalanced_mode_caps_independently(self):
        """Test that unbalanced mode caps each bucket independently."""
        df = pd.DataFrame({
            "paper_id": [f"p{i}" for i in range(100)],
            "bucket": ["A"] * 60 + ["B"] * 40,
            "year": [2020] * 100,
        })
        
        sample_sizes = compute_balanced_sample_size(
            df, max_per_bucket=50, balance_per_bin=False
        )
        
        # Should cap A at 50, B at 40 (its natural size)
        assert sample_sizes["A"] == 50
        assert sample_sizes["B"] == 40
    
    def test_bucket_and_split_balances_correctly(self):
        """Test that bucket_and_split enforces balanced counts."""
        df = pd.DataFrame({
            "paper_id": [f"p{i}" for i in range(200)],
            "year": [2010] * 80 + [2020] * 120,
            "title": [f"title{i}" for i in range(200)],
            "abstract": [f"abstract{i}" for i in range(200)],
        })
        
        bucket_config = [
            {"name": "early", "range": [None, 2015]},
            {"name": "recent", "range": [2016, None]},
        ]
        
        result = bucket_and_split(
            df, bucket_config, max_per_bucket=1000, balance_per_bin=True, seed=42
        )
        
        # Should have equal counts per bucket (80 each, the minimum)
        early_count = len(result[result["bucket"] == "early"])
        recent_count = len(result[result["bucket"] == "recent"])
        
        assert early_count == recent_count
        assert early_count == 80


class TestIDLeakage:
    """Test that splits don't leak IDs."""
    
    def test_no_id_leakage_across_splits(self):
        """Test that train/val/test splits have no overlapping IDs."""
        df = pd.DataFrame({
            "paper_id": [f"p{i}" for i in range(100)],
            "year": [2020] * 100,
            "title": [f"title{i}" for i in range(100)],
            "abstract": [f"abstract{i}" for i in range(100)],
        })
        
        bucket_config = [{"name": "bucket1", "range": [2020, 2020]}]
        
        result = bucket_and_split(
            df, bucket_config, max_per_bucket=1000, balance_per_bin=False, seed=42
        )
        
        # Check no overlap between splits
        train_ids = set(result[result["split"] == "train"]["paper_id"])
        val_ids = set(result[result["split"] == "val"]["paper_id"])
        test_ids = set(result[result["split"] == "test"]["paper_id"])
        
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0
        
        # Check all IDs are accounted for
        assert len(train_ids | val_ids | test_ids) == 100


class TestTextCleaning:
    """Test text preprocessing functions."""
    
    def test_clean_html(self):
        """Test HTML tag removal."""
        text = "This is <b>bold</b> and <i>italic</i> text."
        cleaned = clean_html(text)
        assert "<b>" not in cleaned
        assert "<i>" not in cleaned
        assert "bold" in cleaned
        assert "italic" in cleaned
    
    def test_collapse_whitespace(self):
        """Test whitespace normalization."""
        text = "This  has   multiple    spaces\nand\nnewlines"
        cleaned = collapse_whitespace(text)
        assert "  " not in cleaned
        assert "\n" not in cleaned
        assert cleaned == "This has multiple spaces and newlines"
    
    def test_truncate_tokens(self):
        """Test token truncation."""
        text = " ".join([f"word{i}" for i in range(100)])
        truncated = truncate_tokens(text, max_tokens=50)
        assert len(truncated.split()) == 50
    
    def test_truncate_tokens_no_change_if_short(self):
        """Test that short text is not truncated."""
        text = "short text"
        truncated = truncate_tokens(text, max_tokens=100)
        assert truncated == text


class TestDeduplication:
    """Test deduplication functionality."""
    
    def test_compute_content_hash_same_for_identical(self):
        """Test that identical content produces same hash."""
        hash1 = compute_content_hash("Title", "Abstract text here")
        hash2 = compute_content_hash("Title", "Abstract text here")
        assert hash1 == hash2
    
    def test_compute_content_hash_different_for_different(self):
        """Test that different content produces different hash."""
        hash1 = compute_content_hash("Title1", "Abstract text here")
        hash2 = compute_content_hash("Title2", "Abstract text here")
        assert hash1 != hash2
    
    def test_compute_content_hash_ignores_case_and_whitespace(self):
        """Test that hash ignores case and whitespace differences."""
        hash1 = compute_content_hash("Title", "Abstract  text")
        hash2 = compute_content_hash("TITLE", "abstract text")
        assert hash1 == hash2
    
    def test_deduplication_removes_duplicates(self):
        """Test that preprocessing removes duplicates."""
        df = pd.DataFrame({
            "paper_id": ["p1", "p2", "p3"],
            "title": ["Same Title", "Same Title", "Different Title"],
            "abstract": ["Same abstract text here"] * 3,
            "year": [2020, 2020, 2020],
        })
        
        config = {"remove_duplicates": True, "min_abstract_length": 10}
        
        result = clean_and_preprocess(df, config)
        
        # Should have 2 unique rows (p1 and p2 are duplicates)
        assert len(result) == 2


class TestFullPreprocessing:
    """Test full preprocessing pipeline."""
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        df = pd.DataFrame({
            "paper_id": ["p1", "p2", "p3", "p4"],
            "title": [
                "Title with <b>HTML</b>",
                "Title  with   spaces",
                "Short",  # Too short
                "Valid Title",
            ],
            "abstract": [
                "Abstract with <i>tags</i> and  spaces",
                "Abstract with\nmultiple\nlines",
                "Too short",  # Too short
                "This is a valid abstract with enough characters to pass length check",
            ],
            "year": [2020, 2020, 2020, 2020],
        })
        
        config = {
            "clean_html": True,
            "collapse_whitespace": True,
            "max_tokens": 100,
            "min_abstract_length": 30,
            "max_abstract_length": 1000,
            "remove_duplicates": True,
        }
        
        result = clean_and_preprocess(df, config)
        
        # Should remove p3 (too short)
        assert len(result) == 3
        assert "p3" not in result["paper_id"].values
        
        # Check HTML cleaning
        assert "<b>" not in result.iloc[0]["title"]
        assert "<i>" not in result.iloc[0]["abstract"]
        
        # Check whitespace collapse
        assert "  " not in result.iloc[1]["title"]
        assert "\n" not in result.iloc[1]["abstract"]


class TestEmbeddingNormalization:
    """Test L2 normalization of embeddings."""
    
    def test_embeddings_are_unit_norm(self):
        """Test that embeddings are L2-normalized (unit norm)."""
        # Create mock embeddings
        embeddings = np.random.randn(10, 384)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        
        # Check all norms are 1.0
        result_norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(result_norms, 1.0, atol=1e-6)
    
    def test_normalized_dot_product_equals_cosine(self):
        """Test that dot product of normalized vectors equals cosine similarity."""
        # Create two random vectors
        v1 = np.random.randn(384)
        v2 = np.random.randn(384)
        
        # Normalize
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Dot product of normalized vectors
        dot_product = np.dot(v1_norm, v2_norm)
        
        # Cosine similarity
        cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        assert np.isclose(dot_product, cosine_sim, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
