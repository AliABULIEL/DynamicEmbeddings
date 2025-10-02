"""Tests for hard temporal negatives."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from sentence_transformers import InputExample

from temporal_lora.train.hard_negatives import (
    HardNegativeSampler,
    get_adjacent_bins,
    tokenize_text,
    add_hard_temporal_negatives,
)


class TestAdjacentBins:
    """Test adjacent bin identification."""
    
    def test_get_adjacent_bins_middle(self):
        """Test adjacent bins for middle bin."""
        all_bins = ["bin1", "bin2", "bin3", "bin4"]
        adjacent = get_adjacent_bins("bin2", all_bins)
        
        assert set(adjacent) == {"bin1", "bin3"}
    
    def test_get_adjacent_bins_first(self):
        """Test adjacent bins for first bin."""
        all_bins = ["bin1", "bin2", "bin3"]
        adjacent = get_adjacent_bins("bin1", all_bins)
        
        # Only next bin
        assert adjacent == ["bin2"]
    
    def test_get_adjacent_bins_last(self):
        """Test adjacent bins for last bin."""
        all_bins = ["bin1", "bin2", "bin3"]
        adjacent = get_adjacent_bins("bin3", all_bins)
        
        # Only previous bin
        assert adjacent == ["bin2"]
    
    def test_get_adjacent_bins_single(self):
        """Test adjacent bins when only one bin exists."""
        all_bins = ["bin1"]
        adjacent = get_adjacent_bins("bin1", all_bins)
        
        # Should return empty or all other bins (empty in this case)
        assert adjacent == []
    
    def test_get_adjacent_bins_not_in_list(self):
        """Test when current bin not in list."""
        all_bins = ["bin1", "bin2", "bin3"]
        adjacent = get_adjacent_bins("bin_unknown", all_bins)
        
        # Should return all other bins
        assert set(adjacent) == {"bin1", "bin2", "bin3"}


class TestHardNegativeSampler:
    """Test hard negative sampling."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup
        if temp_path.exists():
            shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_data_dir(self, temp_dir):
        """Create sample bucket data for testing."""
        # Create three buckets with synthetic data
        for i, bin_name in enumerate(["bin1", "bin2", "bin3"]):
            bin_dir = temp_dir / bin_name
            bin_dir.mkdir(parents=True)
            
            # Create train.parquet
            df = pd.DataFrame({
                "paper_id": [f"paper_{bin_name}_{j}" for j in range(10)],
                "text_a": [f"Title {bin_name} {j}" for j in range(10)],
                "text_b": [f"Abstract for {bin_name} paper number {j}" for j in range(10)],
                "year": [2015 + i * 2] * 10,
            })
            df.to_parquet(bin_dir / "train.parquet")
        
        return temp_dir
    
    def test_sampler_loads_all_bins(self, sample_data_dir):
        """Test that sampler loads data from all bins."""
        all_bins = ["bin1", "bin2", "bin3"]
        
        sampler = HardNegativeSampler(
            data_dir=sample_data_dir,
            all_bins=all_bins,
            neg_k=2,
            seed=42,
        )
        
        assert len(sampler.bin_data) == 3
        assert "bin1" in sampler.bin_data
        assert "bin2" in sampler.bin_data
        assert "bin3" in sampler.bin_data
    
    def test_sampler_augments_with_negatives(self, sample_data_dir):
        """Test that sampler adds hard negatives to training examples."""
        all_bins = ["bin1", "bin2", "bin3"]
        
        sampler = HardNegativeSampler(
            data_dir=sample_data_dir,
            all_bins=all_bins,
            neg_k=2,
            seed=42,
        )
        
        # Create positive examples for bin2
        train_examples = [
            InputExample(texts=["Title 1", "Abstract 1"], label=1.0, guid="p1"),
            InputExample(texts=["Title 2", "Abstract 2"], label=1.0, guid="p2"),
        ]
        
        augmented = sampler.augment_with_hard_negatives("bin2", train_examples)
        
        # Should have more examples than original (positives + negatives)
        assert len(augmented) > len(train_examples)
        
        # Original positives should be included
        positive_count = sum(1 for ex in augmented if ex.label == 1.0)
        assert positive_count >= len(train_examples)
        
        # Should have negative examples
        negative_count = sum(1 for ex in augmented if ex.label == 0.0)
        assert negative_count > 0
    
    def test_negatives_come_from_different_bins(self, sample_data_dir):
        """Test that hard negatives come from adjacent bins, not current bin."""
        all_bins = ["bin1", "bin2", "bin3"]
        
        sampler = HardNegativeSampler(
            data_dir=sample_data_dir,
            all_bins=all_bins,
            neg_k=4,
            seed=42,
        )
        
        # Create positive examples for bin2
        train_examples = [
            InputExample(texts=["Title bin2", "Abstract bin2"], label=1.0, guid="p1"),
        ]
        
        augmented = sampler.augment_with_hard_negatives("bin2", train_examples)
        
        # Get negative examples
        negatives = [ex for ex in augmented if ex.label == 0.0]
        
        # Negatives should have guid indicating they come from adjacent bins
        for neg in negatives:
            if hasattr(neg, "guid") and neg.guid:
                # guid format: {current_bin}_neg_{i}_{adj_bin}
                assert "bin2_neg_" in neg.guid
                # Should come from bin1 or bin3 (adjacent to bin2)
                assert ("bin1" in neg.guid or "bin3" in neg.guid)
    
    def test_negatives_not_duplicates_of_positives(self, sample_data_dir):
        """Test that negatives are different from positives."""
        all_bins = ["bin1", "bin2"]
        
        sampler = HardNegativeSampler(
            data_dir=sample_data_dir,
            all_bins=all_bins,
            neg_k=2,
            seed=42,
        )
        
        train_examples = [
            InputExample(texts=["Unique text", "Unique abstract"], label=1.0, guid="p1"),
        ]
        
        augmented = sampler.augment_with_hard_negatives("bin1", train_examples)
        
        # Get texts from positives and negatives
        positive_texts = set()
        negative_texts = set()
        
        for ex in augmented:
            text = " ".join(ex.texts)
            if ex.label == 1.0:
                positive_texts.add(text)
            else:
                negative_texts.add(text)
        
        # Negatives should not be identical to positives
        overlap = positive_texts & negative_texts
        assert len(overlap) == 0, "Found duplicate texts in positives and negatives"


class TestTokenization:
    """Test tokenization utility."""
    
    def test_tokenize_text_basic(self):
        """Test basic tokenization."""
        text = "This is a test sentence"
        tokens = tokenize_text(text)
        
        assert tokens == ["this", "is", "a", "test", "sentence"]
    
    def test_tokenize_text_lowercase(self):
        """Test that tokenization lowercases."""
        text = "THIS IS UPPERCASE"
        tokens = tokenize_text(text)
        
        assert all(t.islower() for t in tokens)
    
    def test_tokenize_text_splits_on_whitespace(self):
        """Test splitting on whitespace."""
        text = "word1\tword2\nword3  word4"
        tokens = tokenize_text(text)
        
        assert len(tokens) == 4


class TestEndToEndHardNegatives:
    """End-to-end test of hard negative generation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        if temp_path.exists():
            shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_data_dir(self, temp_dir):
        """Create sample bucket data."""
        for i, bin_name in enumerate(["early", "mid", "late"]):
            bin_dir = temp_dir / bin_name
            bin_dir.mkdir(parents=True)
            
            df = pd.DataFrame({
                "paper_id": [f"paper_{bin_name}_{j}" for j in range(5)],
                "text_a": [f"Title {bin_name} sample {j}" for j in range(5)],
                "text_b": [f"Abstract discussing {bin_name} period research topic {j}" for j in range(5)],
                "year": [2010 + i * 5] * 5,
            })
            df.to_parquet(bin_dir / "train.parquet")
        
        return temp_dir
    
    def test_add_hard_temporal_negatives_increases_examples(self, sample_data_dir):
        """Test that hard negatives increase training set size."""
        all_bins = ["early", "mid", "late"]
        
        train_examples = [
            InputExample(texts=["Test title", "Test abstract"], label=1.0),
        ]
        
        augmented = add_hard_temporal_negatives(
            data_dir=sample_data_dir,
            all_bins=all_bins,
            bucket_name="mid",
            train_examples=train_examples,
            neg_k=2,
            seed=42,
        )
        
        # Should have more examples
        assert len(augmented) > len(train_examples)
        
        # Should have both positives and negatives
        labels = [ex.label for ex in augmented]
        assert 1.0 in labels  # positives
        assert 0.0 in labels  # negatives


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
