"""Unit tests for data ingestion, bucketing, and pairs."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from temporal_lora.data import (
    BucketConfig,
    assign_bucket,
    create_pairs,
    create_splits,
    load_dataset_from_csv,
)
from temporal_lora.data.datasets import DatasetValidationError


class TestDatasetLoading:
    """Tests for dataset loading and validation."""

    def test_load_csv_success(self, tmp_path):
        """Test successful CSV loading with valid schema."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame(
            {
                "paper_id": ["p1", "p2", "p3"],
                "title": ["Title 1", "Title 2", "Title 3"],
                "abstract": ["Abstract 1", "Abstract 2", "Abstract 3"],
                "year": [2017, 2020, 2023],
            }
        )
        df.to_csv(csv_path, index=False)

        loaded_df = load_dataset_from_csv(csv_path)
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["paper_id", "title", "abstract", "year"]

    def test_load_csv_missing_file(self, tmp_path):
        """Test error handling for missing CSV file."""
        csv_path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError) as exc_info:
            load_dataset_from_csv(csv_path)
        assert "not found" in str(exc_info.value).lower()

    def test_load_csv_missing_columns(self, tmp_path):
        """Test error handling for missing required columns."""
        csv_path = tmp_path / "incomplete.csv"
        df = pd.DataFrame(
            {
                "paper_id": ["p1", "p2"],
                "title": ["Title 1", "Title 2"],
                # Missing 'abstract' and 'year'
            }
        )
        df.to_csv(csv_path, index=False)

        with pytest.raises(DatasetValidationError) as exc_info:
            load_dataset_from_csv(csv_path)
        assert "missing required columns" in str(exc_info.value).lower()
        assert "abstract" in str(exc_info.value)
        assert "year" in str(exc_info.value)

    def test_load_csv_invalid_year_type(self, tmp_path):
        """Test error handling for non-numeric year column."""
        csv_path = tmp_path / "bad_year.csv"
        df = pd.DataFrame(
            {
                "paper_id": ["p1", "p2"],
                "title": ["Title 1", "Title 2"],
                "abstract": ["Abstract 1", "Abstract 2"],
                "year": ["not_a_year", "2020"],  # Invalid year
            }
        )
        df.to_csv(csv_path, index=False)

        with pytest.raises(DatasetValidationError) as exc_info:
            load_dataset_from_csv(csv_path)
        assert "year" in str(exc_info.value).lower()
        assert "numeric" in str(exc_info.value).lower()

    def test_load_csv_null_values(self, tmp_path):
        """Test error handling for null values in required columns."""
        csv_path = tmp_path / "nulls.csv"
        df = pd.DataFrame(
            {
                "paper_id": ["p1", "p2", None],
                "title": ["Title 1", None, "Title 3"],
                "abstract": ["Abstract 1", "Abstract 2", "Abstract 3"],
                "year": [2020, 2021, 2022],
            }
        )
        df.to_csv(csv_path, index=False)

        with pytest.raises(DatasetValidationError) as exc_info:
            load_dataset_from_csv(csv_path)
        assert "null values" in str(exc_info.value).lower()

    def test_load_csv_duplicate_paper_ids(self, tmp_path):
        """Test error handling for duplicate paper IDs."""
        csv_path = tmp_path / "duplicates.csv"
        df = pd.DataFrame(
            {
                "paper_id": ["p1", "p2", "p1"],  # Duplicate p1
                "title": ["Title 1", "Title 2", "Title 3"],
                "abstract": ["Abstract 1", "Abstract 2", "Abstract 3"],
                "year": [2020, 2021, 2022],
            }
        )
        df.to_csv(csv_path, index=False)

        with pytest.raises(DatasetValidationError) as exc_info:
            load_dataset_from_csv(csv_path)
        assert "duplicate" in str(exc_info.value).lower()
        assert "paper_id" in str(exc_info.value).lower()


class TestBucketing:
    """Tests for time-based bucketing logic."""

    def test_bucket_config_validation(self):
        """Test bucket configuration validation."""
        # Valid config
        config = BucketConfig(
            boundaries=[("≤2018", 2018), ("2019-2024", 2024)],
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
        )
        assert len(config.boundaries) == 2

        # Invalid: ratios don't sum to 1.0
        with pytest.raises(ValueError) as exc_info:
            BucketConfig(
                boundaries=[("≤2018", 2018)],
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.1,  # Sum = 0.9, not 1.0
            )
        assert "sum to 1.0" in str(exc_info.value)

        # Invalid: boundaries not sorted
        with pytest.raises(ValueError) as exc_info:
            BucketConfig(
                boundaries=[("2019-2024", 2024), ("≤2018", 2018)]  # Wrong order
            )
        assert "sorted" in str(exc_info.value).lower()

        # Invalid: duplicate boundaries
        with pytest.raises(ValueError) as exc_info:
            BucketConfig(boundaries=[("bucket1", 2020), ("bucket2", 2020)])
        assert "duplicate" in str(exc_info.value).lower()

    def test_assign_bucket_boundaries(self):
        """Test bucket assignment at boundaries."""
        config = BucketConfig(
            boundaries=[("≤2018", 2018), ("2019-2024", 2024)]
        )

        assert assign_bucket(2015, config) == "≤2018"
        assert assign_bucket(2018, config) == "≤2018"  # Boundary inclusive
        assert assign_bucket(2019, config) == "2019-2024"
        assert assign_bucket(2024, config) == "2019-2024"  # Boundary inclusive
        assert assign_bucket(2025, config) == "2019-2024"  # Beyond last boundary

    def test_assign_bucket_multiple_buckets(self):
        """Test bucket assignment with 3 buckets."""
        config = BucketConfig(
            boundaries=[("≤2015", 2015), ("2016-2020", 2020), ("2021+", 2024)]
        )

        assert assign_bucket(2010, config) == "≤2015"
        assert assign_bucket(2015, config) == "≤2015"
        assert assign_bucket(2016, config) == "2016-2020"
        assert assign_bucket(2020, config) == "2016-2020"
        assert assign_bucket(2021, config) == "2021+"
        assert assign_bucket(2025, config) == "2021+"


class TestSplitting:
    """Tests for dataset splitting logic."""

    def test_create_splits_basic(self):
        """Test basic split creation."""
        df = pd.DataFrame(
            {
                "paper_id": [f"p{i}" for i in range(100)],
                "title": [f"Title {i}" for i in range(100)],
                "abstract": [f"Abstract {i}" for i in range(100)],
                "year": [2017 if i < 50 else 2021 for i in range(100)],
            }
        )

        config = BucketConfig(
            boundaries=[("≤2018", 2018), ("2019-2024", 2024)],
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            seed=42,
        )

        splits = create_splits(df, config)

        # Check structure
        assert "≤2018" in splits
        assert "2019-2024" in splits
        assert all(
            split in splits["≤2018"] for split in ["train", "val", "test"]
        )

        # Check counts
        bucket1_total = sum(len(splits["≤2018"][s]) for s in ["train", "val", "test"])
        bucket2_total = sum(
            len(splits["2019-2024"][s]) for s in ["train", "val", "test"]
        )
        assert bucket1_total == 50
        assert bucket2_total == 50

    def test_splits_no_leakage(self):
        """Test that splits have no overlapping paper_ids."""
        df = pd.DataFrame(
            {
                "paper_id": [f"p{i}" for i in range(100)],
                "title": [f"Title {i}" for i in range(100)],
                "abstract": [f"Abstract {i}" for i in range(100)],
                "year": [2020] * 100,  # All in same bucket
            }
        )

        config = BucketConfig(boundaries=[("2019-2024", 2024)], seed=42)
        splits = create_splits(df, config)

        bucket_splits = splits["2019-2024"]
        train_ids = set(bucket_splits["train"]["paper_id"])
        val_ids = set(bucket_splits["val"]["paper_id"])
        test_ids = set(bucket_splits["test"]["paper_id"])

        # Check no overlap
        assert len(train_ids & val_ids) == 0, "Train/val overlap detected"
        assert len(train_ids & test_ids) == 0, "Train/test overlap detected"
        assert len(val_ids & test_ids) == 0, "Val/test overlap detected"

        # Check all IDs accounted for
        assert len(train_ids | val_ids | test_ids) == 100

    def test_splits_correct_ratios(self):
        """Test that splits match configured ratios."""
        df = pd.DataFrame(
            {
                "paper_id": [f"p{i}" for i in range(1000)],
                "title": [f"Title {i}" for i in range(1000)],
                "abstract": [f"Abstract {i}" for i in range(1000)],
                "year": [2020] * 1000,
            }
        )

        config = BucketConfig(
            boundaries=[("2019-2024", 2024)],
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            seed=42,
        )

        splits = create_splits(df, config)
        bucket_splits = splits["2019-2024"]

        n_train = len(bucket_splits["train"])
        n_val = len(bucket_splits["val"])
        n_test = len(bucket_splits["test"])
        total = n_train + n_val + n_test

        # Check ratios (with small tolerance for rounding)
        assert abs(n_train / total - 0.7) < 0.02
        assert abs(n_val / total - 0.1) < 0.02
        assert abs(n_test / total - 0.2) < 0.02

    def test_max_per_bucket_cap(self):
        """Test that max_per_bucket cap is applied correctly."""
        df = pd.DataFrame(
            {
                "paper_id": [f"p{i}" for i in range(1000)],
                "title": [f"Title {i}" for i in range(1000)],
                "abstract": [f"Abstract {i}" for i in range(1000)],
                "year": [2020] * 1000,
            }
        )

        config = BucketConfig(
            boundaries=[("2019-2024", 2024)], max_per_bucket=100, seed=42
        )

        splits = create_splits(df, config)
        bucket_splits = splits["2019-2024"]

        total = sum(len(bucket_splits[s]) for s in ["train", "val", "test"])
        assert total == 100, f"Expected 100 total samples, got {total}"

    def test_splits_deterministic(self):
        """Test that splits are deterministic with same seed."""
        df = pd.DataFrame(
            {
                "paper_id": [f"p{i}" for i in range(100)],
                "title": [f"Title {i}" for i in range(100)],
                "abstract": [f"Abstract {i}" for i in range(100)],
                "year": [2020] * 100,
            }
        )

        config = BucketConfig(boundaries=[("2019-2024", 2024)], seed=42)

        splits1 = create_splits(df, config)
        splits2 = create_splits(df, config)

        # Compare train sets
        train1_ids = set(splits1["2019-2024"]["train"]["paper_id"])
        train2_ids = set(splits2["2019-2024"]["train"]["paper_id"])
        assert train1_ids == train2_ids, "Splits not deterministic"


class TestPairs:
    """Tests for positive pair generation."""

    def test_create_pairs_basic(self):
        """Test basic pair creation."""
        df = pd.DataFrame(
            {
                "paper_id": ["p1", "p2", "p3"],
                "title": ["Title 1", "Title 2", "Title 3"],
                "abstract": ["Abstract 1", "Abstract 2", "Abstract 3"],
                "year": [2020, 2021, 2022],
            }
        )

        pairs = create_pairs(df)
        assert len(pairs) == 3
        assert pairs[0] == ("p1", "Title 1", "Abstract 1")
        assert pairs[1] == ("p2", "Title 2", "Abstract 2")
        assert pairs[2] == ("p3", "Title 3", "Abstract 3")

    def test_create_pairs_count_matches_papers(self):
        """Test that pair count equals number of papers."""
        df = pd.DataFrame(
            {
                "paper_id": [f"p{i}" for i in range(50)],
                "title": [f"Title {i}" for i in range(50)],
                "abstract": [f"Abstract {i}" for i in range(50)],
                "year": [2020] * 50,
            }
        )

        pairs = create_pairs(df)
        assert len(pairs) == 50

    def test_create_pairs_missing_columns(self):
        """Test error handling for missing columns."""
        df = pd.DataFrame(
            {
                "paper_id": ["p1", "p2"],
                "title": ["Title 1", "Title 2"],
                # Missing 'abstract'
            }
        )

        with pytest.raises(ValueError) as exc_info:
            create_pairs(df)
        assert "missing required columns" in str(exc_info.value).lower()

    def test_create_pairs_handles_nulls(self):
        """Test that pairs with null title/abstract are skipped."""
        df = pd.DataFrame(
            {
                "paper_id": ["p1", "p2", "p3"],
                "title": ["Title 1", None, "Title 3"],
                "abstract": ["Abstract 1", "Abstract 2", None],
                "year": [2020, 2021, 2022],
            }
        )

        pairs = create_pairs(df)
        # Only p1 should remain (p2 has null title, p3 has null abstract)
        assert len(pairs) == 1
        assert pairs[0][0] == "p1"


class TestIntegration:
    """Integration tests for the full data pipeline."""

    def test_end_to_end_pipeline(self, tmp_path):
        """Test complete pipeline from CSV to splits."""
        # Create test CSV
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame(
            {
                "paper_id": [f"p{i}" for i in range(200)],
                "title": [f"Title {i}" for i in range(200)],
                "abstract": [f"Abstract {i}" * 10 for i in range(200)],  # Long enough
                "year": [2017 if i < 100 else 2021 for i in range(200)],
            }
        )
        df.to_csv(csv_path, index=False)

        # Load dataset
        loaded_df = load_dataset_from_csv(csv_path)
        assert len(loaded_df) == 200

        # Create splits
        config = BucketConfig(
            boundaries=[("≤2018", 2018), ("2019-2024", 2024)], seed=42
        )
        splits = create_splits(loaded_df, config)

        # Verify structure
        assert len(splits) == 2
        for bucket_name, bucket_splits in splits.items():
            assert "train" in bucket_splits
            assert "val" in bucket_splits
            assert "test" in bucket_splits

            # Create pairs for each split
            for split_name, split_df in bucket_splits.items():
                pairs = create_pairs(split_df)
                assert len(pairs) == len(split_df)

        # Verify no cross-bucket leakage
        bucket1_ids = set()
        bucket2_ids = set()
        for split_name in ["train", "val", "test"]:
            bucket1_ids.update(splits["≤2018"][split_name]["paper_id"])
            bucket2_ids.update(splits["2019-2024"][split_name]["paper_id"])

        assert len(bucket1_ids & bucket2_ids) == 0, "Cross-bucket leakage detected"
