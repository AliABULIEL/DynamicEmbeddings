"""Smoke tests for data pipeline."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest


def create_synthetic_csv(path: Path) -> None:
    """Create a small synthetic dataset for testing."""
    data = {
        "paper_id": [f"paper_{i:03d}" for i in range(100)],
        "title": [f"Title about deep learning {i}" for i in range(100)],
        "abstract": [
            f"This is an abstract about neural networks and transformers. " * 5 + f"Paper {i}."
            for i in range(100)
        ],
        "year": [2017 + (i % 8) for i in range(100)],  # 2017-2024
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def test_load_synthetic_csv():
    """Test loading synthetic CSV data."""
    from temporal_lora.data.datasets import load_from_csv
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)
    
    try:
        create_synthetic_csv(csv_path)
        
        # Load with default config
        config = {
            "id_field": "paper_id",
            "title_field": "title",
            "text_field": "abstract",
            "year_field": "year",
        }
        df = load_from_csv(csv_path, config)
        
        assert len(df) == 100
        assert "paper_id" in df.columns
        assert "title" in df.columns
        assert "abstract" in df.columns
        assert "year" in df.columns
        assert df["year"].min() == 2017
        assert df["year"].max() == 2024
    finally:
        csv_path.unlink()


def test_bucket_and_split():
    """Test bucketing and splitting functionality."""
    from temporal_lora.data.bucketing import bucket_and_split
    
    # Create synthetic data
    data = {
        "paper_id": [f"paper_{i:03d}" for i in range(100)],
        "title": [f"Title {i}" for i in range(100)],
        "abstract": [f"Abstract text " * 10 + f"{i}" for i in range(100)],
        "year": [2017 + (i % 8) for i in range(100)],
    }
    df = pd.DataFrame(data)
    
    # Bucket config
    bucket_config = [
        {"name": "early", "range": [None, 2018]},
        {"name": "recent", "range": [2019, 2024]},
    ]
    
    result = bucket_and_split(df, bucket_config, max_per_bucket=100, seed=42)
    
    # Check buckets assigned
    assert "bucket" in result.columns
    assert set(result["bucket"].unique()) == {"early", "recent"}
    
    # Check splits created
    assert "split" in result.columns
    assert set(result["split"].unique()) == {"train", "val", "test"}
    
    # Check no ID leakage
    train_ids = set(result[result["split"] == "train"]["paper_id"])
    val_ids = set(result[result["split"] == "val"]["paper_id"])
    test_ids = set(result[result["split"] == "test"]["paper_id"])
    
    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0


def test_create_pairs():
    """Test positive pair creation."""
    from temporal_lora.data.pairs import create_positive_pairs
    
    # Create bucketed and split data
    data = {
        "paper_id": [f"paper_{i:03d}" for i in range(50)],
        "title": [f"Title {i}" for i in range(50)],
        "abstract": [f"Abstract text " * 10 + f"{i}" for i in range(50)],
        "year": [2020] * 50,
        "bucket": ["recent"] * 50,
        "split": ["train"] * 30 + ["val"] * 10 + ["test"] * 10,
    }
    df = pd.DataFrame(data)
    
    config = {
        "preprocessing": {
            "min_abstract_length": 50,
            "max_abstract_length": 2000,
            "remove_duplicates": True,
        }
    }
    
    pairs_df = create_positive_pairs(df, config)
    
    assert len(pairs_df) == 50
    assert "text_a" in pairs_df.columns
    assert "text_b" in pairs_df.columns
    assert all(pairs_df["text_a"] == df["title"])
    assert all(pairs_df["text_b"] == df["abstract"])


def test_full_pipeline_with_synthetic_data():
    """Test full pipeline with synthetic CSV."""
    from temporal_lora.data.pipeline import run_data_pipeline
    from temporal_lora.utils.seeding import set_seed
    
    set_seed(42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        csv_path = tmpdir / "test_data.csv"
        output_dir = tmpdir / "processed"
        
        # Create synthetic CSV
        create_synthetic_csv(csv_path)
        
        # Config
        config = {
            "dataset": {
                "name": str(csv_path),
                "text_field": "abstract",
                "title_field": "title",
                "id_field": "paper_id",
                "year_field": "year",
            },
            "buckets": [
                {"name": "early", "range": [None, 2018]},
                {"name": "recent", "range": [2019, 2024]},
            ],
            "sampling": {
                "max_per_bucket": 60,
                "seed": 42,
                "stratify": True,
            },
            "preprocessing": {
                "min_abstract_length": 50,
                "max_abstract_length": 2000,
                "remove_duplicates": True,
            },
        }
        
        # Run pipeline
        report = run_data_pipeline(config, output_dir=output_dir)
        
        # Check report
        assert "total_pairs" in report
        assert report["total_pairs"] > 0
        assert "buckets" in report
        assert "early" in report["buckets"]
        assert "recent" in report["buckets"]
        
        # Check files exist
        assert (output_dir / "report.json").exists()
        
        for bucket_name in ["early", "recent"]:
            bucket_dir = output_dir / bucket_name
            assert bucket_dir.exists()
            
            # At least train split should exist
            train_file = bucket_dir / "train.parquet"
            if train_file.exists():
                # Check parquet is non-empty
                df = pd.read_parquet(train_file)
                assert len(df) > 0
                assert "text_a" in df.columns
                assert "text_b" in df.columns
                assert "paper_id" in df.columns
                assert "year" in df.columns
                assert "bucket" in df.columns
                assert "split" in df.columns
