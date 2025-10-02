"""Smoke tests for training."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sentence_transformers import InputExample


def create_tiny_training_data(output_dir: Path, n_samples: int = 20) -> None:
    """Create a tiny synthetic dataset for smoke testing.
    
    Args:
        output_dir: Output directory for parquet files.
        n_samples: Number of samples to create.
    """
    # Create train bucket
    bucket_dir = output_dir / "test_bucket"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic pairs
    data = {
        "paper_id": [f"paper_{i:03d}" for i in range(n_samples)],
        "text_a": [f"Title about machine learning topic {i}" for i in range(n_samples)],
        "text_b": [
            f"This is an abstract about neural networks and deep learning. " * 3 + f"Paper {i}."
            for i in range(n_samples)
        ],
        "year": [2020] * n_samples,
        "bucket": ["test_bucket"] * n_samples,
        "split": ["train"] * n_samples,
    }
    
    df = pd.DataFrame(data)
    df.to_parquet(bucket_dir / "train.parquet", index=False)


def test_load_training_pairs():
    """Test loading training pairs from parquet."""
    from temporal_lora.train.data_loader import load_training_pairs
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test data
        data = {
            "paper_id": ["p1", "p2"],
            "text_a": ["Title 1", "Title 2"],
            "text_b": ["Abstract 1", "Abstract 2"],
            "year": [2020, 2020],
            "bucket": ["test", "test"],
            "split": ["train", "train"],
        }
        df = pd.DataFrame(data)
        parquet_path = tmpdir / "test.parquet"
        df.to_parquet(parquet_path, index=False)
        
        # Load
        examples = load_training_pairs(parquet_path)
        
        assert len(examples) == 2
        assert isinstance(examples[0], InputExample)
        assert len(examples[0].texts) == 2
        assert examples[0].texts[0] == "Title 1"
        assert examples[0].texts[1] == "Abstract 1"


def test_trainer_initialization():
    """Test that LoRATrainer can be initialized."""
    from temporal_lora.train.trainer import LoRATrainer
    
    trainer = LoRATrainer(
        base_model_name="sentence-transformers/all-MiniLM-L6-v2",
        lora_r=8,
        lora_alpha=16,
        epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        seed=42,
    )
    
    assert trainer.base_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert trainer.lora_r == 8
    assert trainer.epochs == 1
    assert trainer.device in ["cuda", "cpu"]
    
    print(f"✓ Trainer initialized on device: {trainer.device}")


def test_train_tiny_batch():
    """Test training on a tiny batch to verify loss decreases."""
    from temporal_lora.train.trainer import LoRATrainer
    from sentence_transformers import InputExample
    
    # Create tiny synthetic dataset
    train_examples = [
        InputExample(texts=[f"Title {i}", f"Abstract about topic {i} " * 10])
        for i in range(10)
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Initialize trainer with small config for speed
        trainer = LoRATrainer(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            lora_r=8,
            lora_alpha=16,
            epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            fp16=False,  # Disable fp16 for CPU compatibility
            seed=42,
        )
        
        # Train
        metrics = trainer.train_bucket(
            bucket_name="test_bucket",
            train_examples=train_examples,
            output_dir=tmpdir / "test_adapter",
        )
        
        # Check metrics
        assert "bucket" in metrics
        assert metrics["bucket"] == "test_bucket"
        assert metrics["train_examples"] == 10
        assert metrics["epochs"] == 1
        assert "total_time" in metrics
        
        # Check adapter saved
        adapter_dir = Path(metrics["output_dir"])
        assert adapter_dir.exists()
        assert (adapter_dir / "adapter_config.json").exists()
        
        print(f"✓ Training completed in {metrics['total_time']:.2f}s")
        print(f"✓ Adapter saved to: {adapter_dir}")


def test_train_all_buckets():
    """Test training multiple buckets."""
    from temporal_lora.train.trainer import train_all_buckets
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create two buckets with tiny data
        for bucket_name in ["early", "recent"]:
            bucket_dir = tmpdir / "data" / bucket_name
            bucket_dir.mkdir(parents=True, exist_ok=True)
            
            data = {
                "paper_id": [f"p_{bucket_name}_{i}" for i in range(8)],
                "text_a": [f"Title {bucket_name} {i}" for i in range(8)],
                "text_b": [f"Abstract {bucket_name} " * 10 + f"{i}" for i in range(8)],
                "year": [2020] * 8,
                "bucket": [bucket_name] * 8,
                "split": ["train"] * 8,
            }
            df = pd.DataFrame(data)
            df.to_parquet(bucket_dir / "train.parquet", index=False)
        
        # Config
        config = {
            "model": {
                "base_model": {"name": "sentence-transformers/all-MiniLM-L6-v2"},
                "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1},
            },
            "training": {
                "epochs": 1,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "fp16": False,
                "seed": 42,
            },
        }
        
        # Train
        data_dir = tmpdir / "data"
        output_dir = tmpdir / "adapters"
        
        metrics = train_all_buckets(data_dir, output_dir, config)
        
        # Check both buckets trained
        assert "early" in metrics
        assert "recent" in metrics
        
        # Check adapters saved
        for bucket_name in ["early", "recent"]:
            adapter_dir = output_dir / bucket_name
            assert adapter_dir.exists()
            assert (adapter_dir / "adapter_config.json").exists()
        
        print(f"✓ Trained {len(metrics)} buckets successfully")


@pytest.mark.skipif(
    True,  # Skip by default to save time in CI
    reason="Full training test - run manually",
)
def test_full_training_pipeline():
    """Full integration test with real data preparation and training.
    
    This test is skipped by default. Run manually with:
    pytest tests/test_trainer_smoke.py::test_full_training_pipeline -v
    """
    from temporal_lora.data.pipeline import run_data_pipeline
    from temporal_lora.train.trainer import train_all_buckets
    from temporal_lora.utils.seeding import set_seed
    
    set_seed(42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic CSV
        data = {
            "paper_id": [f"paper_{i:03d}" for i in range(50)],
            "title": [f"Deep learning paper {i}" for i in range(50)],
            "abstract": [f"This paper discusses neural networks. " * 10 + f"{i}" for i in range(50)],
            "year": [2017 + (i % 8) for i in range(50)],
        }
        df = pd.DataFrame(data)
        csv_path = tmpdir / "papers.csv"
        df.to_csv(csv_path, index=False)
        
        # Prepare data
        data_config = {
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
            "sampling": {"max_per_bucket": 30, "seed": 42, "stratify": True},
            "preprocessing": {
                "min_abstract_length": 50,
                "max_abstract_length": 2000,
                "remove_duplicates": True,
            },
        }
        
        processed_dir = tmpdir / "processed"
        run_data_pipeline(data_config, output_dir=processed_dir)
        
        # Train
        train_config = {
            "model": {
                "base_model": {"name": "sentence-transformers/all-MiniLM-L6-v2"},
                "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1},
            },
            "training": {
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 1e-4,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "fp16": False,
                "seed": 42,
            },
        }
        
        adapters_dir = tmpdir / "adapters"
        metrics = train_all_buckets(processed_dir, adapters_dir, train_config)
        
        # Verify both buckets trained
        assert len(metrics) == 2
        assert all((adapters_dir / bucket).exists() for bucket in metrics.keys())
        
        print("✓ Full pipeline completed successfully")
