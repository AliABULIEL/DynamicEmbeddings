"""Unit tests for data loading and processing."""

import pytest
import torch
from transformers import AutoTokenizer

from src.tide_lite.data.collate import TextBatcher, STSBCollator
from src.tide_lite.data.datasets import DatasetConfig, create_temporal_timestamps


def test_text_batcher():
    """Test text tokenization and batching."""
    tokenizer = TextBatcher(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_length=128
    )
    
    texts = ["Hello world", "This is a test"]
    batch = tokenizer(texts)
    
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] <= 128


def test_temporal_timestamps():
    """Test timestamp generation."""
    n_samples = 100
    timestamps = create_temporal_timestamps(
        n_samples,
        start_date="2020-01-01",
        end_date="2024-01-01",
        noise_std_days=7.0
    )
    
    assert len(timestamps) == n_samples
    assert all(t > 0 for t in timestamps), "Timestamps should be positive"
    
    # Check range (Unix timestamps for 2020-2024)
    min_ts = 1577836800  # 2020-01-01
    max_ts = 1704067200  # 2024-01-01
    
    assert all(min_ts <= t <= max_ts * 1.1 for t in timestamps), \
        "Timestamps out of expected range"


def test_stsb_collator():
    """Test STS-B batch collation."""
    tokenizer = TextBatcher(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_length=128
    )
    
    collator = STSBCollator(tokenizer, include_timestamps=True)
    
    # Mock batch
    batch = [
        {
            "sentence1": "First sentence",
            "sentence2": "Second sentence",
            "label": 3.5,
            "timestamp1": 1609459200.0,  # 2021-01-01
            "timestamp2": 1609459200.0,
        },
        {
            "sentence1": "Another first",
            "sentence2": "Another second",
            "label": 4.0,
            "timestamp1": 1640995200.0,  # 2022-01-01
            "timestamp2": 1640995200.0,
        }
    ]
    
    collated = collator(batch)
    
    assert "sentence1_inputs" in collated
    assert "sentence2_inputs" in collated
    assert "labels" in collated
    assert "timestamps1" in collated
    assert "timestamps2" in collated
    
    assert collated["labels"].shape == (2,)
    assert collated["timestamps1"].shape == (2,)


def test_dataset_config():
    """Test dataset configuration."""
    config = DatasetConfig(
        seed=42,
        cache_dir="./test_cache",
        timestamp_start="2020-01-01",
        timestamp_end="2024-01-01",
        temporal_noise_std=7.0
    )
    
    assert config.seed == 42
    assert config.cache_dir == "./test_cache"
    assert config.temporal_noise_std == 7.0
