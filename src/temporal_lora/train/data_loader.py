"""Data loading utilities for training."""

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sentence_transformers import InputExample

from ..utils.logging import get_logger

logger = get_logger(__name__)


def load_training_pairs(
    parquet_path: Path,
) -> List[InputExample]:
    """Load training pairs from parquet file.
    
    Args:
        parquet_path: Path to parquet file with text_a, text_b columns.
        
    Returns:
        List of InputExample objects for sentence-transformers.
    """
    logger.info(f"Loading training pairs from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    examples = []
    for _, row in df.iterrows():
        example = InputExample(texts=[str(row["text_a"]), str(row["text_b"])])
        examples.append(example)
    
    logger.info(f"Loaded {len(examples)} training pairs")
    return examples


def load_bucket_data(
    bucket_dir: Path,
) -> Tuple[List[InputExample], List[InputExample], List[InputExample]]:
    """Load train/val/test data for a bucket.
    
    Args:
        bucket_dir: Directory containing train.parquet, val.parquet, test.parquet.
        
    Returns:
        Tuple of (train_examples, val_examples, test_examples).
    """
    train_path = bucket_dir / "train.parquet"
    val_path = bucket_dir / "val.parquet"
    test_path = bucket_dir / "test.parquet"
    
    train_examples = load_training_pairs(train_path) if train_path.exists() else []
    val_examples = load_training_pairs(val_path) if val_path.exists() else []
    test_examples = load_training_pairs(test_path) if test_path.exists() else []
    
    logger.info(
        f"Bucket data loaded: train={len(train_examples)}, "
        f"val={len(val_examples)}, test={len(test_examples)}"
    )
    
    return train_examples, val_examples, test_examples


def create_cross_period_negatives(
    bucket_dirs: List[Path], current_bucket: str, num_negatives: int = 100
) -> List[InputExample]:
    """Create hard negatives from other time periods.
    
    Args:
        bucket_dirs: List of all bucket directories.
        current_bucket: Name of current bucket to exclude.
        num_negatives: Number of negative examples to sample.
        
    Returns:
        List of negative examples from other periods.
    """
    negatives = []
    
    for bucket_dir in bucket_dirs:
        if bucket_dir.name == current_bucket:
            continue
        
        train_path = bucket_dir / "train.parquet"
        if not train_path.exists():
            continue
        
        # Sample some examples from this period
        df = pd.read_parquet(train_path)
        sample_size = min(num_negatives // (len(bucket_dirs) - 1), len(df))
        
        if sample_size > 0:
            sampled = df.sample(n=sample_size, random_state=42)
            for _, row in sampled.iterrows():
                example = InputExample(texts=[str(row["text_a"]), str(row["text_b"])])
                negatives.append(example)
    
    logger.info(f"Created {len(negatives)} cross-period negatives")
    return negatives
