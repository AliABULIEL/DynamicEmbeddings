"""Bucketing and splitting utilities for temporal data."""

from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

from ..utils.logging import get_logger
from ..utils.seeding import get_rng

logger = get_logger(__name__)


def assign_buckets(df: pd.DataFrame, bucket_config: List[Dict[str, Any]]) -> pd.DataFrame:
    """Assign time buckets to data based on year.
    
    Args:
        df: DataFrame with 'year' column.
        bucket_config: List of bucket definitions from config.
        
    Returns:
        DataFrame with added 'bucket' column.
        
    Raises:
        ValueError: If year column is missing or invalid.
    """
    if "year" not in df.columns:
        raise ValueError("DataFrame must have 'year' column")
    
    df = df.copy()
    df["bucket"] = None
    
    for bucket_def in bucket_config:
        name = bucket_def["name"]
        year_range = bucket_def["range"]
        
        # Handle None values in range (e.g., [None, 2018] means ≤2018)
        start_year = year_range[0] if year_range[0] is not None else -np.inf
        end_year = year_range[1] if year_range[1] is not None else np.inf
        
        # Assign bucket
        mask = (df["year"] >= start_year) & (df["year"] <= end_year)
        df.loc[mask, "bucket"] = name
        
        logger.info(f"Bucket '{name}': {mask.sum()} samples (years {start_year}-{end_year})")
    
    # Remove rows without bucket assignment
    unassigned = df["bucket"].isna().sum()
    if unassigned > 0:
        logger.warning(f"Dropping {unassigned} rows with years outside bucket ranges")
        df = df[df["bucket"].notna()]
    
    return df


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test with stratification by year.
    
    Args:
        df: DataFrame to split.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    rng = get_rng(seed)
    
    # Shuffle and split
    indices = np.arange(len(df))
    rng.shuffle(indices)
    
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]
    
    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    return train_df, val_df, test_df


def bucket_and_split(
    df: pd.DataFrame,
    bucket_config: List[Dict[str, Any]],
    max_per_bucket: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign buckets and create splits with caps.
    
    Args:
        df: Raw DataFrame with year column.
        bucket_config: Bucket definitions from config.
        max_per_bucket: Maximum samples per bucket.
        seed: Random seed.
        
    Returns:
        DataFrame with bucket and split assignments.
    """
    # Assign buckets
    df = assign_buckets(df, bucket_config)
    
    # Process each bucket separately to ensure no ID leakage
    all_splits = []
    rng = get_rng(seed)
    
    for bucket_name in df["bucket"].unique():
        bucket_df = df[df["bucket"] == bucket_name].copy()
        
        # Cap samples if needed
        if len(bucket_df) > max_per_bucket:
            logger.info(
                f"Bucket '{bucket_name}': Capping from {len(bucket_df)} to {max_per_bucket}"
            )
            # Shuffle and sample
            sample_indices = rng.choice(len(bucket_df), size=max_per_bucket, replace=False)
            bucket_df = bucket_df.iloc[sample_indices]
        
        # Split within bucket (70/10/20)
        train_df, val_df, test_df = stratified_split(
            bucket_df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=seed
        )
        
        all_splits.extend([train_df, val_df, test_df])
        
        logger.info(
            f"Bucket '{bucket_name}': train={len(train_df)}, "
            f"val={len(val_df)}, test={len(test_df)}"
        )
    
    # Combine all splits
    result = pd.concat(all_splits, ignore_index=True)
    
    # Verify no ID leakage across splits
    for split_name in ["train", "val", "test"]:
        split_ids = set(result[result["split"] == split_name]["paper_id"])
        other_ids = set(result[result["split"] != split_name]["paper_id"])
        overlap = split_ids & other_ids
        if overlap:
            raise ValueError(f"ID leakage detected in {split_name} split: {len(overlap)} IDs")
    
    logger.info(f"Total samples after bucketing and splitting: {len(result)}")
    return result
