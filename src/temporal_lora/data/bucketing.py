"""Bucketing and splitting utilities for temporal data."""

from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

from ..utils.logging import get_logger
from ..utils.seeding import get_rng

logger = get_logger(__name__)


def parse_bucket_spec(bucket_config: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    """Parse bucket specifications into (start_year, end_year) tuples.
    
    Args:
        bucket_config: List of bucket definitions from config.
        
    Returns:
        Dictionary mapping bucket name to (start_year, end_year) tuple.
    """
    bucket_ranges = {}
    
    for bucket_def in bucket_config:
        name = bucket_def["name"]
        year_range = bucket_def["range"]
        
        # Handle None values (open-ended ranges)
        start_year = year_range[0] if year_range[0] is not None else -np.inf
        end_year = year_range[1] if year_range[1] is not None else np.inf
        
        bucket_ranges[name] = (start_year, end_year)
    
    return bucket_ranges


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
    
    bucket_ranges = parse_bucket_spec(bucket_config)
    
    for name, (start_year, end_year) in bucket_ranges.items():
        # Assign bucket
        mask = (df["year"] >= start_year) & (df["year"] <= end_year)
        df.loc[mask, "bucket"] = name
        
        # Format year range for logging
        start_str = str(int(start_year)) if start_year != -np.inf else "≤"
        end_str = str(int(end_year)) if end_year != np.inf else "∞"
        year_range_str = f"{start_str}–{end_str}" if start_year != -np.inf else f"≤{end_str}"
        
        logger.info(f"Bucket '{name}': {mask.sum()} samples (years {year_range_str})")
    
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


def compute_balanced_sample_size(
    df: pd.DataFrame,
    max_per_bucket: int,
    balance_per_bin: bool,
) -> Dict[str, int]:
    """Compute sample size per bucket for balanced sampling.
    
    Args:
        df: DataFrame with bucket assignments.
        max_per_bucket: Maximum samples per bucket.
        balance_per_bin: If True, use minimum bucket size across all buckets.
        
    Returns:
        Dictionary mapping bucket name to target sample size.
    """
    bucket_counts = df["bucket"].value_counts().to_dict()
    
    if balance_per_bin:
        # Use minimum count across all buckets (capped by max_per_bucket)
        min_count = min(bucket_counts.values())
        target_size = min(min_count, max_per_bucket)
        
        logger.info(
            f"Balance mode enabled: using {target_size} samples per bucket "
            f"(min bucket size: {min_count}, max: {max_per_bucket})"
        )
        
        # All buckets get same size
        sample_sizes = {name: target_size for name in bucket_counts.keys()}
    else:
        # Use max_per_bucket cap per bucket independently
        sample_sizes = {
            name: min(count, max_per_bucket)
            for name, count in bucket_counts.items()
        }
        
        logger.info(
            f"Independent capping: using up to {max_per_bucket} samples per bucket"
        )
    
    return sample_sizes


def bucket_and_split(
    df: pd.DataFrame,
    bucket_config: List[Dict[str, Any]],
    max_per_bucket: int,
    balance_per_bin: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign buckets and create splits with caps and optional balancing.
    
    Args:
        df: Raw DataFrame with year column.
        bucket_config: Bucket definitions from config.
        max_per_bucket: Maximum samples per bucket.
        balance_per_bin: If True, enforce equal counts across buckets.
        seed: Random seed.
        
    Returns:
        DataFrame with bucket and split assignments.
    """
    # Assign buckets
    df = assign_buckets(df, bucket_config)
    
    # Compute sample sizes per bucket
    sample_sizes = compute_balanced_sample_size(df, max_per_bucket, balance_per_bin)
    
    # Process each bucket separately to ensure no ID leakage
    all_splits = []
    rng = get_rng(seed)
    
    for bucket_name in sorted(df["bucket"].unique()):
        bucket_df = df[df["bucket"] == bucket_name].copy()
        target_size = sample_sizes[bucket_name]
        
        # Sample if needed
        if len(bucket_df) > target_size:
            logger.info(
                f"Bucket '{bucket_name}': Sampling {target_size} from {len(bucket_df)}"
            )
            # Shuffle and sample
            sample_indices = rng.choice(len(bucket_df), size=target_size, replace=False)
            bucket_df = bucket_df.iloc[sample_indices]
        else:
            logger.info(
                f"Bucket '{bucket_name}': Using all {len(bucket_df)} samples"
            )
        
        # Split within bucket (70/10/20) to prevent ID leakage
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
    
    # Log final counts per bucket
    logger.info("\nFinal bucket distribution:")
    for bucket_name in sorted(result["bucket"].unique()):
        bucket_count = len(result[result["bucket"] == bucket_name])
        logger.info(f"  {bucket_name}: {bucket_count} samples")
    
    return result
