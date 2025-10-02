"""Time bucketing and dataset splitting utilities."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class BucketConfig:
    """
    Configuration for time-based bucketing.

    Attributes:
        boundaries: List of (bucket_name, max_year) tuples defining bucket ranges.
                   Last bucket extends to infinity.
                   Example: [("≤2018", 2018), ("2019-2024", 2024)]
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.1)
        test_ratio: Proportion of data for testing (default: 0.2)
        max_per_bucket: Optional cap on samples per bucket (applied after bucketing)
        seed: Random seed for reproducible splits
    """

    boundaries: List[Tuple[str, int]]
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    max_per_bucket: Optional[int] = None
    seed: int = 42

    def __post_init__(self):
        """Validate bucket configuration."""
        if not self.boundaries:
            raise ValueError("boundaries must contain at least one bucket")

        # Check ratios sum to 1.0
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio:.4f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )

        # Validate boundaries are sorted
        years = [max_year for _, max_year in self.boundaries]
        if years != sorted(years):
            raise ValueError(
                f"Bucket boundaries must be sorted by year. Got: {years}"
            )

        # Check for duplicate years
        if len(years) != len(set(years)):
            raise ValueError(f"Duplicate year boundaries found: {years}")


def assign_bucket(year: int, config: BucketConfig) -> str:
    """
    Assign a year to its corresponding bucket.

    Args:
        year: Year value to assign
        config: Bucket configuration

    Returns:
        Bucket name (e.g., "≤2018", "2019-2024")
    """
    for bucket_name, max_year in config.boundaries:
        if year <= max_year:
            return bucket_name

    # If year exceeds all boundaries, assign to last bucket
    return config.boundaries[-1][0]


def create_splits(
    df: pd.DataFrame, config: BucketConfig
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Create train/val/test splits per bucket with no leakage.

    Args:
        df: Input DataFrame with 'year' column
        config: Bucket configuration

    Returns:
        Nested dict: {bucket_name: {"train": df, "val": df, "test": df}}

    Raises:
        ValueError: If any bucket is too small to split or has duplicate paper_ids
    """
    # Assign buckets
    df = df.copy()
    df["bucket"] = df["year"].apply(lambda y: assign_bucket(y, config))

    results = {}

    for bucket_name, bucket_df in df.groupby("bucket"):
        # Check for duplicates within bucket
        if bucket_df["paper_id"].duplicated().any():
            n_dups = bucket_df["paper_id"].duplicated().sum()
            raise ValueError(
                f"Bucket '{bucket_name}' contains {n_dups} duplicate paper_ids"
            )

        # Apply max_per_bucket cap if specified
        if config.max_per_bucket and len(bucket_df) > config.max_per_bucket:
            bucket_df = bucket_df.sample(
                n=config.max_per_bucket, random_state=config.seed
            )

        # Check minimum size for splits
        min_samples_needed = 3  # At least 1 per split
        if len(bucket_df) < min_samples_needed:
            raise ValueError(
                f"Bucket '{bucket_name}' has only {len(bucket_df)} samples, "
                f"need at least {min_samples_needed} for train/val/test splits"
            )

        # Shuffle and split
        bucket_df = bucket_df.sample(frac=1, random_state=config.seed).reset_index(
            drop=True
        )

        n_train = int(len(bucket_df) * config.train_ratio)
        n_val = int(len(bucket_df) * config.val_ratio)

        train_df = bucket_df.iloc[:n_train]
        val_df = bucket_df.iloc[n_train : n_train + n_val]
        test_df = bucket_df.iloc[n_train + n_val :]

        # Verify no leakage
        train_ids = set(train_df["paper_id"])
        val_ids = set(val_df["paper_id"])
        test_ids = set(test_df["paper_id"])

        if train_ids & val_ids:
            raise ValueError(
                f"Bucket '{bucket_name}': train/val overlap detected! "
                f"Overlapping IDs: {train_ids & val_ids}"
            )
        if train_ids & test_ids:
            raise ValueError(
                f"Bucket '{bucket_name}': train/test overlap detected! "
                f"Overlapping IDs: {train_ids & test_ids}"
            )
        if val_ids & test_ids:
            raise ValueError(
                f"Bucket '{bucket_name}': val/test overlap detected! "
                f"Overlapping IDs: {val_ids & test_ids}"
            )

        # Sanity check: all samples accounted for
        assert (
            len(train_df) + len(val_df) + len(test_df) == len(bucket_df)
        ), f"Split size mismatch in bucket '{bucket_name}'"

        results[bucket_name] = {
            "train": train_df.drop(columns=["bucket"]),
            "val": val_df.drop(columns=["bucket"]),
            "test": test_df.drop(columns=["bucket"]),
        }

    return results


def get_split_summary(splits: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
    """
    Generate summary statistics for splits.

    Args:
        splits: Output from create_splits()

    Returns:
        Dict with counts per bucket and split
    """
    summary = {}

    for bucket_name, bucket_splits in splits.items():
        summary[bucket_name] = {
            split_name: len(split_df)
            for split_name, split_df in bucket_splits.items()
        }
        summary[bucket_name]["total"] = sum(
            len(split_df) for split_df in bucket_splits.values()
        )

    return summary
