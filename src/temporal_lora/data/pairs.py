"""Pair generation for contrastive learning."""

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_positive_pairs(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Create positive pairs (title, abstract) for each paper.
    
    Args:
        df: DataFrame with paper_id, title, abstract, year, bucket, split columns.
        config: Data config with preprocessing settings.
        
    Returns:
        DataFrame with columns: paper_id, text_a, text_b, year, bucket, split.
    """
    # Apply preprocessing filters
    preprocessing = config.get("preprocessing", {})
    min_len = preprocessing.get("min_abstract_length", 50)
    max_len = preprocessing.get("max_abstract_length", 2000)
    remove_dupes = preprocessing.get("remove_duplicates", True)
    
    df = df.copy()
    
    # Filter by abstract length
    initial_count = len(df)
    df["abstract_len"] = df["abstract"].str.len()
    df = df[(df["abstract_len"] >= min_len) & (df["abstract_len"] <= max_len)]
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        logger.info(f"Filtered {filtered_count} papers by abstract length ({min_len}-{max_len})")
    
    # Remove duplicates based on abstract
    if remove_dupes:
        initial_count = len(df)
        df = df.drop_duplicates(subset=["abstract"], keep="first")
        dupe_count = initial_count - len(df)
        if dupe_count > 0:
            logger.info(f"Removed {dupe_count} duplicate abstracts")
    
    # Create pairs: (title, abstract)
    pairs_df = pd.DataFrame(
        {
            "paper_id": df["paper_id"],
            "text_a": df["title"],
            "text_b": df["abstract"],
            "year": df["year"],
            "bucket": df["bucket"],
            "split": df["split"],
        }
    )
    
    logger.info(f"Created {len(pairs_df)} positive pairs")
    return pairs_df


def save_pairs_by_bucket_and_split(
    pairs_df: pd.DataFrame, output_dir: Path
) -> Dict[str, Dict[str, Path]]:
    """Save pairs to parquet files organized by bucket and split.
    
    Args:
        pairs_df: DataFrame with pairs and metadata.
        output_dir: Base output directory (e.g., data/processed).
        
    Returns:
        Dictionary mapping bucket -> split -> file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_map = {}
    
    for bucket_name in pairs_df["bucket"].unique():
        bucket_df = pairs_df[pairs_df["bucket"] == bucket_name]
        bucket_dir = output_dir / bucket_name
        bucket_dir.mkdir(parents=True, exist_ok=True)
        
        file_map[bucket_name] = {}
        
        for split_name in ["train", "val", "test"]:
            split_df = bucket_df[bucket_df["split"] == split_name]
            
            if len(split_df) > 0:
                output_path = bucket_dir / f"{split_name}.parquet"
                split_df.to_parquet(output_path, index=False)
                file_map[bucket_name][split_name] = output_path
                logger.info(
                    f"Saved {len(split_df)} pairs to {output_path.relative_to(output_dir.parent)}"
                )
            else:
                logger.warning(f"No data for bucket '{bucket_name}', split '{split_name}'")
    
    return file_map
