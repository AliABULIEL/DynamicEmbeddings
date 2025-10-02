"""Data pipeline orchestration and reporting."""

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .datasets import load_hf_or_csv
from .preprocessing import clean_and_preprocess
from .bucketing import bucket_and_split
from .pairs import create_positive_pairs, save_pairs_by_bucket_and_split
from ..utils.logging import get_logger
from ..utils.paths import DATA_PROCESSED_DIR
from ..utils.seeding import set_seed

logger = get_logger(__name__)


def generate_report(
    pairs_df: pd.DataFrame, file_map: Dict[str, Dict[str, Path]]
) -> Dict[str, Any]:
    """Generate audit report with counts, histograms, and samples.
    
    Args:
        pairs_df: Full pairs DataFrame.
        file_map: Mapping of bucket -> split -> file path.
        
    Returns:
        Report dictionary.
    """
    report = {
        "total_pairs": len(pairs_df),
        "buckets": {},
        "year_histogram": {},
        "sample_rows": [],
    }
    
    # Counts per bucket and split
    for bucket_name in pairs_df["bucket"].unique():
        bucket_df = pairs_df[pairs_df["bucket"] == bucket_name]
        report["buckets"][bucket_name] = {
            "total": len(bucket_df),
            "splits": {},
        }
        
        for split_name in ["train", "val", "test"]:
            split_df = bucket_df[bucket_df["split"] == split_name]
            report["buckets"][bucket_name]["splits"][split_name] = len(split_df)
    
    # Year histogram (overall)
    year_counts = pairs_df["year"].value_counts().sort_index()
    report["year_histogram"] = {int(year): int(count) for year, count in year_counts.items()}
    
    # Sample rows (5 random samples)
    sample_df = pairs_df.sample(min(5, len(pairs_df)), random_state=42)
    report["sample_rows"] = sample_df.to_dict(orient="records")
    
    # File paths
    report["output_files"] = {}
    for bucket_name, splits in file_map.items():
        report["output_files"][bucket_name] = {
            split_name: str(path) for split_name, path in splits.items()
        }
    
    return report


def run_data_pipeline(
    config: Dict[str, Any], output_dir: Path = DATA_PROCESSED_DIR
) -> Dict[str, Any]:
    """Run full data preparation pipeline.
    
    Args:
        config: Full data configuration.
        output_dir: Output directory for processed data.
        
    Returns:
        Report dictionary with statistics and file paths.
    """
    # Set seed for reproducibility
    seed = config["sampling"]["seed"]
    set_seed(seed)
    logger.info(f"Using random seed: {seed}")
    
    # Load data
    logger.info("Loading dataset...")
    df = load_hf_or_csv(config)
    logger.info(f"Loaded {len(df)} papers")
    
    # Clean and preprocess
    logger.info("Cleaning and preprocessing...")
    preprocessing_config = config.get("preprocessing", {})
    df = clean_and_preprocess(df, preprocessing_config)
    logger.info(f"After preprocessing: {len(df)} papers")
    
    # Bucket and split
    logger.info("Bucketing and splitting...")
    max_per_bucket = config["sampling"]["max_per_bucket"]
    balance_per_bin = config["sampling"].get("balance_per_bin", False)
    bucket_config = config["buckets"]
    
    bucketed_df = bucket_and_split(
        df, bucket_config, max_per_bucket, balance_per_bin=balance_per_bin, seed=seed
    )
    logger.info(f"After bucketing: {len(bucketed_df)} papers")
    
    # Create positive pairs
    logger.info("Creating positive pairs...")
    pairs_df = create_positive_pairs(bucketed_df, config)
    logger.info(f"Created {len(pairs_df)} pairs")
    
    # Save pairs by bucket and split
    logger.info("Saving pairs to parquet files...")
    file_map = save_pairs_by_bucket_and_split(pairs_df, output_dir)
    
    # Generate report
    logger.info("Generating report...")
    report = generate_report(pairs_df, file_map)
    
    # Save report
    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Total pairs: {report['total_pairs']}")
    print("\nBuckets:")
    for bucket_name, bucket_info in report["buckets"].items():
        print(f"  {bucket_name}:")
        print(f"    Total: {bucket_info['total']}")
        for split_name, count in bucket_info["splits"].items():
            print(f"    {split_name}: {count}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Report: {report_path}")
    print("=" * 60 + "\n")
    
    return report
