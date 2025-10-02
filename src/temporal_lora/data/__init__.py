"""Data loading and preparation modules."""

from .datasets import load_hf_or_csv, load_from_csv, load_from_hf, validate_schema
from .bucketing import assign_buckets, stratified_split, bucket_and_split
from .pairs import create_positive_pairs, save_pairs_by_bucket_and_split
from .pipeline import run_data_pipeline, generate_report

__all__ = [
    # datasets
    "load_hf_or_csv",
    "load_from_csv",
    "load_from_hf",
    "validate_schema",
    # bucketing
    "assign_buckets",
    "stratified_split",
    "bucket_and_split",
    # pairs
    "create_positive_pairs",
    "save_pairs_by_bucket_and_split",
    # pipeline
    "run_data_pipeline",
    "generate_report",
]
