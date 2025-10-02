"""Data loading and preparation modules."""

from .datasets import load_hf_or_csv, load_from_csv, load_from_hf, validate_schema
from .preprocessing import (
    clean_and_preprocess,
    clean_html,
    collapse_whitespace,
    truncate_tokens,
    compute_content_hash,
)
from .bucketing import (
    assign_buckets,
    stratified_split,
    bucket_and_split,
    parse_bucket_spec,
    compute_balanced_sample_size,
)
from .pairs import create_positive_pairs, save_pairs_by_bucket_and_split
from .pipeline import run_data_pipeline, generate_report

__all__ = [
    # datasets
    "load_hf_or_csv",
    "load_from_csv",
    "load_from_hf",
    "validate_schema",
    # preprocessing
    "clean_and_preprocess",
    "clean_html",
    "collapse_whitespace",
    "truncate_tokens",
    "compute_content_hash",
    # bucketing
    "assign_buckets",
    "stratified_split",
    "bucket_and_split",
    "parse_bucket_spec",
    "compute_balanced_sample_size",
    # pairs
    "create_positive_pairs",
    "save_pairs_by_bucket_and_split",
    # pipeline
    "run_data_pipeline",
    "generate_report",
]
