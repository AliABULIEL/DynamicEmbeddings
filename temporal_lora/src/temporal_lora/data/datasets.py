"""Dataset loading utilities for HuggingFace and CSV sources."""

from pathlib import Path
from typing import Optional

import pandas as pd


class DatasetValidationError(Exception):
    """Raised when dataset schema validation fails."""

    pass


REQUIRED_COLUMNS = {"paper_id", "title", "abstract", "year"}


def load_dataset_from_hf(dataset_id: str, split: str = "train") -> pd.DataFrame:
    """
    Load dataset from HuggingFace Hub.

    Args:
        dataset_id: HuggingFace dataset identifier (e.g., "user/dataset-name")
        split: Dataset split to load (default: "train")

    Returns:
        DataFrame with columns: paper_id, title, abstract, year

    Raises:
        ImportError: If datasets library not available
        DatasetValidationError: If required columns are missing
        Exception: If dataset loading fails
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets library not installed. Install with: pip install datasets"
        ) from e

    try:
        dataset = load_dataset(dataset_id, split=split)
        df = dataset.to_pandas()
    except Exception as e:
        raise Exception(
            f"Failed to load dataset '{dataset_id}' from HuggingFace: {e}"
        ) from e

    _validate_schema(df, source=f"HF:{dataset_id}")
    return df


def load_dataset_from_csv(
    csv_path: str | Path, delimiter: str = ","
) -> pd.DataFrame:
    """
    Load dataset from CSV file with schema validation.

    Args:
        csv_path: Path to CSV file
        delimiter: CSV delimiter (default: ",")

    Returns:
        DataFrame with columns: paper_id, title, abstract, year

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        DatasetValidationError: If schema validation fails
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            f"Expected file at: {csv_path.absolute()}\n"
            f"Please ensure the CSV file exists and the path is correct."
        )

    try:
        df = pd.read_csv(csv_path, delimiter=delimiter)
    except Exception as e:
        raise DatasetValidationError(
            f"Failed to parse CSV file '{csv_path}': {e}\n"
            f"Ensure the file is valid CSV with delimiter '{delimiter}'."
        ) from e

    _validate_schema(df, source=str(csv_path))
    return df


def _validate_schema(df: pd.DataFrame, source: str) -> None:
    """
    Validate that DataFrame has required columns.

    Args:
        df: DataFrame to validate
        source: Source identifier for error messages

    Raises:
        DatasetValidationError: If required columns are missing or invalid
    """
    actual_columns = set(df.columns)
    missing_columns = REQUIRED_COLUMNS - actual_columns

    if missing_columns:
        raise DatasetValidationError(
            f"Dataset from '{source}' is missing required columns: {missing_columns}\n"
            f"Required columns: {sorted(REQUIRED_COLUMNS)}\n"
            f"Found columns: {sorted(actual_columns)}\n"
            f"Please ensure your dataset has all required columns."
        )

    # Validate year column is numeric
    if not pd.api.types.is_numeric_dtype(df["year"]):
        try:
            df["year"] = pd.to_numeric(df["year"])
        except Exception as e:
            raise DatasetValidationError(
                f"Column 'year' in '{source}' must be numeric. "
                f"Found dtype: {df['year'].dtype}\n{e}"
            ) from e

    # Check for null values in required columns
    null_counts = df[list(REQUIRED_COLUMNS)].isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]

    if not columns_with_nulls.empty:
        raise DatasetValidationError(
            f"Dataset from '{source}' contains null values in required columns:\n"
            f"{columns_with_nulls.to_dict()}\n"
            f"Please remove or fill null values before processing."
        )

    # Ensure paper_id uniqueness
    if df["paper_id"].duplicated().any():
        n_duplicates = df["paper_id"].duplicated().sum()
        raise DatasetValidationError(
            f"Dataset from '{source}' contains {n_duplicates} duplicate paper_id values.\n"
            f"Each paper must have a unique paper_id."
        )
