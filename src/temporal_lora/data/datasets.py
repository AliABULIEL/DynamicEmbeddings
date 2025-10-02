"""Dataset loading utilities for HuggingFace and CSV sources."""

from pathlib import Path
from typing import Dict, Any, Union

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["paper_id", "title", "abstract", "year"]


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate.
        
    Raises:
        ValueError: If required columns are missing.
    """
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Required: {REQUIRED_COLUMNS}. Found: {list(df.columns)}"
        )


def load_from_csv(csv_path: Path, config: Dict[str, Any]) -> pd.DataFrame:
    """Load dataset from CSV file with schema validation.
    
    Args:
        csv_path: Path to CSV file.
        config: Data config with field mappings.
        
    Returns:
        DataFrame with required columns.
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist.
        ValueError: If schema validation fails.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. "
            f"Please provide a CSV with columns: {REQUIRED_COLUMNS}"
        )
    
    logger.info(f"Loading dataset from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Map columns if field mappings provided
    field_mapping = {
        config.get("id_field", "paper_id"): "paper_id",
        config.get("title_field", "title"): "title",
        config.get("text_field", "abstract"): "abstract",
        config.get("year_field", "year"): "year",
    }
    
    # Rename columns if needed
    rename_map = {k: v for k, v in field_mapping.items() if k in df.columns and k != v}
    if rename_map:
        df = df.rename(columns=rename_map)
    
    validate_schema(df)
    
    logger.info(f"Loaded {len(df)} rows from CSV")
    return df


def load_from_hf(dataset_name: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Load dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset identifier.
        config: Data config with field mappings.
        
    Returns:
        DataFrame with required columns.
        
    Raises:
        ImportError: If datasets library not available.
        ValueError: If dataset loading or schema validation fails.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required. Install with: pip install datasets"
        )
    
    logger.info(f"Loading dataset from HuggingFace: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name)
        # Use 'train' split or first available split
        split_name = "train" if "train" in dataset else list(dataset.keys())[0]
        df = dataset[split_name].to_pandas()
        
        # Map columns if field mappings provided
        field_mapping = {
            config.get("id_field", "paper_id"): "paper_id",
            config.get("title_field", "title"): "title",
            config.get("text_field", "abstract"): "abstract",
            config.get("year_field", "year"): "year",
        }
        
        # Rename columns if needed
        rename_map = {k: v for k, v in field_mapping.items() if k in df.columns and k != v}
        if rename_map:
            df = df.rename(columns=rename_map)
        
        validate_schema(df)
        
        logger.info(f"Loaded {len(df)} rows from HuggingFace")
        return df
        
    except Exception as e:
        raise ValueError(f"Failed to load HuggingFace dataset '{dataset_name}': {e}")


def load_hf_or_csv(config: Dict[str, Any]) -> pd.DataFrame:
    """Load dataset from HuggingFace or CSV fallback.
    
    Tries HuggingFace first, then falls back to CSV if HF fails.
    
    Args:
        config: Data config dictionary.
        
    Returns:
        DataFrame with required columns.
        
    Raises:
        ValueError: If both HF and CSV loading fail.
    """
    dataset_name = config["dataset"]["name"]
    
    # Try HuggingFace first if name doesn't look like a file path
    if not dataset_name.endswith(".csv"):
        try:
            return load_from_hf(dataset_name, config["dataset"])
        except Exception as e:
            logger.warning(f"HuggingFace loading failed: {e}")
            logger.info("Falling back to CSV loading...")
    
    # Fall back to CSV
    csv_path = Path(dataset_name)
    if not csv_path.is_absolute():
        # Try relative to data/raw/
        from ..utils.paths import DATA_RAW_DIR
        csv_path = DATA_RAW_DIR / dataset_name
    
    try:
        return load_from_csv(csv_path, config["dataset"])
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset from both HuggingFace and CSV. "
            f"Last error: {e}. "
            f"Please provide either a valid HuggingFace dataset name or a CSV file "
            f"with columns: {REQUIRED_COLUMNS}"
        )
