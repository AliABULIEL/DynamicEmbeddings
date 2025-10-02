"""
I/O utilities for loading/saving data, models, and results.

Provides consistent interfaces for JSON, YAML, CSV, and pickle files.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import yaml


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Dictionary of loaded data
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.

    Args:
        data: Dictionary to save
        path: Path to output JSON file
        indent: Indentation level for pretty printing
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary of loaded data
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save data to YAML file.

    Args:
        data: Dictionary to save
        path: Path to output YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)


def load_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load CSV file as pandas DataFrame.

    Args:
        path: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        Pandas DataFrame
    """
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """
    Save pandas DataFrame to CSV file.

    Args:
        df: DataFrame to save
        path: Path to output CSV file
        **kwargs: Additional arguments for df.to_csv
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


def load_pickle(path: Union[str, Path]) -> Any:
    """
    Load pickle file.

    Args:
        path: Path to pickle file

    Returns:
        Unpickled object
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to pickle
        path: Path to output pickle file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_text(text: str, path: Union[str, Path]) -> None:
    """
    Save text to file.

    Args:
        text: Text content
        path: Path to output file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_text(path: Union[str, Path]) -> str:
    """
    Load text from file.

    Args:
        path: Path to text file

    Returns:
        Text content
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
