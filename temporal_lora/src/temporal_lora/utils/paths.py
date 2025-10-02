"""
Path utilities for Temporal LoRA project.

Provides centralized path management for data, models, outputs, and deliverables.
"""

from pathlib import Path
from typing import Optional

# Project root (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT


def get_data_dir() -> Path:
    """Get the data directory."""
    return PROJECT_ROOT / "data"


def get_raw_data_dir() -> Path:
    """Get the raw data directory."""
    path = get_data_dir() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_processed_data_dir() -> Path:
    """Get the processed data directory."""
    path = get_data_dir() / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_models_dir() -> Path:
    """Get the models directory."""
    return PROJECT_ROOT / "models"


def get_adapters_dir() -> Path:
    """Get the adapters directory."""
    path = get_models_dir() / "adapters"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_checkpoints_dir() -> Path:
    """Get the checkpoints directory."""
    path = get_models_dir() / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_indexes_dir() -> Path:
    """Get the indexes directory."""
    path = get_models_dir() / "indexes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logs_dir() -> Path:
    """Get the logs directory."""
    path = PROJECT_ROOT / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_deliverables_dir() -> Path:
    """Get the deliverables directory."""
    path = PROJECT_ROOT / "deliverables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_figures_dir() -> Path:
    """Get the figures directory."""
    path = get_deliverables_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_tables_dir() -> Path:
    """Get the tables directory."""
    path = get_deliverables_dir() / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_repro_dir() -> Path:
    """Get the reproducibility directory."""
    path = get_deliverables_dir() / "repro"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to directory

    Returns:
        Path object for the directory
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
