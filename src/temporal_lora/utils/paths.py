"""Path utilities for project structure."""

from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_CACHE_DIR = DATA_DIR / ".cache"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
ADAPTERS_DIR = MODELS_DIR / "adapters"
INDEXES_DIR = MODELS_DIR / "indexes"

# Output directories
DELIVERABLES_DIR = PROJECT_ROOT / "deliverables"
DELIVERABLES_REPRO_DIR = DELIVERABLES_DIR / "repro"
DELIVERABLES_FIGURES_DIR = DELIVERABLES_DIR / "figures"
DELIVERABLES_RESULTS_DIR = DELIVERABLES_DIR / "results"

# Config directory
CONFIG_DIR = PROJECT_ROOT / "src" / "temporal_lora" / "config"

# Test directories
TESTS_DIR = PROJECT_ROOT / "tests"


def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    dirs = [
        DATA_DIR,
        DATA_PROCESSED_DIR,
        DATA_CACHE_DIR,
        MODELS_DIR,
        CHECKPOINTS_DIR,
        ADAPTERS_DIR,
        INDEXES_DIR,
        DELIVERABLES_DIR,
        DELIVERABLES_REPRO_DIR,
        DELIVERABLES_FIGURES_DIR,
        DELIVERABLES_RESULTS_DIR,
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
