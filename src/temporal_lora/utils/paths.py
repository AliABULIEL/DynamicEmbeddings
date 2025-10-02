"""Path management utilities."""

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path: Absolute path to project root (contains src/, tests/, etc.)
    """
    # Start from this file's location
    current = Path(__file__).resolve()
    
    # Navigate up: utils/ -> temporal_lora/ -> src/ -> root/
    return current.parent.parent.parent.parent


def setup_directories(root: Optional[Path] = None) -> dict[str, Path]:
    """Create all required project directories.
    
    Args:
        root: Project root directory. If None, auto-detect.
    
    Returns:
        dict: Mapping of directory names to Path objects
    """
    if root is None:
        root = get_project_root()
    
    directories = {
        # Data directories
        "data": root / "data",
        "data_processed": root / "data" / "processed",
        "data_cache": root / "data" / ".cache",
        
        # Model directories
        "models": root / "models",
        "adapters": root / "models" / "adapters",
        "indexes": root / "models" / "indexes",
        
        # Output directories
        "deliverables": root / "deliverables",
        "results": root / "deliverables" / "results",
        "figures": root / "deliverables" / "figures",
        "repro": root / "deliverables" / "repro",
        
        # Documentation
        "report": root / "report",
        "slides": root / "slides",
        
        # Development
        "notebooks": root / "notebooks",
        "scripts": root / "scripts",
        "tests": root / "tests",
    }
    
    # Create all directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories


def get_config_dir() -> Path:
    """Get configuration directory path.
    
    Returns:
        Path: Path to config directory
    """
    return get_project_root() / "src" / "temporal_lora" / "config"


def get_data_dir() -> Path:
    """Get data directory path.
    
    Returns:
        Path: Path to data directory
    """
    return get_project_root() / "data"


def get_models_dir() -> Path:
    """Get models directory path.
    
    Returns:
        Path: Path to models directory
    """
    return get_project_root() / "models"


def get_deliverables_dir() -> Path:
    """Get deliverables directory path.
    
    Returns:
        Path: Path to deliverables directory
    """
    return get_project_root() / "deliverables"
