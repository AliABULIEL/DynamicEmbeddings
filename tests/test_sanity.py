"""Sanity tests for basic imports and CLI functionality."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_imports():
    """Test that main package imports work."""
    # Basic imports
    import temporal_lora
    from temporal_lora.utils import get_logger, set_seed, setup_logging

    assert temporal_lora.__version__ == "0.1.0"
    assert callable(setup_logging)
    assert callable(get_logger)
    assert callable(set_seed)


def test_cli_help():
    """Test that CLI can be invoked and shows help."""
    result = subprocess.run(
        [sys.executable, "-m", "temporal_lora.cli", "--help"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "temporal-lora" in result.stdout.lower() or "temporal_lora" in result.stdout.lower()
    assert "command" in result.stdout.lower()


def test_cli_commands_exist():
    """Test that all expected CLI commands exist."""
    result = subprocess.run(
        [sys.executable, "-m", "temporal_lora.cli", "--help"],
        capture_output=True,
        text=True,
    )
    
    expected_commands = [
        "env-dump",
        "prepare-data",
        "train-adapters",
        "build-indexes",
        "evaluate",
        "visualize",
        "ablate",
        "export-deliverables",
    ]
    
    for command in expected_commands:
        assert command in result.stdout, f"Command '{command}' not found in CLI help"


def test_utils_seeding():
    """Test that seeding utilities work."""
    from temporal_lora.utils.seeding import set_seed

    # Should not raise
    set_seed(42)
    set_seed(42, deterministic=True)
    set_seed(42, deterministic=False)


def test_utils_paths():
    """Test that path utilities work."""
    from temporal_lora.utils.paths import (
        get_config_dir,
        get_data_dir,
        get_deliverables_dir,
        get_models_dir,
        get_project_root,
    )

    root = get_project_root()
    assert root.exists()
    assert root.is_dir()
    
    # Check that key paths exist or can be created
    assert get_config_dir().exists()
    
    # These may not exist yet but should be valid paths
    assert isinstance(get_data_dir(), Path)
    assert isinstance(get_models_dir(), Path)
    assert isinstance(get_deliverables_dir(), Path)


def test_utils_io():
    """Test that I/O utilities can be imported."""
    from temporal_lora.utils.io import load_config, save_json

    assert callable(load_config)
    assert callable(save_json)


def test_config_files_exist():
    """Test that all config files exist."""
    from temporal_lora.utils.paths import get_config_dir

    config_dir = get_config_dir()
    
    expected_configs = [
        "data.yaml",
        "model.yaml",
        "train.yaml",
        "eval.yaml",
    ]
    
    for config_file in expected_configs:
        config_path = config_dir / config_file
        assert config_path.exists(), f"Config file missing: {config_file}"


def test_load_configs():
    """Test that config files can be loaded."""
    from temporal_lora.utils.io import load_config
    from temporal_lora.utils.paths import get_config_dir

    config_dir = get_config_dir()
    
    configs_to_test = [
        "data.yaml",
        "model.yaml",
        "train.yaml",
        "eval.yaml",
    ]
    
    for config_file in configs_to_test:
        config_path = config_dir / config_file
        config = load_config(config_path)
        assert config is not None, f"Failed to load config: {config_file}"
