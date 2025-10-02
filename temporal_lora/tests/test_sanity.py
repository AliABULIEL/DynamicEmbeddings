"""
Sanity tests for Temporal LoRA project.

Ensures basic functionality: imports, CLI availability, and file structure.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_imports():
    """Test that core modules can be imported."""
    import temporal_lora
    from temporal_lora import cli
    from temporal_lora.utils import (
        get_logger,
        get_project_root,
        load_json,
        save_json,
        set_seed,
    )

    assert temporal_lora.__version__ == "0.1.0"
    assert get_project_root().exists()


def test_cli_help():
    """Test that CLI shows help without errors."""
    result = subprocess.run(
        [sys.executable, "-m", "temporal_lora.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Temporal LoRA" in result.stdout


def test_cli_commands():
    """Test that all expected CLI commands are available."""
    result = subprocess.run(
        [sys.executable, "-m", "temporal_lora.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    expected_commands = [
        "prepare-data",
        "train-adapters",
        "build-indexes",
        "evaluate",
        "visualize",
        "ablate",
        "export-deliverables",
        "env-dump",
    ]

    for cmd in expected_commands:
        assert cmd in result.stdout, f"Command '{cmd}' not found in CLI help"


def test_seeding():
    """Test that seeding works correctly."""
    import numpy as np
    import torch

    from temporal_lora.utils.seeding import set_seed

    set_seed(42)
    val1 = np.random.rand()
    val2 = torch.rand(1).item()

    set_seed(42)
    val1_repeat = np.random.rand()
    val2_repeat = torch.rand(1).item()

    assert val1 == val1_repeat
    assert val2 == val2_repeat


def test_io_json(tmp_path):
    """Test JSON I/O utilities."""
    from temporal_lora.utils.io import load_json, save_json

    test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    test_file = tmp_path / "test.json"

    save_json(test_data, test_file)
    loaded_data = load_json(test_file)

    assert loaded_data == test_data


def test_paths():
    """Test that path utilities return valid paths."""
    from temporal_lora.utils.paths import (
        get_adapters_dir,
        get_data_dir,
        get_deliverables_dir,
        get_project_root,
    )

    root = get_project_root()
    assert root.exists()

    # These should create directories if they don't exist
    data_dir = get_data_dir()
    adapters_dir = get_adapters_dir()
    deliverables_dir = get_deliverables_dir()

    assert isinstance(data_dir, Path)
    assert isinstance(adapters_dir, Path)
    assert isinstance(deliverables_dir, Path)


def test_config_files_exist():
    """Test that all config YAML files exist."""
    from temporal_lora.utils.paths import get_project_root

    config_dir = get_project_root() / "src" / "temporal_lora" / "config"
    assert config_dir.exists()

    expected_configs = ["data.yaml", "model.yaml", "train.yaml", "eval.yaml"]
    for config in expected_configs:
        config_path = config_dir / config
        assert config_path.exists(), f"Config file {config} not found"


def test_logger():
    """Test that logger can be initialized."""
    from temporal_lora.utils.logging import get_logger, setup_logger

    setup_logger(level="INFO")
    logger = get_logger(__name__)

    logger.info("Test log message")
    # If this runs without error, logger is working
