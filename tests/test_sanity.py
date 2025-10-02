"""Sanity tests for basic project setup."""

import subprocess
import sys


def test_imports():
    """Test that core package imports work."""
    import temporal_lora
    from temporal_lora import cli
    from temporal_lora.utils import paths, seeding, logging, io, env
    
    assert temporal_lora.__version__ == "0.1.0"
    assert hasattr(cli, "app")
    assert hasattr(paths, "PROJECT_ROOT")
    assert hasattr(seeding, "set_seed")
    assert hasattr(logging, "setup_logger")
    assert hasattr(io, "load_yaml")
    assert hasattr(env, "get_cuda_info")


def test_cli_help():
    """Test that CLI help command runs without error."""
    result = subprocess.run(
        [sys.executable, "-m", "temporal_lora.cli", "--help"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "temporal-lora" in result.stdout.lower()
    assert "env-dump" in result.stdout
    assert "prepare-data" in result.stdout
    assert "train-adapters" in result.stdout
    assert "build-indexes" in result.stdout
    assert "evaluate" in result.stdout
    assert "visualize" in result.stdout
    assert "export-deliverables" in result.stdout


def test_config_files_exist():
    """Test that all config files exist."""
    from temporal_lora.utils.paths import CONFIG_DIR
    
    config_files = ["data.yaml", "model.yaml", "train.yaml", "eval.yaml"]
    
    for config_file in config_files:
        config_path = CONFIG_DIR / config_file
        assert config_path.exists(), f"Config file missing: {config_file}"


def test_load_configs():
    """Test that all configs load without errors."""
    from temporal_lora.utils.paths import CONFIG_DIR
    from temporal_lora.utils.io import load_config
    
    configs = ["data", "model", "train", "eval"]
    
    for config_name in configs:
        config = load_config(config_name, CONFIG_DIR)
        assert isinstance(config, dict)
        assert len(config) > 0, f"Config {config_name} is empty"
