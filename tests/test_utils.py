"""Unit tests for TIDE-Lite utilities."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.tide_lite.utils.config import set_global_seed
from src.tide_lite.utils.common import format_metrics_table


def test_set_global_seed():
    """Test reproducible seeding."""
    set_global_seed(42)
    
    # Test torch randomness
    t1 = torch.rand(5)
    set_global_seed(42)
    t2 = torch.rand(5)
    
    assert torch.allclose(t1, t2), "Torch seed not set correctly"
    
    # Test numpy randomness
    set_global_seed(42)
    n1 = np.random.rand(5)
    set_global_seed(42)
    n2 = np.random.rand(5)
    
    np.testing.assert_array_equal(n1, n2)


def test_format_metrics_table():
    """Test metrics table formatting."""
    metrics = {
        "spearman": 0.785,
        "pearson": 0.792,
        "mse": 0.423,
    }
    
    table = format_metrics_table(metrics, "Test Model")
    
    assert "Test Model" in table
    assert "0.785" in table
    assert "spearman" in table.lower()


def test_imports():
    """Test all critical imports work."""
    try:
        from src.tide_lite.models import TIDELite, TIDELiteConfig
        from src.tide_lite.train import TIDETrainer, TrainingConfig
        from src.tide_lite.data.datasets import load_stsb_with_timestamps
        from src.tide_lite.eval.eval_stsb import evaluate_stsb
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
