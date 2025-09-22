"""Unit tests for utility functions."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.tide_lite.utils.common import (
    EarlyStopping,
    Timer,
    count_parameters,
    create_output_dir,
    format_metrics,
    get_device,
    load_json,
    save_json,
    seed_everything,
)
from src.tide_lite.utils.config import setup_logging


class TestTimer:
    """Test Timer utility."""
    
    def test_timer_basic(self):
        """Test basic timer functionality."""
        timer = Timer(verbose=False)
        timer.start()
        assert timer.start_time is not None
        elapsed = timer.stop()
        assert elapsed > 0
        assert timer.elapsed == elapsed
    
    def test_timer_context_manager(self):
        """Test timer as context manager."""
        with Timer(name="test", verbose=False) as t:
            assert t.start_time is not None
        assert t.elapsed > 0
    
    def test_timer_not_started_error(self):
        """Test error when stopping unstarted timer."""
        timer = Timer(verbose=False)
        with pytest.raises(RuntimeError, match="Timer not started"):
            timer.stop()


class TestEarlyStopping:
    """Test early stopping functionality."""
    
    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode."""
        es = EarlyStopping(patience=2, mode="max")
        
        # Initial score
        assert es(1.0) is False
        assert es.best_score == 1.0
        
        # Improvement
        assert es(1.5) is False
        assert es.best_score == 1.5
        assert es.counter == 0
        
        # No improvement
        assert es(1.4) is False
        assert es.counter == 1
        
        # Still no improvement - should stop
        assert es(1.3) is False
        assert es.counter == 2
        assert es(1.2) is True
        assert es.should_stop is True
    
    def test_early_stopping_min_mode(self):
        """Test early stopping in min mode."""
        es = EarlyStopping(patience=1, mode="min", min_delta=0.1)
        
        # Initial score
        assert es(2.0) is False
        
        # Small improvement (not enough due to min_delta)
        assert es(1.95) is False
        assert es.counter == 1
        
        # No improvement - should stop
        assert es(1.96) is True
        
    def test_early_stopping_with_min_delta(self):
        """Test early stopping with minimum delta."""
        es = EarlyStopping(patience=2, mode="max", min_delta=0.1)
        
        assert es(1.0) is False
        assert es(1.05) is False  # Not enough improvement
        assert es.counter == 1
        assert es(1.15) is False  # Improvement above delta
        assert es.counter == 0


class TestUtilityFunctions:
    """Test various utility functions."""
    
    def test_seed_everything(self):
        """Test seed setting for reproducibility."""
        seed_everything(42)
        
        # Check PyTorch seed
        torch1 = torch.randn(5)
        seed_everything(42)
        torch2 = torch.randn(5)
        assert torch.allclose(torch1, torch2)
    
    def test_get_device_auto(self):
        """Test automatic device selection."""
        with patch('torch.cuda.is_available', return_value=True):
            device = get_device("auto")
            assert device.type == "cuda"
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                device = get_device("auto")
                assert device.type == "mps"
            
            with patch('torch.backends.mps.is_available', return_value=False):
                device = get_device("auto")
                assert device.type == "cpu"
    
    def test_get_device_explicit(self):
        """Test explicit device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        device = get_device("cuda:0")
        assert "cuda" in str(device)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.Linear(20, 5)
        )
        
        # Total parameters: (10*20 + 20) + (20*5 + 5) = 220 + 105 = 325
        total = count_parameters(model, trainable_only=False)
        assert total == 325
        
        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False
        
        trainable = count_parameters(model, trainable_only=True)
        assert trainable == 105  # Only second layer
    
    def test_format_metrics(self):
        """Test metrics formatting."""
        metrics = {"loss": 0.1234, "accuracy": 0.9876, "epoch": 10}
        formatted = format_metrics(metrics, precision=2)
        assert "loss: 0.12" in formatted
        assert "accuracy: 0.99" in formatted
        assert "epoch: 10" in formatted
    
    def test_create_output_dir(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            
            # Without timestamp
            output_dir = create_output_dir(base_dir, "test_exp", timestamp=False)
            assert output_dir.exists()
            assert output_dir.name == "test_exp"
            
            # With timestamp
            output_dir = create_output_dir(base_dir, "test_exp", timestamp=True)
            assert output_dir.exists()
            assert output_dir.name.startswith("test_exp_")
    
    def test_save_load_json(self):
        """Test JSON save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test.json"
            
            data = {"key": "value", "number": 42, "list": [1, 2, 3]}
            save_json(data, json_path)
            
            assert json_path.exists()
            
            loaded_data = load_json(json_path)
            assert loaded_data == data


class TestLoggingSetup:
    """Test logging configuration."""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            
            setup_logging(level="DEBUG", log_file=log_file)
            
            # Log file should be created
            assert log_file.exists()
    
    def test_setup_logging_levels(self):
        """Test different logging levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            setup_logging(level=level)
            # No errors should be raised


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
