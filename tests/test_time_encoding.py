"""Test time encoding utilities."""

import pytest
import torch

from tide_lite.utils.common import sinusoidal_time_encoding


def test_sinusoidal_time_encoding():
    """Test sinusoidal time encoding generation."""
    batch_size = 4
    dims = 32
    
    # Create sample timestamps
    timestamps = torch.tensor([0.0, 100.0, 1000.0, 10000.0])
    
    # Generate encodings
    encodings = sinusoidal_time_encoding(timestamps, dims=dims)
    
    # Check shape
    assert encodings.shape == (batch_size, dims)
    
    # Check that encodings are bounded [-1, 1]
    assert encodings.min() >= -1.0
    assert encodings.max() <= 1.0
    
    # Check that different timestamps give different encodings
    assert not torch.allclose(encodings[0], encodings[1])
    assert not torch.allclose(encodings[1], encodings[2])


def test_sinusoidal_encoding_dimensions():
    """Test that encoding requires even dimensions."""
    timestamps = torch.tensor([1.0])
    
    # Should work with even dims
    encodings = sinusoidal_time_encoding(timestamps, dims=32)
    assert encodings.shape == (1, 32)
    
    # Should fail with odd dims
    with pytest.raises(ValueError, match="dims must be even"):
        sinusoidal_time_encoding(timestamps, dims=31)
