"""Test pooling utilities."""

import pytest
import torch

from tide_lite.utils.common import mean_pool, max_pool, cls_pool


def test_mean_pooling():
    """Test mean pooling with attention mask."""
    # Create sample data
    batch_size, seq_len, hidden_dim = 2, 5, 4
    last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.tensor([
        [1, 1, 1, 0, 0],  # 3 valid tokens
        [1, 1, 1, 1, 1],  # 5 valid tokens
    ])
    
    # Apply mean pooling
    pooled = mean_pool(last_hidden_state, attention_mask)
    
    # Check shape
    assert pooled.shape == (batch_size, hidden_dim)
    
    # Check that padding is ignored
    # First sample should only average first 3 tokens
    expected_first = last_hidden_state[0, :3, :].mean(dim=0)
    assert torch.allclose(pooled[0], expected_first, atol=1e-6)


def test_max_pooling():
    """Test max pooling with attention mask."""
    batch_size, seq_len, hidden_dim = 2, 5, 4
    last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ])
    
    pooled = max_pool(last_hidden_state, attention_mask)
    
    assert pooled.shape == (batch_size, hidden_dim)


def test_cls_pooling():
    """Test CLS token extraction."""
    batch_size, seq_len, hidden_dim = 2, 5, 4
    last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
    
    pooled = cls_pool(last_hidden_state)
    
    assert pooled.shape == (batch_size, hidden_dim)
    assert torch.allclose(pooled, last_hidden_state[:, 0, :])
