"""Test utilities for pooling and time encoding."""

import pytest
import torch
from tide_lite.utils.common import (
    mean_pool,
    max_pool,
    cls_pool,
    sinusoidal_time_encoding,
    cosine_similarity_matrix,
    count_parameters,
)


class TestPooling:
    """Test pooling functions."""
    
    def test_mean_pooling(self):
        """Test mean pooling with padding."""
        batch_size = 2
        seq_len = 5
        hidden_dim = 8
        
        # Create dummy data
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 3:] = 0  # Mask last 2 tokens of first sample
        
        # Apply mean pooling
        pooled = mean_pool(hidden_states, attention_mask)
        
        # Check shape
        assert pooled.shape == (batch_size, hidden_dim)
        
        # Check that masked tokens are ignored
        expected_first = hidden_states[0, :3].mean(dim=0)
        torch.testing.assert_close(pooled[0], expected_first, rtol=1e-5)
    
    def test_max_pooling(self):
        """Test max pooling with padding."""
        batch_size = 2
        seq_len = 5
        hidden_dim = 8
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 3:] = 0
        
        pooled = max_pool(hidden_states, attention_mask)
        
        assert pooled.shape == (batch_size, hidden_dim)
        
        # Masked tokens should not affect max
        expected_first = hidden_states[0, :3].max(dim=0)[0]
        torch.testing.assert_close(pooled[0], expected_first, rtol=1e-5)
    
    def test_cls_pooling(self):
        """Test CLS token extraction."""
        batch_size = 2
        seq_len = 5
        hidden_dim = 8
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        pooled = cls_pool(hidden_states)
        
        assert pooled.shape == (batch_size, hidden_dim)
        torch.testing.assert_close(pooled, hidden_states[:, 0, :])


class TestTimeEncoding:
    """Test time encoding functions."""
    
    def test_sinusoidal_encoding_shape(self):
        """Test sinusoidal encoding output shape."""
        batch_size = 4
        dims = 32
        
        timestamps = torch.tensor([0.0, 1000.0, 10000.0, 100000.0])
        encoding = sinusoidal_time_encoding(timestamps, dims)
        
        assert encoding.shape == (batch_size, dims)
    
    def test_sinusoidal_encoding_range(self):
        """Test that sinusoidal encodings are bounded."""
        timestamps = torch.linspace(0, 1e6, 100)
        encoding = sinusoidal_time_encoding(timestamps, dims=64)
        
        # Sin and cos are bounded in [-1, 1]
        assert encoding.min() >= -1.01  # Small tolerance
        assert encoding.max() <= 1.01
    
    def test_sinusoidal_encoding_odd_dims(self):
        """Test that odd dimensions raise error."""
        timestamps = torch.tensor([1.0])
        
        with pytest.raises(ValueError, match="dims must be even"):
            sinusoidal_time_encoding(timestamps, dims=33)


class TestSimilarity:
    """Test similarity computation functions."""
    
    def test_cosine_similarity_matrix(self):
        """Test cosine similarity matrix computation."""
        batch_size = 3
        hidden_dim = 8
        
        x = torch.randn(batch_size, hidden_dim)
        sim_matrix = cosine_similarity_matrix(x)
        
        assert sim_matrix.shape == (batch_size, batch_size)
        
        # Diagonal should be ~1 (self-similarity)
        diag = torch.diagonal(sim_matrix)
        torch.testing.assert_close(diag, torch.ones(batch_size), atol=1e-5)
        
        # Matrix should be symmetric
        torch.testing.assert_close(sim_matrix, sim_matrix.T, atol=1e-5)
    
    def test_cosine_similarity_cross(self):
        """Test cross-similarity between two sets."""
        x = torch.randn(3, 8)
        y = torch.randn(4, 8)
        
        sim_matrix = cosine_similarity_matrix(x, y)
        
        assert sim_matrix.shape == (3, 4)
        
        # Values should be in [-1, 1]
        assert sim_matrix.min() >= -1.01
        assert sim_matrix.max() <= 1.01


class TestModelUtils:
    """Test model utility functions."""
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.Linear(20, 10),
        )
        
        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False
        
        counts = count_parameters(model)
        
        # First layer: 10*20 + 20 = 220
        # Second layer: 20*10 + 10 = 210
        # Total: 430
        assert counts["total"] == 430
        assert counts["trainable"] == 210
        assert counts["frozen"] == 220


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
