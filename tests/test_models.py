"""Unit tests for TIDE-Lite models."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from src.tide_lite.models.tide_lite import (
    SinusoidalTimeEncoding,
    TemporalGatingMLP,
    TIDELite,
    TIDELiteConfig,
)


class TestSinusoidalTimeEncoding:
    """Test sinusoidal time encoding."""
    
    def test_encoding_shape(self):
        """Test output shape of time encoding."""
        batch_size = 8
        dim = 32
        
        encoder = SinusoidalTimeEncoding(dim)
        timestamps = torch.randn(batch_size, 1)
        
        encoding = encoder(timestamps)
        assert encoding.shape == (batch_size, dim)
    
    def test_encoding_dim_even(self):
        """Test that encoding dimension must be even."""
        with pytest.raises(ValueError, match="must be even"):
            SinusoidalTimeEncoding(33)
    
    def test_encoding_deterministic(self):
        """Test that encoding is deterministic."""
        encoder = SinusoidalTimeEncoding(16)
        timestamps = torch.tensor([[1.0], [2.0], [3.0]])
        
        encoding1 = encoder(timestamps)
        encoding2 = encoder(timestamps)
        
        assert torch.allclose(encoding1, encoding2)
    
    def test_encoding_range(self):
        """Test that encodings are in expected range [-1, 1]."""
        encoder = SinusoidalTimeEncoding(32)
        timestamps = torch.randn(100, 1) * 1000  # Large time values
        
        encoding = encoder(timestamps)
        assert encoding.min() >= -1.1  # Small tolerance for numerical errors
        assert encoding.max() <= 1.1


class TestTemporalGatingMLP:
    """Test temporal gating MLP."""
    
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        input_dim = 16
        hidden_dim = 32
        output_dim = 8
        batch_size = 4
        
        mlp = TemporalGatingMLP(input_dim, hidden_dim, output_dim)
        x = torch.randn(batch_size, input_dim)
        
        output = mlp(x)
        assert output.shape == (batch_size, output_dim)
    
    def test_mlp_activation_sigmoid(self):
        """Test sigmoid activation."""
        mlp = TemporalGatingMLP(10, 20, 5, activation="sigmoid")
        x = torch.randn(2, 10)
        output = mlp(x)
        
        # Sigmoid output should be in [0, 1]
        assert output.min() >= 0.0
        assert output.max() <= 1.0
    
    def test_mlp_activation_tanh(self):
        """Test tanh activation."""
        mlp = TemporalGatingMLP(10, 20, 5, activation="tanh")
        x = torch.randn(2, 10)
        output = mlp(x)
        
        # Tanh output should be in [-1, 1]
        assert output.min() >= -1.0
        assert output.max() <= 1.0
    
    def test_mlp_dropout(self):
        """Test dropout in training mode."""
        mlp = TemporalGatingMLP(10, 20, 5, dropout=0.5)
        mlp.train()
        
        x = torch.ones(100, 10)
        outputs = []
        for _ in range(10):
            outputs.append(mlp(x))
        
        # Outputs should vary due to dropout
        stacked = torch.stack(outputs)
        std = stacked.std(dim=0)
        assert std.mean() > 0.01  # Some variation expected


class TestTIDELiteConfig:
    """Test TIDE-Lite configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TIDELiteConfig()
        
        assert config.encoder_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.hidden_dim == 384
        assert config.time_encoding_dim == 32
        assert config.mlp_hidden_dim == 128
        assert config.mlp_dropout == 0.1
        assert config.gate_activation == "sigmoid"
        assert config.freeze_encoder is True
        assert config.pooling_strategy == "mean"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TIDELiteConfig(
            encoder_name="bert-base-uncased",
            hidden_dim=768,
            time_encoding_dim=64,
            freeze_encoder=False
        )
        
        assert config.encoder_name == "bert-base-uncased"
        assert config.hidden_dim == 768
        assert config.time_encoding_dim == 64
        assert config.freeze_encoder is False


class TestTIDELite:
    """Test TIDE-Lite model."""
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_model_initialization(self, mock_tokenizer, mock_model):
        """Test model initialization."""
        # Create mock encoder
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 384
        mock_model.return_value = mock_encoder
        
        # Create mock tokenizer
        mock_tok = MagicMock()
        mock_tokenizer.return_value = mock_tok
        
        config = TIDELiteConfig()
        model = TIDELite(config)
        
        assert model.config == config
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'time_encoder')
        assert hasattr(model, 'temporal_gate')
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_freeze_encoder(self, mock_tokenizer, mock_model):
        """Test encoder freezing."""
        # Create mock encoder with parameters
        mock_encoder = nn.Linear(10, 10)
        mock_encoder.config = MagicMock()
        mock_encoder.config.hidden_size = 384
        mock_model.return_value = mock_encoder
        
        config = TIDELiteConfig(freeze_encoder=True)
        model = TIDELite(config)
        
        # Check that encoder parameters are frozen
        for param in model.encoder.parameters():
            assert param.requires_grad is False
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')  
    def test_forward_shape(self, mock_tokenizer, mock_model):
        """Test forward pass output shape."""
        batch_size = 4
        seq_len = 128
        hidden_dim = 384
        
        # Create mock encoder
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = hidden_dim
        
        # Mock encoder output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)
        mock_encoder.return_value = mock_output
        mock_model.return_value = mock_encoder
        
        config = TIDELiteConfig(hidden_dim=hidden_dim)
        model = TIDELite(config)
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        timestamps = torch.randn(batch_size, 1)
        
        output = model(input_ids, attention_mask, timestamps)
        
        # Check output shape
        assert output.shape == (batch_size, hidden_dim)
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_parameter_count(self, mock_tokenizer, mock_model):
        """Test parameter counting with frozen encoder."""
        # Create a simple mock encoder
        mock_encoder = nn.Sequential(
            nn.Linear(100, 384),
            nn.Linear(384, 384)
        )
        mock_encoder.config = MagicMock()
        mock_encoder.config.hidden_size = 384
        mock_model.return_value = mock_encoder
        
        config = TIDELiteConfig(
            hidden_dim=384,
            time_encoding_dim=32,
            mlp_hidden_dim=128,
            freeze_encoder=True
        )
        model = TIDELite(config)
        
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Count trainable parameters (should exclude encoder)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Encoder params should be frozen
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        
        assert trainable_params < total_params
        assert trainable_params == total_params - encoder_params
    
    def test_pooling_strategies(self):
        """Test different pooling strategies."""
        strategies = ["mean", "cls", "max"]
        batch_size = 2
        seq_len = 10
        hidden_dim = 8
        
        for strategy in strategies:
            # Test pooling function directly
            from src.tide_lite.models.tide_lite import _pool_embeddings
            
            hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
            attention_mask = torch.ones(batch_size, seq_len)
            attention_mask[0, 5:] = 0  # Mask out second half of first sequence
            
            pooled = _pool_embeddings(hidden_states, attention_mask, strategy)
            assert pooled.shape == (batch_size, hidden_dim)
            
            if strategy == "cls":
                # CLS pooling should return first token
                expected = hidden_states[:, 0, :]
                assert torch.allclose(pooled, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
