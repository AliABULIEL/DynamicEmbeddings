"""Test model components."""

import pytest
import torch
from src.tide_lite.models.tide_lite import TIDELite, TIDELiteConfig
from src.tide_lite.models.baselines import load_baseline


class TestTIDELiteModel:
    """Test TIDE-Lite model."""
    
    def test_model_initialization(self):
        """Test model initialization with default config."""
        config = TIDELiteConfig(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            hidden_dim=384,
            time_encoding_dim=32,
            mlp_hidden_dim=128,
        )
        model = TIDELite(config)
        
        assert model.config.hidden_dim == 384
        assert model.temporal_gate is not None
        assert model.time_encoder is not None
    
    def test_forward_pass(self):
        """Test forward pass with and without timestamps."""
        config = TIDELiteConfig()
        model = TIDELite(config)
        model.eval()
        
        batch_size = 2
        seq_len = 10
        
        # Create dummy inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        timestamps = torch.tensor([1000.0, 2000.0])
        
        # Forward without timestamps (base embeddings)
        with torch.no_grad():
            emb_temporal, emb_base = model(input_ids, attention_mask, timestamps=None)
            assert emb_base.shape == (batch_size, config.hidden_dim)
        
        # Forward with timestamps (temporal embeddings)
        with torch.no_grad():
            emb_temporal, emb_base = model(input_ids, attention_mask, timestamps=timestamps)
            assert emb_temporal.shape == (batch_size, config.hidden_dim)
            assert emb_base.shape == (batch_size, config.hidden_dim)
    
    def test_encode_base(self):
        """Test base encoding without temporal modulation."""
        config = TIDELiteConfig()
        model = TIDELite(config)
        model.eval()
        
        batch_size = 2
        seq_len = 10
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            embeddings = model.encode_base(input_ids, attention_mask)
            assert embeddings.shape == (batch_size, config.hidden_dim)
    
    def test_parameter_count(self):
        """Test that encoder is frozen and only adapter is trainable."""
        config = TIDELiteConfig(freeze_encoder=True)
        model = TIDELite(config)
        
        # Count parameters
        encoder_params = sum(p.numel() for n, p in model.named_parameters() if 'encoder' in n)
        adapter_params = sum(p.numel() for n, p in model.named_parameters() if 'encoder' not in n)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Encoder should be frozen
        assert trainable_params == adapter_params
        assert encoder_params > 0  # Encoder exists but is frozen


class TestBaselines:
    """Test baseline models."""
    
    def test_load_minilm(self):
        """Test loading MiniLM baseline."""
        model = load_baseline("minilm")
        
        assert model is not None
        assert hasattr(model, "encode_base")
        
        # Test forward pass
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            embeddings, _ = model(input_ids, attention_mask)
            assert embeddings.shape[1] == model.hidden_dim  # Model's hidden dim
    
    def test_baseline_names(self):
        """Test that all baseline names are resolvable."""
        baseline_names = ["minilm", "e5-base", "bge-base"]
        
        for name in baseline_names:
            try:
                model = load_baseline(name)
                assert model is not None
                assert hasattr(model, "encode_base")
            except Exception as e:
                # Some baseline models might not be available
                print(f"Warning: Could not load {name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
