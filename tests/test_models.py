"""Unit tests for embedding models."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.append('..')

from src.models.base_embedding import BaseEmbedding
from src.models.matryoshka_embedding import MatryoshkaEmbedding
from src.models.temporal_embedding import TemporalEmbedding
from src.models.contextual_embedding import ContextualEmbedding


class TestBaseEmbedding:
    """Tests for base embedding functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 1000
        self.embedding_dim = 128
        self.batch_size = 8
        self.seq_len = 32

    def test_similarity_computation(self):
        """Test similarity computation methods."""
        # Create dummy embeddings
        embed1 = torch.randn(self.batch_size, self.embedding_dim)
        embed2 = torch.randn(self.batch_size, self.embedding_dim)

        # Create a concrete implementation for testing
        class TestEmbedding(BaseEmbedding):
            def forward(self, inputs, context=None, **kwargs):
                return torch.randn(inputs.shape[0], inputs.shape[1], self.embedding_dim)

        model = TestEmbedding(self.input_dim, self.embedding_dim)

        # Test cosine similarity
        cos_sim = model.compute_similarity(embed1, embed2, metric='cosine')
        assert cos_sim.shape == (self.batch_size,)
        assert torch.all(cos_sim >= -1) and torch.all(cos_sim <= 1)

        # Test euclidean distance
        euc_sim = model.compute_similarity(embed1, embed2, metric='euclidean')
        assert euc_sim.shape == (self.batch_size,)
        assert torch.all(euc_sim <= 0)

        # Test dot product
        dot_sim = model.compute_similarity(embed1, embed2, metric='dot')
        assert dot_sim.shape == (self.batch_size,)

    def test_pooling_methods(self):
        """Test different pooling strategies."""

        class TestEmbedding(BaseEmbedding):
            def forward(self, inputs, context=None, **kwargs):
                return torch.randn(inputs.shape[0], inputs.shape[1], self.embedding_dim)

        model = TestEmbedding(self.input_dim, self.embedding_dim)

        # Create test embeddings and mask
        embeddings = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)
        mask = torch.ones(self.batch_size, self.seq_len)
        mask[:, self.seq_len // 2:] = 0  # Mask second half

        # Test mean pooling
        mean_pooled = model.pool_embeddings(embeddings, mask, pooling='mean')
        assert mean_pooled.shape == (self.batch_size, self.embedding_dim)

        # Test max pooling
        max_pooled = model.pool_embeddings(embeddings, pooling='max')
        assert max_pooled.shape == (self.batch_size, self.embedding_dim)

        # Test CLS pooling
        cls_pooled = model.pool_embeddings(embeddings, pooling='cls')
        assert cls_pooled.shape == (self.batch_size, self.embedding_dim)
        assert torch.allclose(cls_pooled, embeddings[:, 0, :])


class TestMatryoshkaEmbedding:
    """Tests for Matryoshka embeddings."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = MatryoshkaEmbedding(
            input_dim=1000,
            embedding_dim=768,
            nested_dims=[64, 128, 256, 512, 768],
            num_layers=2,
            num_heads=8
        )
        self.batch_size = 4
        self.seq_len = 32

    def test_forward_pass(self):
        """Test forward pass with different configurations."""
        inputs = torch.randint(0, 1000, (self.batch_size, self.seq_len))

        # Test default forward pass
        output = self.model(inputs)
        assert output.shape == (self.batch_size, 768)

        # Test specific dimension
        output_128 = self.model(inputs, target_dim=128)
        assert output_128.shape == (self.batch_size, 128)

        # Test truncated dimension
        output_100 = self.model(inputs, target_dim=100)
        assert output_100.shape == (self.batch_size, 100)

        # Test all dimensions
        all_outputs = self.model(inputs, return_all_dims=True)
        assert isinstance(all_outputs, dict)
        assert set(all_outputs.keys()) == {64, 128, 256, 512, 768}

        for dim, output in all_outputs.items():
            assert output.shape == (self.batch_size, dim)

    def test_matryoshka_consistency(self):
        """Test that truncated embeddings are consistent."""
        inputs = torch.randint(0, 1000, (self.batch_size, self.seq_len))

        # Get full embedding
        full_embed = self.model(inputs, target_dim=768)

        # Get smaller embedding
        small_embed = self.model(inputs, target_dim=128)

        # Check that embeddings are normalized if specified
        if self.model.normalize:
            full_norm = torch.norm(full_embed, p=2, dim=-1)
            small_norm = torch.norm(small_embed, p=2, dim=-1)
            assert torch.allclose(full_norm, torch.ones_like(full_norm), atol=1e-5)
            assert torch.allclose(small_norm, torch.ones_like(small_norm), atol=1e-5)

    def test_loss_computation(self):
        """Test Matryoshka loss computation."""
        inputs = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        embeddings_dict = self.model(inputs, return_all_dims=True)

        # Create dummy positive and negative pairs
        positive_pairs = torch.tensor([[0, 1], [2, 3]])
        negative_pairs = torch.tensor([[0, 2], [1, 3]])

        loss = self.model.compute_matryoshka_loss(
            embeddings_dict,
            positive_pairs,
            negative_pairs,
            temperature=0.05
        )

        assert isinstance(loss.item(), float)
        assert loss.item() > 0


class TestTemporalEmbedding:
    """Tests for temporal embeddings."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = TemporalEmbedding(
            input_dim=1000,
            embedding_dim=256,
            num_ssm_layers=2,
            state_dim=32
        )
        self.batch_size = 4
        self.seq_len = 16

    def test_forward_with_timestamps(self):
        """Test forward pass with temporal information."""
        inputs = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        timestamps = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch_size, -1).float()

        # Test without timestamps
        output = self.model(inputs)
        assert output.shape == (self.batch_size, 256)

        # Test with timestamps
        output_temporal = self.model(inputs, timestamps=timestamps)
        assert output_temporal.shape == (self.batch_size, 256)

        # Test sequence output
        output_seq = self.model(inputs, timestamps=timestamps, return_sequence=True)
        assert output_seq.shape == (self.batch_size, self.seq_len, 256)

    def test_temporal_consistency_loss(self):
        """Test temporal consistency loss computation."""
        # Create embeddings at different time points
        embeddings_t1 = torch.randn(self.batch_size, 256)
        embeddings_t2 = torch.randn(self.batch_size, 256)
        time_diff = torch.tensor([1.0, 2.0, 5.0, 10.0])

        loss = self.model.compute_temporal_consistency_loss(
            embeddings_t1,
            embeddings_t2,
            time_diff,
            temperature=1.0
        )

        assert isinstance(loss.item(), float)
        assert loss.item() >= 0

    def test_state_space_layer(self):
        """Test SSM layer functionality."""
        from src.models.temporal_embedding import StateSpaceLayer

        layer = StateSpaceLayer(d_model=256, state_dim=32)
        inputs = torch.randn(self.batch_size, self.seq_len, 256)

        output = layer(inputs)
        assert output.shape == inputs.shape

        # Check that output is different from input (transformation applied)
        assert not torch.allclose(output, inputs)


class TestContextualEmbedding:
    """Tests for contextual embeddings."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = ContextualEmbedding(
            input_dim=1000,
            embedding_dim=256,
            context_window=32,
            num_context_layers=4,
            use_position_agnostic=True
        )
        self.batch_size = 4
        self.seq_len = 64

    def test_context_window_extraction(self):
        """Test context window extraction."""
        inputs = torch.randint(0, 1000, (self.batch_size, self.seq_len))

        windows, masks = self.model.extract_context_windows(inputs, window_size=32)

        assert windows.shape == (self.batch_size, self.seq_len, 32)
        assert masks.shape == (self.batch_size, self.seq_len, 32)

        # Check that masks are binary
        assert torch.all((masks == 0) | (masks == 1))

    def test_forward_pass(self):
        """Test forward pass with different configurations."""
        inputs = torch.randint(0, 1000, (self.batch_size, self.seq_len))

        # Basic forward pass
        output = self.model(inputs)
        assert output.shape == (self.batch_size, 256)

        # With external context
        context = torch.randn(self.batch_size, 128)
        output_with_context = self.model(inputs, context=context)
        assert output_with_context.shape == (self.batch_size, 256)

    def test_position_agnostic_mode(self):
        """Test position-agnostic configuration."""
        # Create model without position embeddings
        model_agnostic = ContextualEmbedding(
            input_dim=1000,
            embedding_dim=256,
            use_position_agnostic=True
        )

        # Create model with position embeddings
        model_with_pos = ContextualEmbedding(
            input_dim=1000,
            embedding_dim=256,
            use_position_agnostic=False
        )

        assert model_agnostic.position_embedding is None
        assert model_with_pos.position_embedding is not None

        # Test that both produce valid outputs
        inputs = torch.randint(0, 1000, (self.batch_size, 32))

        output_agnostic = model_agnostic(inputs)
        output_with_pos = model_with_pos(inputs)

        assert output_agnostic.shape == (self.batch_size, 256)
        assert output_with_pos.shape == (self.batch_size, 256)


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for model interactions."""

    def test_model_comparison(self):
        """Test that different models produce compatible outputs."""
        input_dim = 1000
        embedding_dim = 256
        batch_size = 4
        seq_len = 32

        # Create models
        models = {
            'matryoshka': MatryoshkaEmbedding(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                nested_dims=[256],
                num_layers=2
            ),
            'temporal': TemporalEmbedding(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                num_ssm_layers=2
            ),
            'contextual': ContextualEmbedding(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                context_window=16
            )
        }

        # Test with same input
        inputs = torch.randint(0, input_dim, (batch_size, seq_len))

        outputs = {}
        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                outputs[name] = model(inputs)

        # Check all outputs have same shape
        for name, output in outputs.items():
            assert output.shape == (batch_size, embedding_dim)

        # Check outputs are normalized if specified
        for name, output in outputs.items():
            if models[name].normalize:
                norms = torch.norm(output, p=2, dim=-1)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_model_serialization(self, tmp_path):
        """Test model saving and loading."""
        model = MatryoshkaEmbedding(
            input_dim=1000,
            embedding_dim=256,
            nested_dims=[128, 256]
        )

        # Generate sample output
        inputs = torch.randint(0, 1000, (2, 16))
        output_before = model(inputs).detach()

        # Save model
        save_path = tmp_path / "model.pt"
        model.save(str(save_path))

        # Load model
        loaded_model = MatryoshkaEmbedding.load(str(save_path))

        # Compare outputs
        output_after = loaded_model(inputs).detach()
        assert torch.allclose(output_before, output_after, atol=1e-6)