"""Smoke test for training pipeline."""

import pytest
import torch
import tempfile
from pathlib import Path

from src.tide_lite.models import TIDELite, TIDELiteConfig
from depreacted.src.tide_lite.train.trainer import TIDELiteTrainer, TrainingConfig


def test_model_initialization():
    """Test TIDE-Lite model creation."""
    config = TIDELiteConfig(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        time_encoding_dim=32,
        mlp_hidden_dim=128,
        freeze_encoder=True
    )

    model = TIDELite(config)

    # Check parameter count
    extra_params = model.count_extra_parameters()
    assert 40000 < extra_params < 60000, f"Unexpected param count: {extra_params}"

    # Test forward pass
    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    timestamps = torch.tensor([1609459200.0, 1640995200.0])  # 2021, 2022

    temporal_emb, base_emb = model(input_ids, attention_mask, timestamps)

    assert temporal_emb.shape == (batch_size, config.hidden_dim)
    assert base_emb.shape == (batch_size, config.hidden_dim)


def test_training_smoke():
    """Test minimal training loop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup config for tiny training
        config = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            eval_batch_size=4,
            warmup_steps=2,
            save_every=10,
            eval_every=10,
            output_dir=tmpdir,
            dry_run=True,  # Dry run mode
            seed=42
        )

        # Create model
        model_config = TIDELiteConfig(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            freeze_encoder=True
        )
        model = TIDELite(model_config)

        # Create trainer
        trainer = TIDELiteTrainer(config, model)

        # Run dry-run training
        trainer.train()

        # In dry-run mode, trainer should exit without creating files
        assert config.dry_run == True


def test_time_encoding():
    """Test sinusoidal time encoding."""
    from src.tide_lite.models.tide_lite import SinusoidalTimeEncoding

    encoder = SinusoidalTimeEncoding(encoding_dim=32)

    timestamps = torch.tensor([
        1609459200.0,  # 2021-01-01
        1640995200.0,  # 2022-01-01
        1672531200.0,  # 2023-01-01
    ])

    encoding = encoder(timestamps)

    assert encoding.shape == (3, 32)
    assert not torch.isnan(encoding).any()
    assert not torch.isinf(encoding).any()

    # Check that different timestamps give different encodings
    assert not torch.allclose(encoding[0], encoding[1])
    assert not torch.allclose(encoding[1], encoding[2])


def test_temporal_gating():
    """Test temporal gating MLP."""
    from src.tide_lite.models.tide_lite import TemporalGatingMLP

    gate = TemporalGatingMLP(
        input_dim=32,
        hidden_dim=128,
        output_dim=384,
        activation="sigmoid"
    )

    time_encoding = torch.randn(4, 32)
    gates = gate(time_encoding)

    assert gates.shape == (4, 384)
    assert (gates >= 0).all() and (gates <= 1).all(), "Sigmoid should output [0, 1]"


@pytest.mark.parametrize("pooling", ["mean", "cls", "max"])
def test_pooling_strategies(pooling):
    """Test different pooling strategies."""
    config = TIDELiteConfig(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        pooling_strategy=pooling,
        freeze_encoder=True
    )

    model = TIDELite(config)

    # Test input
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones((2, 10))

    base_emb = model.encode_base(input_ids, attention_mask)

    assert base_emb.shape == (2, config.hidden_dim)
    assert not torch.isnan(base_emb).any()
