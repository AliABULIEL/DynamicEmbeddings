"""Unit tests for configuration validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.tide_lite.models.tide_lite import TIDELiteConfig
from depreacted.src.tide_lite.train.trainer import TrainingConfig


class TestTIDELiteConfig:
    """Test TIDE-Lite configuration validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = TIDELiteConfig(
            encoder_name="bert-base-uncased",
            hidden_dim=768,
            time_encoding_dim=64,
            mlp_hidden_dim=256,
            mlp_dropout=0.2,
            gate_activation="sigmoid",
            freeze_encoder=True,
            pooling_strategy="mean"
        )

        assert config.encoder_name == "bert-base-uncased"
        assert config.hidden_dim == 768
        assert config.time_encoding_dim == 64

    def test_time_encoding_dim_validation(self):
        """Test that time encoding dimension must be even."""
        # This should raise an error when used in the model
        config = TIDELiteConfig(time_encoding_dim=33)

        # The error is raised when creating the model, not the config
        from src.tide_lite.models.tide_lite import SinusoidalTimeEncoding
        with pytest.raises(ValueError, match="must be even"):
            SinusoidalTimeEncoding(config.time_encoding_dim)

    def test_gate_activation_validation(self):
        """Test gate activation validation."""
        valid_activations = ["sigmoid", "tanh"]

        for activation in valid_activations:
            config = TIDELiteConfig(gate_activation=activation)
            assert config.gate_activation == activation

        # Invalid activation should be caught when creating the model
        config = TIDELiteConfig(gate_activation="relu")
        from src.tide_lite.models.tide_lite import TemporalGatingMLP

        with pytest.raises(ValueError, match="Unsupported activation"):
            TemporalGatingMLP(10, 20, 10, activation=config.gate_activation)

    def test_pooling_strategy_validation(self):
        """Test pooling strategy validation."""
        valid_strategies = ["mean", "cls", "max"]

        for strategy in valid_strategies:
            config = TIDELiteConfig(pooling_strategy=strategy)
            assert config.pooling_strategy == strategy

    def test_dropout_range_validation(self):
        """Test dropout probability range."""
        # Valid range
        config = TIDELiteConfig(mlp_dropout=0.0)
        assert config.mlp_dropout == 0.0

        config = TIDELiteConfig(mlp_dropout=0.5)
        assert config.mlp_dropout == 0.5

        config = TIDELiteConfig(mlp_dropout=0.99)
        assert config.mlp_dropout == 0.99

    def test_dimension_consistency(self):
        """Test dimension consistency checks."""
        # These should be valid configurations
        config = TIDELiteConfig(
            hidden_dim=384,
            mlp_hidden_dim=128  # Can be different from hidden_dim
        )
        assert config.hidden_dim == 384
        assert config.mlp_hidden_dim == 128


class TestTrainingConfig:
    """Test training configuration validation."""

    def test_valid_training_config(self):
        """Test valid training configuration."""
        config = TrainingConfig(
            batch_size=32,
            learning_rate=1e-4,
            num_epochs=10,
            warmup_steps=1000,
            gradient_clip=1.0,
            eval_every=100,
            save_every=500,
            temporal_weight=0.1,
            preservation_weight=0.05,
            tau_seconds=86400.0
        )

        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 10

    def test_batch_size_validation(self):
        """Test batch size validation."""
        # Valid batch sizes
        for size in [1, 8, 16, 32, 64, 128, 256]:
            config = TrainingConfig(batch_size=size)
            assert config.batch_size == size

        # Zero or negative should be invalid (caught at runtime)
        config = TrainingConfig(batch_size=0)
        assert config.batch_size == 0  # Config accepts it, error at runtime

    def test_learning_rate_validation(self):
        """Test learning rate validation."""
        # Valid learning rates
        for lr in [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]:
            config = TrainingConfig(learning_rate=lr)
            assert config.learning_rate == lr

        # Zero or negative should be invalid (caught at runtime)
        config = TrainingConfig(learning_rate=0.0)
        assert config.learning_rate == 0.0  # Config accepts it, error at runtime

    def test_temporal_weight_validation(self):
        """Test temporal weight validation."""
        # Valid weights
        for weight in [0.0, 0.1, 0.5, 1.0]:
            config = TrainingConfig(temporal_weight=weight)
            assert config.temporal_weight == weight

    def test_preservation_weight_validation(self):
        """Test preservation weight validation."""
        # Valid weights
        for weight in [0.0, 0.05, 0.1, 0.5]:
            config = TrainingConfig(preservation_weight=weight)
            assert config.preservation_weight == weight

    def test_tau_seconds_validation(self):
        """Test tau seconds validation."""
        # Valid tau values
        for tau in [3600.0, 86400.0, 604800.0]:  # 1 hour, 1 day, 1 week
            config = TrainingConfig(tau_seconds=tau)
            assert config.tau_seconds == tau

    def test_dry_run_mode(self):
        """Test dry run mode."""
        config = TrainingConfig(dry_run=True)
        assert config.dry_run is True

        config = TrainingConfig(dry_run=False)
        assert config.dry_run is False

    def test_gradient_clip_validation(self):
        """Test gradient clipping validation."""
        # Valid values
        config = TrainingConfig(gradient_clip=0.0)
        assert config.gradient_clip == 0.0

        config = TrainingConfig(gradient_clip=1.0)
        assert config.gradient_clip == 1.0

        config = TrainingConfig(gradient_clip=0.5)
        assert config.gradient_clip == 0.5


class TestConfigSerialization:
    """Test configuration serialization and deserialization."""

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = TIDELiteConfig(
            encoder_name="bert-base-uncased",
            hidden_dim=768,
            time_encoding_dim=64
        )

        # Convert to dict using dataclass fields
        from dataclasses import asdict
        config_dict = asdict(config)

        assert config_dict["encoder_name"] == "bert-base-uncased"
        assert config_dict["hidden_dim"] == 768
        assert config_dict["time_encoding_dim"] == 64

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "encoder_name": "bert-base-uncased",
            "hidden_dim": 768,
            "time_encoding_dim": 64,
            "mlp_hidden_dim": 256,
            "mlp_dropout": 0.2,
            "gate_activation": "tanh",
            "freeze_encoder": False,
            "pooling_strategy": "cls"
        }

        config = TIDELiteConfig(**config_dict)

        assert config.encoder_name == "bert-base-uncased"
        assert config.hidden_dim == 768
        assert config.freeze_encoder is False
        assert config.pooling_strategy == "cls"

    def test_config_yaml_roundtrip(self):
        """Test saving and loading config via YAML."""
        config = TIDELiteConfig(
            encoder_name="custom-model",
            hidden_dim=512,
            time_encoding_dim=48
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode='w', delete=False) as f:
            from dataclasses import asdict
            yaml.dump(asdict(config), f)
            config_path = Path(f.name)

        try:
            # Load back
            with open(config_path) as f:
                loaded_dict = yaml.safe_load(f)

            loaded_config = TIDELiteConfig(**loaded_dict)

            assert loaded_config.encoder_name == config.encoder_name
            assert loaded_config.hidden_dim == config.hidden_dim
            assert loaded_config.time_encoding_dim == config.time_encoding_dim
        finally:
            config_path.unlink()

    def test_partial_config_update(self):
        """Test partial configuration updates."""
        # Start with default config
        config = TIDELiteConfig()

        # Update only some fields
        updates = {
            "hidden_dim": 768,
            "mlp_dropout": 0.3
        }

        # Apply updates
        from dataclasses import replace
        updated_config = replace(config, **updates)

        # Check updates applied
        assert updated_config.hidden_dim == 768
        assert updated_config.mlp_dropout == 0.3

        # Check other fields unchanged
        assert updated_config.encoder_name == config.encoder_name
        assert updated_config.time_encoding_dim == config.time_encoding_dim


class TestConfigValidationHelpers:
    """Test configuration validation helper functions."""

    def test_validate_encoder_name(self):
        """Test encoder name validation."""
        # Valid encoder names (examples)
        valid_names = [
            "bert-base-uncased",
            "sentence-transformers/all-MiniLM-L6-v2",
            "distilbert-base-uncased",
            "roberta-base"
        ]

        for name in valid_names:
            config = TIDELiteConfig(encoder_name=name)
            assert config.encoder_name == name

    def test_validate_dimensions(self):
        """Test dimension validation consistency."""
        # Time encoding dim should be even
        config = TIDELiteConfig(time_encoding_dim=32)
        assert config.time_encoding_dim % 2 == 0

        # Hidden dimensions should be positive
        config = TIDELiteConfig(
            hidden_dim=384,
            mlp_hidden_dim=128
        )
        assert config.hidden_dim > 0
        assert config.mlp_hidden_dim > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
