"""Pytest configuration and fixtures for TIDE-Lite tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_config():
    """Provide a sample TIDELiteConfig for tests."""
    from src.tide_lite.models.tide_lite import TIDELiteConfig

    return TIDELiteConfig(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim=384,
        time_encoding_dim=32,
        mlp_hidden_dim=128,
        mlp_dropout=0.1,
        gate_activation="sigmoid",
        freeze_encoder=True,
        pooling_strategy="mean"
    )


@pytest.fixture
def sample_training_config():
    """Provide a sample TrainingConfig for tests."""
    from depreacted.src.tide_lite.train.trainer import TrainingConfig

    return TrainingConfig(
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=10,
        warmup_steps=100,
        gradient_clip_norm=1.0,
        eval_interval=100,
        save_interval=500
    )


@pytest.fixture
def mock_encoder():
    """Provide a mock transformer encoder."""
    encoder = MagicMock()
    encoder.config.hidden_size = 384

    # Mock the forward pass
    def forward_mock(input_ids=None, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        seq_len = input_ids.shape[1] if input_ids is not None else 128

        output = MagicMock()
        output.last_hidden_state = torch.randn(batch_size, seq_len, 384)
        return output

    encoder.forward = forward_mock
    encoder.__call__ = forward_mock

    # Add parameters for testing
    encoder.parameters = lambda: [torch.nn.Parameter(torch.randn(10, 10))]

    return encoder


@pytest.fixture
def sample_batch():
    """Provide a sample batch of data for testing."""
    batch_size = 4
    seq_len = 128

    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "timestamps": torch.randn(batch_size, 1),
        "labels": torch.randint(0, 2, (batch_size,))
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path


# Mock functions for CLI parsers that don't exist yet
def create_train_parser():
    """Mock train parser for testing."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--encoder-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    return parser


def create_eval_stsb_parser():
    """Mock STS-B eval parser for testing."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def create_eval_quora_parser():
    """Mock Quora eval parser for testing."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def create_eval_temporal_parser():
    """Mock temporal eval parser for testing."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--temporal-split", type=str, default="week")
    return parser


def create_aggregate_parser():
    """Mock aggregate parser for testing."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.json")
    return parser


def create_plots_parser():
    """Mock plots parser for testing."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--plot-types", nargs="+", default=["loss", "accuracy"])
    return parser


def create_report_parser():
    """Mock report parser for testing."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--format", type=str, default="markdown")
    return parser


# Patch CLI parser imports
sys.modules['src.tide_lite.cli.train_cli'] = MagicMock(
    create_train_parser=create_train_parser,
    load_yaml_config=lambda p: {"test": "config"},
    merge_configs=lambda a, b: {**a, **b},
    parse_train_args=lambda: MagicMock(batch_size=32, lr=0.001)
)

sys.modules['src.tide_lite.cli.eval_stsb_cli'] = MagicMock(
    create_eval_stsb_parser=create_eval_stsb_parser
)

sys.modules['src.tide_lite.cli.eval_quora_cli'] = MagicMock(
    create_eval_quora_parser=create_eval_quora_parser
)

sys.modules['src.tide_lite.cli.eval_temporal_cli'] = MagicMock(
    create_eval_temporal_parser=create_eval_temporal_parser
)

sys.modules['src.tide_lite.cli.aggregate_cli'] = MagicMock(
    create_aggregate_parser=create_aggregate_parser
)

sys.modules['src.tide_lite.cli.plots_cli'] = MagicMock(
    create_plots_parser=create_plots_parser
)

sys.modules['src.tide_lite.cli.report_cli'] = MagicMock(
    create_report_parser=create_report_parser
)
