"""Training utilities for TIDE-Lite."""

from .losses import (
    cosine_regression_loss,
    temporal_consistency_loss,
    combined_tide_loss,
)
from .trainer import TIDETrainer, TrainingConfig

__all__ = [
    "cosine_regression_loss",
    "temporal_consistency_loss",
    "combined_tide_loss",
    "TIDETrainer",
    "TrainingConfig",
]
