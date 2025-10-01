"""Training utilities for TIDE-Lite."""

from .losses import (
    cosine_regression_loss,
    temporal_consistency_loss,
    preservation_loss,
    combined_tide_loss,
    TIDELiteLoss,
)
from .trainer import (
    TIDELiteTrainer,
    TrainingConfig,
    train_tide_lite,
)

__all__ = [
    # Losses
    "cosine_regression_loss",
    "temporal_consistency_loss",
    "preservation_loss",
    "combined_tide_loss",
    "TIDELiteLoss",
    # Trainer
    "TIDELiteTrainer",
    "TrainingConfig",
    "train_tide_lite",
]
