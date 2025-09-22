"""Command-line interfaces for TIDE-Lite."""

from .train_cli import main as train_main
from .tide import main as tide_main
from .aggregate_cli import main as aggregate_main
from .plots_cli import main as plots_main

__all__ = [
    "train_main",
    "tide_main",
    "aggregate_main",
    "plots_main",
]
