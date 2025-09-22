"""Command-line interfaces for TIDE-Lite."""

from .train_cli import main as train_main
from .tide import main as tide_main

__all__ = ["train_main", "tide_main"]
