"""
Logging utilities for Temporal LoRA.

Provides structured logging with loguru for training, evaluation, and debugging.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from temporal_lora.utils.paths import get_logs_dir


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure loguru logger with file and console output.

    Args:
        log_file: Path to log file (relative to logs dir). If None, no file logging.
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        rotation: When to rotate log files (size or time)
        retention: How long to keep old log files
    """
    # Remove default handler
    logger.remove()

    # Add console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file is not None:
        log_path = get_logs_dir() / log_file
        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )
        logger.info(f"Logging to file: {log_path}")


def get_logger(name: str):
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Loguru logger instance
    """
    return logger.bind(name=name)
