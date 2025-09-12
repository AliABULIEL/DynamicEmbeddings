"""
Improved logger configuration
src/utils/logger.py
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

# Suppress duplicate logging
logging.getLogger().handlers = []

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output"""

    grey = "\x1b[38;21m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey + "%(message)s" + reset,
        logging.INFO: green + "%(message)s" + reset,
        logging.WARNING: yellow + "⚠ %(message)s" + reset,
        logging.ERROR: red + "✗ %(message)s" + reset,
        logging.CRITICAL: bold_red + "✗✗ %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a clean, readable logger

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)

    # Prevent duplicate logs
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level))
    logger.propagate = False  # Prevent duplicate logs

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))

    # Use colored formatter for better readability
    console_handler.setFormatter(ColoredFormatter())

    logger.addHandler(console_handler)

    return logger


# Suppress warnings from libraries
def suppress_warnings():
    """Suppress annoying warnings from libraries"""
    import warnings
    import os

    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Suppress transformers warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

    # Suppress sentence-transformers warnings
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    # Suppress general warnings
    warnings.filterwarnings("ignore")

    # Suppress CUDA warnings if needed
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")