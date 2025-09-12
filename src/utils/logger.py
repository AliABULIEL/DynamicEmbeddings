"""
Logging utilities
"""
import logging
import sys
from pathlib import Path
from config.settings import LOG_LEVEL, LOG_FORMAT, LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL))

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, LOG_LEVEL))
        console_formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = LOGS_DIR / f"{name.replace('.', '_')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

    return logger