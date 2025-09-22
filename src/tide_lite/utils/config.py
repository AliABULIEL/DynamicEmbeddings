"""Configuration utilities for TIDE-Lite.

Provides logging setup and global seed management for reproducibility.
"""

import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        format_string: Custom format string for log messages.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )
    
    # Set levels for common noisy libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at level {level}")
    if log_file:
        logger.info(f"Logging to file: {log_file}")


def set_global_seed(seed: int) -> None:
    """Set random seed for all libraries to ensure reproducibility.
    
    Args:
        seed: Random seed value.
        
    Note:
        This sets seeds for:
        - Python's random module
        - NumPy
        - PyTorch (CPU and CUDA)
        
        Some operations may still be non-deterministic:
        - CUDA atomic operations
        - Some CuDNN operations
        - Dataloader with multiple workers
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Try to make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger = logging.getLogger(__name__)
    logger.info(f"Global random seed set to {seed}")
    logger.debug(
        "Note: Some CUDA operations may still be non-deterministic. "
        "See PyTorch reproducibility guide for details."
    )
