"""
Seeding utilities for reproducible experiments.

Ensures deterministic behavior across Python, NumPy, PyTorch, and CUDA.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # CUDA deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variable for CUDA
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # PyTorch deterministic algorithms (may impact performance)
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        # Older PyTorch versions
        pass


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Get a NumPy random number generator with optional seed.

    Args:
        seed: Random seed (if None, uses non-deterministic seed)

    Returns:
        NumPy random generator
    """
    return np.random.default_rng(seed)
