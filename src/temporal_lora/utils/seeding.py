"""Seeding utilities for reproducibility."""

import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Try to set torch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Additional settings for deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Get a numpy random generator with optional seed.
    
    Args:
        seed: Optional seed for the generator.
        
    Returns:
        Numpy random generator.
    """
    return np.random.default_rng(seed)
