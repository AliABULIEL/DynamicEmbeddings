"""Reproducibility utilities for deterministic runs."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Use deterministic algorithms (may be slower)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Make PyTorch deterministic
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for PyTorch operations
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        # Enable deterministic algorithms (may fail for some ops)
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some operations don't have deterministic implementations
            pass


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """Get a PyTorch random generator with optional seed.
    
    Args:
        seed: Random seed. If None, use default generator.
    
    Returns:
        torch.Generator: Random generator
    """
    if seed is None:
        return torch.default_generator
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
