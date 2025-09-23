"""Common utilities for TIDE-Lite.

This module provides centralized utility functions for pooling, time encoding,
similarity computation, logging, and other common operations.
"""

import json
import logging
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ============================================================================
# POOLING UTILITIES
# ============================================================================

def mean_pool(
    last_hidden_state: Tensor,
    attention_mask: Tensor
) -> Tensor:
    """Apply mean pooling to token embeddings.
    
    Args:
        last_hidden_state: Token embeddings [batch_size, seq_len, hidden_dim]
        attention_mask: Attention mask [batch_size, seq_len]
        
    Returns:
        Pooled embeddings [batch_size, hidden_dim]
        
    Note:
        Performs mean pooling over valid tokens (non-masked positions),
        handling padding tokens correctly.
    """
    # Expand mask to match hidden state dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    # Sum embeddings for valid tokens
    sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
    
    # Count valid tokens per sample
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    
    # Compute mean
    return sum_embeddings / sum_mask


def max_pool(
    last_hidden_state: Tensor,
    attention_mask: Tensor
) -> Tensor:
    """Apply max pooling to token embeddings.
    
    Args:
        last_hidden_state: Token embeddings [batch_size, seq_len, hidden_dim]
        attention_mask: Attention mask [batch_size, seq_len]
        
    Returns:
        Pooled embeddings [batch_size, hidden_dim]
    """
    # Set padding tokens to large negative value
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    hidden_states_masked = last_hidden_state.clone()
    hidden_states_masked[~mask_expanded.bool()] = -1e9
    
    # Max pool
    return hidden_states_masked.max(dim=1)[0]


def cls_pool(last_hidden_state: Tensor) -> Tensor:
    """Extract CLS token embedding.
    
    Args:
        last_hidden_state: Token embeddings [batch_size, seq_len, hidden_dim]
        
    Returns:
        CLS embeddings [batch_size, hidden_dim]
    """
    return last_hidden_state[:, 0, :]


# ============================================================================
# TIME ENCODING
# ============================================================================

def sinusoidal_time_encoding(
    timestamps: Tensor,
    dims: int = 32,
    max_period: float = 10000.0
) -> Tensor:
    """Generate sinusoidal time encodings.
    
    Args:
        timestamps: Unix timestamps [batch_size] or [batch_size, 1]
        dims: Dimension of encoding (must be even)
        max_period: Maximum period for sinusoidal functions
        
    Returns:
        Time encodings [batch_size, dims]
        
    Note:
        Similar to positional encodings in Transformers but for continuous time.
        Uses sin/cos pairs with exponentially increasing periods.
    """
    if dims % 2 != 0:
        raise ValueError(f"dims must be even, got {dims}")
    
    # Ensure correct shape
    if timestamps.dim() == 1:
        timestamps = timestamps.unsqueeze(-1)  # [batch_size, 1]
    
    # Convert to float
    timestamps = timestamps.float()
    
    # Create frequency scales
    half_dims = dims // 2
    device = timestamps.device
    scales = torch.pow(
        max_period,
        -torch.arange(half_dims, dtype=torch.float32, device=device) / half_dims
    )
    
    # Apply scales to timestamps
    scaled_time = timestamps * scales  # [batch_size, half_dims]
    
    # Apply sin and cos
    sin_enc = torch.sin(scaled_time)
    cos_enc = torch.cos(scaled_time)
    
    # Concatenate
    encoding = torch.cat([sin_enc, cos_enc], dim=-1)
    
    return encoding


# ============================================================================
# SIMILARITY UTILITIES
# ============================================================================

def cosine_similarity_matrix(
    x: Tensor,
    y: Optional[Tensor] = None,
    eps: float = 1e-8
) -> Tensor:
    """Compute pairwise cosine similarity matrix.
    
    Args:
        x: First set of embeddings [batch_size_x, hidden_dim]
        y: Second set of embeddings [batch_size_y, hidden_dim] or None
        eps: Small value for numerical stability
        
    Returns:
        Similarity matrix [batch_size_x, batch_size_y] or [batch_size_x, batch_size_x]
    """
    # Normalize embeddings
    x_norm = F.normalize(x, p=2, dim=1, eps=eps)
    
    if y is None:
        # Self-similarity
        sim_matrix = torch.mm(x_norm, x_norm.t())
    else:
        # Cross-similarity
        y_norm = F.normalize(y, p=2, dim=1, eps=eps)
        sim_matrix = torch.mm(x_norm, y_norm.t())
    
    return sim_matrix


def euclidean_distance_matrix(
    x: Tensor,
    y: Optional[Tensor] = None
) -> Tensor:
    """Compute pairwise Euclidean distance matrix.
    
    Args:
        x: First set of embeddings [batch_size_x, hidden_dim]
        y: Second set of embeddings [batch_size_y, hidden_dim] or None
        
    Returns:
        Distance matrix [batch_size_x, batch_size_y] or [batch_size_x, batch_size_x]
    """
    if y is None:
        y = x
    
    # Use cdist for efficient computation
    return torch.cdist(x, y, p=2)


# ============================================================================
# RANDOM SEED MANAGEMENT
# ============================================================================

def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        
    Note:
        Sets seeds for Python random, NumPy, and PyTorch.
        Some operations may still be non-deterministic (e.g., certain CUDA ops).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For newer PyTorch versions
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some operations don't have deterministic implementations
            torch.use_deterministic_algorithms(True, warn_only=True)
    
    logger.debug(f"Set global random seed to {seed}")


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class JSONLogger:
    """Lightweight JSON logger for metrics tracking.
    
    Writes JSON lines (one dict per line) for easy parsing and appending.
    """
    
    def __init__(self, log_path: Union[str, Path], mode: str = "a"):
        """Initialize JSON logger.
        
        Args:
            log_path: Path to JSON log file
            mode: File mode ('w' for write, 'a' for append)
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if mode == "w" and self.log_path.exists():
            self.log_path.unlink()
        
        logger.debug(f"JSONLogger initialized: {self.log_path}")
    
    def log(self, data: Dict[str, Any], timestamp: bool = True) -> None:
        """Log data to JSON file (append-safe).
        
        Args:
            data: Dictionary to log
            timestamp: Whether to add timestamp
        """
        if timestamp and "timestamp" not in data:
            data = {"timestamp": datetime.now().isoformat(), **data}
        
        # Write as JSON line
        with open(self.log_path, "a") as f:
            json.dump(data, f, default=str)
            f.write("\n")
    
    def read(self) -> List[Dict[str, Any]]:
        """Read all logs from file.
        
        Returns:
            List of logged dictionaries
        """
        if not self.log_path.exists():
            return []
        
        logs = []
        with open(self.log_path) as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        return logs


class Timer:
    """Context manager and utility for timing operations.
    
    Example:
        >>> with Timer("forward pass") as t:
        ...     output = model(input)
        ... print(f"Took {t.elapsed:.3f}s")
    """
    
    def __init__(self, name: Optional[str] = None, verbose: bool = False):
        """Initialize timer.
        
        Args:
            name: Name of operation being timed
            verbose: Whether to print on completion
        """
        self.name = name
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0
    
    def __enter__(self) -> "Timer":
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Stop timing and optionally print."""
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            if self.verbose and self.name:
                logger.info(f"{self.name}: {self.elapsed:.4f} seconds")
    
    def reset(self) -> None:
        """Reset timer."""
        self.start_time = None
        self.elapsed = 0.0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_device(device: Optional[str] = None) -> torch.device:
    """Get torch device with auto-detection.
    
    Args:
        device: Device string ('cuda', 'cpu', 'mps', 'auto') or None
        
    Returns:
        Torch device
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    return torch.device(device)


def count_parameters(
    model: torch.nn.Module,
    trainable_only: bool = True
) -> Dict[str, int]:
    """Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def format_metrics(
    metrics: Dict[str, Any],
    precision: int = 4
) -> str:
    """Format metrics dictionary as readable string.
    
    Args:
        metrics: Dictionary of metric values
        precision: Decimal precision for floats
        
    Returns:
        Formatted string
    """
    items = []
    for key, value in metrics.items():
        if isinstance(value, float):
            items.append(f"{key}: {value:.{precision}f}")
        else:
            items.append(f"{key}: {value}")
    return " | ".join(items)


@contextmanager
def timer(name: Optional[str] = None) -> Generator[Timer, None, None]:
    """Context manager for timing operations.
    
    Args:
        name: Name of operation
        
    Yields:
        Timer instance
    """
    t = Timer(name, verbose=False)
    try:
        yield t.__enter__()
    finally:
        t.__exit__()


def save_json(
    data: Any,
    path: Union[str, Path],
    indent: int = 2
) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        path: Path to JSON file
        indent: Indentation for pretty printing
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """Load data from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(path) as f:
        return json.load(f)


class EarlyStopping:
    """Early stopping callback for training.
    
    Example:
        >>> early_stop = EarlyStopping(patience=3, mode="max")
        >>> for epoch in range(100):
        ...     val_score = evaluate()
        ...     if early_stop(val_score):
        ...         break
    """
    
    def __init__(
        self,
        patience: int = 5,
        mode: str = "max",
        min_delta: float = 0.0
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            mode: "max" for maximizing, "min" for minimizing
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if should stop.
        
        Args:
            score: Current score
            
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.patience} epochs")
        
        return self.should_stop
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
