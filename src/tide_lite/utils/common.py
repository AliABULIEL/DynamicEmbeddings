"""Common utilities for TIDE-Lite."""

import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    
    Note:
        This sets seeds for Python random, NumPy, and PyTorch.
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
    
    # For PyTorch 1.8+
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


class Timer:
    """Context manager and utility for timing operations."""
    
    def __init__(self, name: Optional[str] = None, verbose: bool = True):
        """Initialize timer.
        
        Args:
            name: Name of the operation being timed
            verbose: Whether to print timing results
        """
        self.name = name
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0
    
    def start(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        
        if self.verbose:
            name = f"{self.name}: " if self.name else ""
            print(f"{name}{self.elapsed:.4f} seconds")
        
        return self.elapsed
    
    def __enter__(self) -> "Timer":
        """Enter context manager."""
        self.start()
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        self.stop()


@contextmanager
def timer(name: Optional[str] = None) -> Generator[Timer, None, None]:
    """Context manager for timing operations.
    
    Args:
        name: Name of the operation
        
    Yields:
        Timer instance
        
    Example:
        >>> with timer("forward pass") as t:
        ...     output = model(input)
        ... print(f"Took {t.elapsed:.3f}s")
    """
    t = Timer(name, verbose=False)
    t.start()
    try:
        yield t
    finally:
        t.stop()


def mean_pooling(
    token_embeddings: Tensor,
    attention_mask: Tensor
) -> Tensor:
    """Apply mean pooling to token embeddings.
    
    Args:
        token_embeddings: Token embeddings [batch_size, seq_len, hidden_dim]
        attention_mask: Attention mask [batch_size, seq_len]
        
    Returns:
        Pooled embeddings [batch_size, hidden_dim]
        
    Note:
        This performs mean pooling over valid tokens (non-masked positions).
    """
    # Implementation will be provided in models module
    raise NotImplementedError("Implement in models.pooling module")


def sinusoidal_time_encoding(
    timestamps: Tensor,
    dim: int,
    max_period: float = 10000.0
) -> Tensor:
    """Generate sinusoidal time encodings.
    
    Args:
        timestamps: Timestamps tensor [batch_size] or [batch_size, 1]
        dim: Dimension of the encoding (must be even)
        max_period: Maximum period for sinusoidal functions
        
    Returns:
        Time encodings [batch_size, dim]
        
    Note:
        Similar to positional encodings in Transformers but for continuous time.
        Uses sin/cos pairs with exponentially increasing periods.
    """
    # Implementation will be provided in models module
    raise NotImplementedError("Implement in models.temporal module")


def create_output_dir(
    base_dir: Union[str, Path],
    experiment_name: str,
    timestamp: bool = True
) -> Path:
    """Create output directory for experiment.
    
    Args:
        base_dir: Base output directory
        experiment_name: Name of the experiment
        timestamp: Whether to add timestamp to directory name
        
    Returns:
        Path to created directory
    """
    base_dir = Path(base_dir)
    
    if timestamp:
        from datetime import datetime
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{experiment_name}_{time_str}"
    else:
        dir_name = experiment_name
    
    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        path: Path to JSON file
        indent: Indentation for pretty printing
    """
    import json
    
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
    import json
    
    with open(path) as f:
        return json.load(f)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get torch device.
    
    Args:
        device: Device string (cuda, cpu, mps, auto) or None for auto
        
    Returns:
        Torch device
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    return torch.device(device)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dictionary as string.
    
    Args:
        metrics: Dictionary of metric values
        precision: Decimal precision
        
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


class EarlyStopping:
    """Early stopping callback."""
    
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
        
        return self.should_stop
