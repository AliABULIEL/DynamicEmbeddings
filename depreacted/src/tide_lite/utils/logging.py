"""Logging utilities for TIDE-Lite."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


class JSONLogger:
    """Append-safe JSON logger for experiment tracking."""
    
    def __init__(self, log_path: Union[str, Path], append: bool = True):
        """Initialize JSON logger.
        
        Args:
            log_path: Path to JSON log file
            append: Whether to append to existing file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not append and self.log_path.exists():
            self.log_path.unlink()
    
    def log(self, data: Dict[str, Any]) -> None:
        """Log data to JSON file (append-safe).
        
        Args:
            data: Dictionary to log
        """
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
        
        # Write as JSON line (append-safe)
        with open(self.log_path, "a") as f:
            json.dump(data, f)
            f.write("\n")
    
    def read_logs(self) -> list[Dict[str, Any]]:
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


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format message
        message = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return message


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    name: Optional[str] = None,
    colored: bool = True
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional file to log to
        name: Logger name
        colored: Whether to use colored output
        
    Returns:
        Configured logger
    """
    # Get or create logger
    logger = logging.getLogger(name or "tide_lite")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Format
    if colored and sys.stdout.isatty():
        formatter = ColoredFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricLogger:
    """Logger for tracking metrics during training/evaluation."""
    
    def __init__(self, json_path: Optional[Union[str, Path]] = None):
        """Initialize metric logger.
        
        Args:
            json_path: Optional path for JSON logging
        """
        self.metrics: Dict[str, list] = {}
        self.json_logger = JSONLogger(json_path) if json_path else None
    
    def update(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Update metrics.
        
        Args:
            metrics: Dictionary of metric values
            step: Optional step/epoch number
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Log to JSON if available
        if self.json_logger:
            log_data = {"step": step} if step is not None else {}
            log_data.update(metrics)
            self.json_logger.log(log_data)
    
    def get_last(self, metric: str, default: Any = None) -> Any:
        """Get last value of a metric.
        
        Args:
            metric: Metric name
            default: Default value if metric not found
            
        Returns:
            Last metric value
        """
        if metric in self.metrics and self.metrics[metric]:
            return self.metrics[metric][-1]
        return default
    
    def get_best(self, metric: str, mode: str = "max") -> Optional[float]:
        """Get best value of a metric.
        
        Args:
            metric: Metric name
            mode: "max" or "min"
            
        Returns:
            Best metric value
        """
        if metric not in self.metrics or not self.metrics[metric]:
            return None
        
        values = self.metrics[metric]
        return max(values) if mode == "max" else min(values)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics.
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        import numpy as np
        
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "last": values[-1],
                }
        return summary


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
