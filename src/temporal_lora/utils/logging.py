"""Logging utilities with rich formatting."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


# Global console for rich output
console = Console()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rich_tracebacks: bool = True,
) -> None:
    """Setup logging with rich formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
        rich_tracebacks: Use rich formatting for tracebacks
    """
    # Configure rich handler
    handlers: list[logging.Handler] = [
        RichHandler(
            console=console,
            rich_tracebacks=rich_tracebacks,
            tracebacks_show_locals=False,
            markup=True,
        )
    ]
    
    # Add file handler if specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


def log_section(title: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a section header with rich formatting.
    
    Args:
        title: Section title
        logger: Logger instance. If None, use console.
    """
    separator = "=" * 80
    if logger is not None:
        logger.info(f"\n{separator}")
        logger.info(f"  {title}")
        logger.info(f"{separator}\n")
    else:
        console.rule(f"[bold blue]{title}[/bold blue]")
