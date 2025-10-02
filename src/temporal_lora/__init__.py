"""Temporal LoRA: Dynamic Sentence Embeddings via Time-Bucket Adapters.

This package implements temporal LoRA adapters for creating dynamic sentence embeddings
that adapt to semantic drift over time. Key features:
- Time-bucket LoRA adapters on frozen sentence encoders
- Multi-index FAISS retrieval with configurable merge strategies
- Comprehensive evaluation with bootstrap CIs and permutation tests
- Production-grade code with full type hints and tests
"""

__version__ = "0.1.0"
__author__ = "Ali AB"
__email__ = "aliab@example.com"

from temporal_lora.utils.logging import setup_logging
from temporal_lora.utils.seeding import set_seed

__all__ = ["setup_logging", "set_seed", "__version__"]
