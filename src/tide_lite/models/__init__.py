"""Model implementations for TIDE-Lite with unified embedding API.

All models expose the same interface:
- encode_texts(texts: List[str], batch_size: int) -> torch.Tensor
- embedding_dim property
- count_extra_parameters() method
"""

from .tide_lite import TIDELite, TIDELiteConfig
from .baselines import (
    BaselineEncoder,
    MiniLMBaseline,
    E5BaseBaseline,
    BGEBaseBaseline,
    load_baseline,
    BaselineComparison,
    create_baseline_suite,
)

__all__ = [
    # TIDE-Lite
    "TIDELite",
    "TIDELiteConfig",
    # Baselines
    "BaselineEncoder",
    "MiniLMBaseline",
    "E5BaseBaseline",
    "BGEBaseBaseline",
    "load_baseline",
    # Utilities
    "BaselineComparison",
    "create_baseline_suite",
]
