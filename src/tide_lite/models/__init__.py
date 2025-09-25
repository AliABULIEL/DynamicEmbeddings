"""Model implementations for TIDE-Lite with unified embedding API.

All models expose the same interface:
- encode_texts(texts: List[str], batch_size: int) -> torch.Tensor
- embedding_dim property
- count_extra_parameters() method
"""

from .tide_lite import TIDELite, TIDELiteConfig
from .baselines import (
    BaselineEncoder,
    load_baseline,
    load_minilm_baseline,
    load_e5_base_baseline,
    load_bge_base_baseline,
)

__all__ = [
    # TIDE-Lite
    "TIDELite",
    "TIDELiteConfig",
    # Baselines
    "BaselineEncoder",
    "load_baseline",
    "load_minilm_baseline",
    "load_e5_base_baseline",
    "load_bge_base_baseline",
]
