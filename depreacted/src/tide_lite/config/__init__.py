"""Configuration schema and validation for TIDE-Lite."""

from .schema import TIDEConfig, EvalConfig, HAS_PYDANTIC
from .config_validate import (
    load_and_validate_config,
    validate_config_dict,
    write_normalized_config,
    preflight_check,
    get_unknown_keys,
)

__all__ = [
    "TIDEConfig",
    "EvalConfig",
    "HAS_PYDANTIC",
    "load_and_validate_config",
    "validate_config_dict",
    "write_normalized_config",
    "preflight_check",
    "get_unknown_keys",
]
