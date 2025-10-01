"""Configuration validation utilities."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set

import yaml

from .schema import TIDEConfig, HAS_PYDANTIC

logger = logging.getLogger(__name__)


def load_and_validate_config(
    config_path: Path,
    strict: bool = False
) -> TIDEConfig:
    """Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
        strict: If True, fail on unknown keys. If False, warn and continue.
        
    Returns:
        Validated TIDEConfig instance.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
        ValueError: If validation fails in strict mode.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        raw_config = {}
    
    # Get valid field names from schema
    if HAS_PYDANTIC:
        try:
            from .schema import TIDEConfig
            valid_fields = set(TIDEConfig.__fields__.keys())
        except AttributeError:
            # Fallback if __fields__ doesn't exist
            import dataclasses
            from .schema import TIDEConfig
            valid_fields = set(f.name for f in dataclasses.fields(TIDEConfig))
    else:
        import dataclasses
        from .schema import TIDEConfig
        valid_fields = set(f.name for f in dataclasses.fields(TIDEConfig))
    
    # Check for unknown keys
    unknown_keys = set(raw_config.keys()) - valid_fields
    
    if unknown_keys:
        message = f"Unknown configuration keys: {sorted(unknown_keys)}"
        if strict:
            raise ValueError(message)
        else:
            logger.warning(message)
            print(f"⚠️  Warning: {message}", file=sys.stderr)
    
    # Create and validate config
    try:
        config = TIDEConfig.from_dict(raw_config)
        logger.info(f"Successfully validated config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def validate_config_dict(
    config_dict: Dict[str, Any],
    strict: bool = False
) -> TIDEConfig:
    """Validate a configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary.
        strict: If True, fail on unknown keys.
        
    Returns:
        Validated TIDEConfig instance.
        
    Raises:
        ValueError: If validation fails.
    """
    # Get valid field names
    if HAS_PYDANTIC:
        try:
            from .schema import TIDEConfig
            valid_fields = set(TIDEConfig.__fields__.keys())
        except AttributeError:
            # Fallback if __fields__ doesn't exist
            import dataclasses
            from .schema import TIDEConfig
            valid_fields = set(f.name for f in dataclasses.fields(TIDEConfig))
    else:
        import dataclasses
        from .schema import TIDEConfig
        valid_fields = set(f.name for f in dataclasses.fields(TIDEConfig))
    
    # Check for unknown keys
    unknown_keys = set(config_dict.keys()) - valid_fields
    
    if unknown_keys:
        message = f"Unknown configuration keys: {sorted(unknown_keys)}"
        if strict:
            raise ValueError(message)
        else:
            logger.warning(message)
    
    # Create and validate config
    return TIDEConfig.from_dict(config_dict)


def write_normalized_config(
    config: TIDEConfig,
    output_path: Path,
    format: str = "yaml"
) -> None:
    """Write normalized configuration to file.
    
    Args:
        config: Validated configuration.
        output_path: Output file path.
        format: Output format ("yaml" or "json").
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary
    if hasattr(config, "__dict__"):
        config_dict = vars(config)
    else:
        # For pydantic dataclass
        config_dict = {f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()}
    
    # Write based on format
    if format.lower() == "json":
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info(f"Wrote normalized config to {output_path} (JSON)")
    else:  # yaml
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
        logger.info(f"Wrote normalized config to {output_path} (YAML)")


def preflight_check(config_path: Path) -> bool:
    """Run preflight validation check on configuration.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        True if validation passes, False otherwise.
    """
    try:
        # Load and validate
        config = load_and_validate_config(config_path, strict=False)
        
        # Write normalized version
        output_dir = Path("outputs/QA")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        normalized_path = output_dir / "run_config_resolved.yaml"
        write_normalized_config(config, normalized_path, format="yaml")
        
        # Print summary
        print("\n" + "=" * 60)
        print("✅ CONFIGURATION VALIDATED")
        print("=" * 60)
        print(f"Input:      {config_path}")
        print(f"Normalized: {normalized_path}")
        print(f"Model:      {config.encoder_name}")
        print(f"Epochs:     {config.num_epochs}")
        print(f"Batch size: {config.batch_size}")
        print(f"Output dir: {config.output_dir}")
        print("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration validation failed: {e}", file=sys.stderr)
        return False


def get_unknown_keys(config_path: Path) -> Set[str]:
    """Get list of unknown keys in config file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Set of unknown key names.
    """
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f) or {}
    
    # Get valid fields
    if HAS_PYDANTIC:
        try:
            from .schema import TIDEConfig
            valid_fields = set(TIDEConfig.__fields__.keys())
        except AttributeError:
            # Fallback if __fields__ doesn't exist
            import dataclasses
            from .schema import TIDEConfig
            valid_fields = set(f.name for f in dataclasses.fields(TIDEConfig))
    else:
        import dataclasses
        from .schema import TIDEConfig
        valid_fields = set(f.name for f in dataclasses.fields(TIDEConfig))
    
    return set(raw_config.keys()) - valid_fields
