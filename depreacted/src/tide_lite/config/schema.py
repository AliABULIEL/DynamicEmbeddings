"""Configuration schema and validation for TIDE-Lite."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import warnings

try:
    from pydantic.dataclasses import dataclass as pydantic_dataclass
    HAS_PYDANTIC = True
except ImportError:
    # Fallback to regular dataclass if pydantic not available
    pydantic_dataclass = dataclass
    HAS_PYDANTIC = False


@pydantic_dataclass if HAS_PYDANTIC else dataclass
class TIDEConfig:
    """Complete TIDE-Lite configuration schema.
    
    This schema validates all configuration parameters for training,
    evaluation, and model architecture.
    """
    
    # Model architecture
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    hidden_dim: int = 384
    time_encoding_dim: int = 32
    mlp_hidden_dim: int = 128
    mlp_dropout: float = 0.1
    freeze_encoder: bool = True
    pooling_strategy: Literal["mean", "cls", "max"] = "mean"
    gate_activation: Literal["sigmoid", "tanh"] = "sigmoid"
    
    # Data configuration
    batch_size: int = 32
    eval_batch_size: int = 64
    max_seq_length: int = 128
    num_workers: int = 2
    
    # Training hyperparameters
    num_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Loss weights
    temporal_weight: float = 0.1
    preservation_weight: float = 0.05
    tau_seconds: float = 86400.0  # 1 day in seconds
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    save_every_n_steps: int = 500
    eval_every_n_steps: int = 500
    
    # Output paths
    output_dir: str = "results/default_run"
    checkpoint_dir: Optional[str] = None
    
    # Hardware
    device: Optional[Literal["cpu", "cuda"]] = None
    
    # Miscellaneous
    seed: int = 42
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    dry_run: bool = False
    temporal_enabled: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Set checkpoint_dir if not provided
        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(Path(self.output_dir) / "checkpoints")
        
        # Validate numeric ranges
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if not 0 <= self.mlp_dropout < 1:
            raise ValueError(f"mlp_dropout must be in [0, 1), got {self.mlp_dropout}")
        
        if self.temporal_weight < 0:
            raise ValueError(f"temporal_weight must be non-negative, got {self.temporal_weight}")
        
        if self.preservation_weight < 0:
            raise ValueError(f"preservation_weight must be non-negative, got {self.preservation_weight}")
        
        # Validate dimensions
        if self.time_encoding_dim % 2 != 0:
            raise ValueError(f"time_encoding_dim must be even, got {self.time_encoding_dim}")
        
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        
        if self.mlp_hidden_dim <= 0:
            raise ValueError(f"mlp_hidden_dim must be positive, got {self.mlp_hidden_dim}")
    
    def to_training_config(self):
        """Convert to TrainingConfig for compatibility."""
        from ..train.trainer import TrainingConfig
        
        # Map fields to TrainingConfig
        return TrainingConfig(
            encoder_name=self.encoder_name,
            hidden_dim=self.hidden_dim,
            time_encoding_dim=self.time_encoding_dim,
            mlp_hidden_dim=self.mlp_hidden_dim,
            mlp_dropout=self.mlp_dropout,
            freeze_encoder=self.freeze_encoder,
            pooling_strategy=self.pooling_strategy,
            gate_activation=self.gate_activation,
            batch_size=self.batch_size,
            eval_batch_size=self.eval_batch_size,
            max_seq_length=self.max_seq_length,
            num_workers=self.num_workers,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            gradient_clip=self.gradient_clip,
            temporal_weight=self.temporal_weight,
            preservation_weight=self.preservation_weight,
            tau_seconds=self.tau_seconds,
            use_amp=self.use_amp,
            save_every_n_steps=self.save_every_n_steps,
            eval_every_n_steps=self.eval_every_n_steps,
            output_dir=self.output_dir,
            checkpoint_dir=self.checkpoint_dir,
            seed=self.seed,
            log_level=self.log_level,
            dry_run=self.dry_run,
            device=self.device,
            temporal_enabled=self.temporal_enabled,
        )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TIDEConfig":
        """Create config from dictionary, filtering unknown keys.
        
        Args:
            config_dict: Configuration dictionary.
            
        Returns:
            TIDEConfig instance.
        """
        # Get valid field names
        if HAS_PYDANTIC:
            try:
                valid_fields = set(cls.__fields__.keys())
            except AttributeError:
                # Fallback if __fields__ doesn't exist
                import dataclasses
                valid_fields = set(f.name for f in dataclasses.fields(cls))
        else:
            # For regular dataclass
            import dataclasses
            valid_fields = set(f.name for f in dataclasses.fields(cls))
        
        # Filter unknown keys
        unknown_keys = set(config_dict.keys()) - valid_fields
        if unknown_keys:
            warnings.warn(
                f"Ignoring unknown configuration keys: {sorted(unknown_keys)}",
                UserWarning
            )
        
        # Create config with only valid fields
        valid_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**valid_config)


@dataclass
class EvalConfig:
    """Configuration for evaluation runs."""
    
    model_path: str
    batch_size: int = 64
    max_seq_length: int = 128
    device: Optional[str] = None
    output_dir: str = "results/eval"
    tasks: list = field(default_factory=lambda: ["stsb", "quora", "temporal"])
    use_temporal: bool = True
    num_workers: int = 2
    seed: int = 42
    log_level: str = "INFO"
