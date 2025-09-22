"""TIDE-Lite: Temporally-Indexed Dynamic Embeddings model.

This module implements the core TIDE-Lite architecture with a frozen encoder
and lightweight temporal modulation via MLP-based gating.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass
class TIDELiteConfig:
    """Configuration for TIDE-Lite model.
    
    Attributes:
        encoder_name: HuggingFace model name for the frozen encoder.
        hidden_dim: Dimension of encoder hidden states.
        time_encoding_dim: Dimension of sinusoidal time encoding.
        mlp_hidden_dim: Hidden dimension of temporal MLP.
        mlp_dropout: Dropout probability in MLP.
        gate_activation: Activation function for gating ('sigmoid' or 'tanh').
        freeze_encoder: Whether to freeze encoder parameters.
        pooling_strategy: How to pool encoder outputs ('mean', 'cls', 'max').
    """
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    hidden_dim: int = 384
    time_encoding_dim: int = 32
    mlp_hidden_dim: int = 128
    mlp_dropout: float = 0.1
    gate_activation: str = "sigmoid"
    freeze_encoder: bool = True
    pooling_strategy: str = "mean"


def _pool_embeddings(hidden_states, attention_mask, strategy="mean"):
    """Pool token embeddings based on strategy.
    
    Args:
        hidden_states: Token embeddings [batch_size, seq_len, hidden_dim]
        attention_mask: Attention mask [batch_size, seq_len]
        strategy: Pooling strategy ("mean", "cls", "max")
    
    Returns:
        Pooled embeddings [batch_size, hidden_dim]
    """
    if strategy == "cls":
        return hidden_states[:, 0, :]
    elif strategy == "max":
        # Mask out padding tokens with -inf
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        hidden_states[~mask_expanded.bool()] = -1e9
        return hidden_states.max(dim=1)[0]
    else:  # mean
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        return sum_embeddings / sum_mask.clamp(min=1e-9)


class SinusoidalTimeEncoding(nn.Module):
    """Sinusoidal position encoding adapted for timestamps.
    
    Converts Unix timestamps into sinusoidal encodings similar to
    positional encodings in Transformers, but for absolute time values.
    """
    
    def __init__(self, encoding_dim: int = 32) -> None:
        """Initialize time encoding.
        
        Args:
            encoding_dim: Dimension of output encoding (must be even).
            
        Raises:
            ValueError: If encoding_dim is odd.
        """
        super().__init__()
        if encoding_dim % 2 != 0:
            raise ValueError(f"encoding_dim must be even, got {encoding_dim}")
        
        self.encoding_dim = encoding_dim
        
        # Create fixed frequency scales (learnable=False for determinism)
        scales = 2 ** torch.arange(encoding_dim // 2, dtype=torch.float32)
        self.register_buffer("scales", scales)
        
        logger.debug(f"Initialized sinusoidal time encoding with dim={encoding_dim}")
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps into sinusoidal features.
        
        Args:
            timestamps: Unix timestamps [batch_size, 1] or [batch_size].
            
        Returns:
            Time encodings [batch_size, encoding_dim].
        """
        # Ensure timestamps have correct shape
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1)
        
        # Scale timestamps by different frequencies
        scaled_time = timestamps / self.scales  # [batch_size, encoding_dim//2]
        
        # Apply sin and cos
        sin_enc = torch.sin(scaled_time)
        cos_enc = torch.cos(scaled_time)
        
        # Concatenate sin and cos encodings
        encoding = torch.cat([sin_enc, cos_enc], dim=-1)
        
        return encoding


class TemporalGatingMLP(nn.Module):
    """MLP for generating temporal gates.
    
    A simple feedforward network that maps time encodings to gate values.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "sigmoid"
    ) -> None:
        """Initialize temporal gating MLP.
        
        Args:
            input_dim: Dimension of input (time encoding).
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (should match encoder hidden_dim).
            dropout: Dropout probability.
            activation: Final activation ('sigmoid' or 'tanh').
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            timestamps: Unix timestamps in seconds [batch_size] or [batch_size, 1].
            
        Returns:
            Encoded timestamps [batch_size, encoding_dim].
        """
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1)
        elif timestamps.dim() > 2:
            raise ValueError(f"Expected 1D or 2D timestamps, got shape {timestamps.shape}")
        
        # Normalize timestamps to reasonable range (days since epoch)
        normalized_time = timestamps / 86400.0  # Convert to days
        
        # Apply multi-scale sinusoidal encoding
        scaled_time = normalized_time / self.scales  # [batch, encoding_dim//2]
        
        # Concatenate sin and cos features
        encoding = torch.cat([
            torch.sin(scaled_time),
            torch.cos(scaled_time),
        ], dim=-1)
        
        return encoding


class TemporalGatingMLP(nn.Module):
    """MLP that generates gating values from temporal encodings.
    
    This module learns to modulate embeddings based on temporal context,
    producing element-wise gates that adjust the frozen encoder outputs.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "sigmoid",
    ) -> None:
        """Initialize temporal gating MLP.
        
        Args:
            input_dim: Dimension of time encoding input.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (must match encoder hidden_dim).
            dropout: Dropout probability.
            activation: Final activation ('sigmoid' or 'tanh').
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Final activation for gating
        if activation == "sigmoid":
            self.gate_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.gate_activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Initialize with small weights to start near identity
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        logger.debug(
            f"Initialized temporal MLP: {input_dim} → {hidden_dim} → {output_dim} "
            f"with {activation} activation"
        )
    
    def forward(self, time_encoding: torch.Tensor) -> torch.Tensor:
        """Generate gating values from time encoding.
        
        Args:
            time_encoding: Temporal features [batch_size, input_dim].
            
        Returns:
            Gating values [batch_size, output_dim].
        """
        gates = self.mlp(time_encoding)
        gates = self.gate_activation(gates)
        return gates


class TIDELite(nn.Module):
    """TIDE-Lite model with frozen encoder and temporal modulation.
    
    This model combines a frozen pre-trained encoder with a lightweight
    temporal modulation mechanism that adjusts embeddings based on timestamps.
    
    The architecture:
    1. Frozen encoder produces base embeddings
    2. Timestamps are encoded via sinusoidal functions
    3. Temporal MLP generates element-wise gates
    4. Gates modulate base embeddings via Hadamard product
    
    Total extra parameters: ~53K (for default configuration)
    """
    
    def __init__(self, config: Optional[TIDELiteConfig] = None) -> None:
        """Initialize TIDE-Lite model.
        
        Args:
            config: Model configuration (uses defaults if None).
        """
        super().__init__()
        
        self.config = config or TIDELiteConfig()
        
        # Load pre-trained encoder
        self.encoder = AutoModel.from_pretrained(self.config.encoder_name)
        
        # Freeze encoder if specified
        if self.config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(f"Frozen encoder: {self.config.encoder_name}")
        
        # Temporal components
        self.time_encoder = SinusoidalTimeEncoding(self.config.time_encoding_dim)
        self.temporal_gate = TemporalGatingMLP(
            input_dim=self.config.time_encoding_dim,
            hidden_dim=self.config.mlp_hidden_dim,
            output_dim=self.config.hidden_dim,
            dropout=self.config.mlp_dropout,
            activation=self.config.gate_activation,
        )
        
        # Latency tracking (for profiling without execution)
        self._forward_times = []
        self._encode_times = []
        
        logger.info(
            f"Initialized TIDE-Lite with {self.count_extra_parameters():,} extra parameters"
        )
    
    def encode_base(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text to base embeddings without temporal modulation.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            
        Returns:
            Base embeddings [batch_size, hidden_dim].
        """
        start_time = time.perf_counter()
        
        # Get encoder outputs
        with torch.no_grad() if self.config.freeze_encoder else torch.enable_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        # Pool based on strategy
        if self.config.pooling_strategy == "mean":
            # Mean pooling over sequence dimension
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            embeddings = embeddings / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.config.pooling_strategy == "cls":
            # Use [CLS] token
            embeddings = outputs.last_hidden_state[:, 0]
        elif self.config.pooling_strategy == "max":
            # Max pooling
            token_embeddings = outputs.last_hidden_state
            # Set padding tokens to large negative value
            token_embeddings[attention_mask == 0] = -1e9
            embeddings, _ = torch.max(token_embeddings, dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling_strategy}")
        
        self._encode_times.append(time.perf_counter() - start_time)
        
        return embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with temporal modulation.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            timestamps: Unix timestamps in seconds [batch_size].
            
        Returns:
            Tuple of:
                - Temporally modulated embeddings [batch_size, hidden_dim]
                - Base embeddings without modulation [batch_size, hidden_dim]
        """
        start_time = time.perf_counter()
        
        # Get base embeddings
        base_embeddings = self.encode_base(input_ids, attention_mask)
        
        # Encode timestamps
        time_encoding = self.time_encoder(timestamps)
        
        # Generate temporal gates
        gates = self.temporal_gate(time_encoding)
        
        # Apply gating via Hadamard product
        temporal_embeddings = base_embeddings * gates
        
        self._forward_times.append(time.perf_counter() - start_time)
        
        return temporal_embeddings, base_embeddings
    
    def count_extra_parameters(self) -> int:
        """Count only the trainable parameters added by TIDE-Lite.
        
        Returns:
            Number of extra parameters (excludes frozen encoder).
        """
        extra_params = 0
        
        # Count temporal MLP parameters only
        for module in [self.temporal_gate]:
            for param in module.parameters():
                if param.requires_grad:
                    extra_params += param.numel()
        
        return extra_params
    
    def get_parameter_summary(self) -> Dict[str, int]:
        """Get detailed parameter count breakdown.
        
        Returns:
            Dictionary with parameter counts for each component.
        """
        summary = {
            "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
            "encoder_trainable": sum(
                p.numel() for p in self.encoder.parameters() if p.requires_grad
            ),
            "temporal_mlp_params": sum(p.numel() for p in self.temporal_gate.parameters()),
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "extra_params": self.count_extra_parameters(),
        }
        
        return summary
    
    def estimate_latency(self, batch_size: int = 32, seq_len: int = 128) -> Dict[str, float]:
        """Estimate latency based on recorded times (without execution).
        
        Args:
            batch_size: Batch size for estimation.
            seq_len: Sequence length for estimation.
            
        Returns:
            Dictionary with latency estimates in milliseconds.
            
        Note:
            This returns mock estimates when no actual timing data is available.
            In production, these would be populated from actual forward passes.
        """
        if not self._forward_times:
            # Return reasonable estimates based on model size
            base_latency = 8.0  # ms for MiniLM
            mlp_overhead = 2.0  # ms for temporal MLP
            
            return {
                "base_encoding_ms": base_latency,
                "temporal_modulation_ms": mlp_overhead,
                "total_forward_ms": base_latency + mlp_overhead,
                "throughput_samples_per_sec": 1000 * batch_size / (base_latency + mlp_overhead),
            }
        
        # Calculate statistics from recorded times
        avg_forward = sum(self._forward_times) / len(self._forward_times) * 1000
        avg_encode = sum(self._encode_times) / len(self._encode_times) * 1000 if self._encode_times else 0
        
        return {
            "base_encoding_ms": avg_encode,
            "temporal_modulation_ms": avg_forward - avg_encode,
            "total_forward_ms": avg_forward,
            "throughput_samples_per_sec": 1000 * batch_size / avg_forward,
        }
    
    def reset_timing_stats(self) -> None:
        """Reset internal timing statistics."""
        self._forward_times = []
        self._encode_times = []
        logger.debug("Reset timing statistics")
    
    @torch.no_grad()
    def encode_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        use_temporal: bool = True,
    ) -> torch.Tensor:
        """Encode a batch for inference (no gradient tracking).
        
        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            timestamps: Optional timestamps [batch_size].
            use_temporal: Whether to apply temporal modulation.
            
        Returns:
            Embeddings [batch_size, hidden_dim].
        """
        if timestamps is None or not use_temporal:
            return self.encode_base(input_ids, attention_mask)
        else:
            temporal_emb, _ = self.forward(input_ids, attention_mask, timestamps)
            return temporal_emb
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save model weights and configuration.
        
        Args:
            save_directory: Directory to save model files.
        """
        import json
        from pathlib import Path
        
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_dict = {
            "encoder_name": self.config.encoder_name,
            "hidden_dim": self.config.hidden_dim,
            "time_encoding_dim": self.config.time_encoding_dim,
            "mlp_hidden_dim": self.config.mlp_hidden_dim,
            "mlp_dropout": self.config.mlp_dropout,
            "gate_activation": self.config.gate_activation,
            "freeze_encoder": self.config.freeze_encoder,
            "pooling_strategy": self.config.pooling_strategy,
        }
        
        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save temporal components only (encoder is frozen)
        torch.save({
            "temporal_gate_state_dict": self.temporal_gate.state_dict(),
            "config": config_dict,
        }, save_path / "tide_lite.pt")
        
        logger.info(f"Saved TIDE-Lite model to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_directory: str) -> "TIDELite":
        """Load model from saved files.
        
        Args:
            model_directory: Directory containing saved model files.
            
        Returns:
            Loaded TIDE-Lite model.
        """
        import json
        from pathlib import Path
        
        load_path = Path(model_directory)
        
        # Load configuration
        with open(load_path / "config.json", "r") as f:
            config_dict = json.load(f)
        
        config = TIDELiteConfig(**config_dict)
        
        # Initialize model
        model = cls(config)
        
        # Load temporal components
        checkpoint = torch.load(load_path / "tide_lite.pt", map_location="cpu")
        model.temporal_gate.load_state_dict(checkpoint["temporal_gate_state_dict"])
        
        logger.info(f"Loaded TIDE-Lite model from {model_directory}")
        
        return model
