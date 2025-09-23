"""TIDE-Lite: Temporally-Indexed Dynamic Embeddings model.

This module implements the core TIDE-Lite architecture with a frozen encoder
and lightweight temporal modulation via MLP-based gating.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

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
        max_seq_length: Maximum sequence length for tokenization.
    """
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    hidden_dim: int = 384
    time_encoding_dim: int = 32
    mlp_hidden_dim: int = 128
    mlp_dropout: float = 0.1
    gate_activation: str = "sigmoid"
    freeze_encoder: bool = True
    pooling_strategy: str = "mean"
    max_seq_length: int = 128


class SinusoidalTimeEncoding(nn.Module):
    """Sinusoidal position encoding adapted for timestamps."""
    
    def __init__(self, encoding_dim: int = 32) -> None:
        """Initialize time encoding.
        
        Args:
            encoding_dim: Dimension of output encoding (must be even).
        """
        super().__init__()
        if encoding_dim % 2 != 0:
            raise ValueError(f"encoding_dim must be even, got {encoding_dim}")
        
        self.encoding_dim = encoding_dim
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps into sinusoidal features.
        
        Args:
            timestamps: Unix timestamps [batch_size].
            
        Returns:
            Time encodings [batch_size, encoding_dim].
        """
        batch_size = timestamps.shape[0]
        device = timestamps.device
        
        # Ensure timestamps are 1D
        if timestamps.dim() > 1:
            timestamps = timestamps.squeeze()
        
        # Create position indices for encoding dimensions
        dim_indices = torch.arange(0, self.encoding_dim, 2, device=device).float()
        
        # Compute frequency scaling factors
        div_term = torch.exp(dim_indices * -(torch.log(torch.tensor(10000.0)) / self.encoding_dim))
        
        # Expand for batch processing
        timestamps = timestamps.unsqueeze(1)
        div_term = div_term.unsqueeze(0)
        
        # Compute sinusoidal encodings
        encodings = torch.zeros(batch_size, self.encoding_dim, device=device)
        encodings[:, 0::2] = torch.sin(timestamps * div_term)
        encodings[:, 1::2] = torch.cos(timestamps * div_term)
        
        return encodings


class TemporalGatingMLP(nn.Module):
    """MLP that generates gating values from temporal encodings."""
    
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
    
    Architecture:
    1. Frozen encoder produces base embeddings
    2. Mean pooling over token embeddings
    3. Timestamps encoded via sinusoidal functions
    4. Temporal MLP generates element-wise gates (Sigmoid)
    5. Hadamard product modulates base embeddings
    """
    
    def __init__(self, config: Optional[TIDELiteConfig] = None) -> None:
        """Initialize TIDE-Lite model.
        
        Args:
            config: Model configuration (uses defaults if None).
        """
        super().__init__()
        
        self.config = config or TIDELiteConfig()
        
        # Load pre-trained encoder and tokenizer
        self.encoder = AutoModel.from_pretrained(self.config.encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_name)
        
        # Update hidden_dim from actual model
        self.config.hidden_dim = self.encoder.config.hidden_size
        
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
        
        logger.info(
            f"Initialized TIDE-Lite with {self.count_extra_parameters():,} extra parameters"
        )
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Embedding dimension.
        """
        return self.config.hidden_dim
    
    def pool_embeddings(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool token embeddings based on strategy.
        
        Args:
            token_embeddings: Token-level embeddings [batch, seq_len, hidden_dim].
            attention_mask: Attention mask [batch, seq_len].
            
        Returns:
            Pooled embeddings [batch, hidden_dim].
        """
        if self.config.pooling_strategy == "mean":
            # Mean pooling with proper masking
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            embeddings = embeddings / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.config.pooling_strategy == "cls":
            # Use [CLS] token
            embeddings = token_embeddings[:, 0]
        elif self.config.pooling_strategy == "max":
            # Max pooling with masking
            token_embeddings = token_embeddings.clone()
            token_embeddings[attention_mask == 0] = -1e9
            embeddings, _ = torch.max(token_embeddings, dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling_strategy}")
        
        return embeddings
    
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
        with torch.no_grad() if self.config.freeze_encoder else torch.enable_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        # Pool embeddings
        embeddings = self.pool_embeddings(outputs.last_hidden_state, attention_mask)
        return embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
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
        # Get base embeddings
        base_embeddings = self.encode_base(input_ids, attention_mask)
        
        if timestamps is None:
            # No temporal modulation
            return base_embeddings, base_embeddings
        
        # Encode timestamps
        time_encoding = self.time_encoder(timestamps)
        
        # Generate temporal gates
        gates = self.temporal_gate(time_encoding)
        
        # Apply gating via Hadamard product
        temporal_embeddings = base_embeddings * gates
        
        return temporal_embeddings, base_embeddings
    
    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        timestamps: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Encode text strings to embeddings (unified API).
        
        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding.
            timestamps: Optional list of Unix timestamps for temporal modulation.
            
        Returns:
            Embeddings tensor [num_texts, embedding_dim].
        """
        self.eval()
        device = next(self.parameters()).device
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Prepare timestamps if provided
            batch_timestamps = None
            if timestamps is not None:
                batch_timestamps = torch.tensor(
                    timestamps[i:i + batch_size],
                    dtype=torch.float32,
                    device=device
                )
            
            # Forward pass
            if batch_timestamps is not None:
                embeddings, _ = self.forward(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    batch_timestamps
                )
            else:
                embeddings = self.encode_base(
                    inputs["input_ids"],
                    inputs["attention_mask"]
                )
            
            all_embeddings.append(embeddings.cpu())
        
        # Concatenate all batches
        return torch.cat(all_embeddings, dim=0)
    
    def count_extra_parameters(self) -> int:
        """Count only the trainable parameters added by TIDE-Lite.
        
        Returns:
            Number of extra parameters (excludes frozen encoder).
        """
        extra_params = 0
        
        # Count temporal MLP parameters only
        for param in self.temporal_gate.parameters():
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
            "max_seq_length": self.config.max_seq_length,
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
