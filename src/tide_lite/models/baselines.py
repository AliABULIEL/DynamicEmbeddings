"""Baseline encoder models for comparison with TIDE-Lite.

This module provides frozen encoder baselines with the same embedding API
as TIDE-Lite but without temporal modulation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class BaselineEncoder(nn.Module):
    """Base class for frozen encoder baselines.
    
    Provides the same API as TIDE-Lite but without temporal modulation,
    enabling fair comparison of temporal vs. static embeddings.
    """
    
    def __init__(
        self,
        model_name: str,
        pooling_strategy: str = "mean",
        freeze: bool = True,
        max_seq_length: int = 128,
    ) -> None:
        """Initialize baseline encoder.
        
        Args:
            model_name: HuggingFace model identifier.
            pooling_strategy: How to pool token embeddings ('mean', 'cls', 'max').
            freeze: Whether to freeze encoder parameters.
            max_seq_length: Maximum sequence length for tokenization.
        """
        super().__init__()
        
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.max_seq_length = max_seq_length
        
        # Load pre-trained model and tokenizer
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get hidden dimension from config
        self.hidden_dim = self.encoder.config.hidden_size
        
        # Freeze if specified
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(f"Initialized frozen baseline: {model_name}")
        else:
            logger.info(f"Initialized trainable baseline: {model_name}")
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Embedding dimension.
        """
        return self.hidden_dim
    
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
        if self.pooling_strategy == "mean":
            # Mean pooling with proper masking
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            embeddings = embeddings / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.pooling_strategy == "cls":
            # Use [CLS] token
            embeddings = token_embeddings[:, 0]
        elif self.pooling_strategy == "max":
            # Max pooling with masking
            token_embeddings = token_embeddings.clone()
            token_embeddings[attention_mask == 0] = -1e9
            embeddings, _ = torch.max(token_embeddings, dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,  # Ignored for baseline
    ) -> Tuple[torch.Tensor, None]:
        """Compute embeddings (timestamps ignored for baselines).
        
        Args:
            input_ids: Token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            timestamps: Ignored (for API compatibility).
            
        Returns:
            Tuple of (embeddings, None).
        """
        # Run encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pool embeddings
        embeddings = self.pool_embeddings(outputs.last_hidden_state, attention_mask)
        
        return embeddings, None
    
    def encode_base(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode without temporal modulation (same as forward).
        
        Args:
            input_ids: Token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            
        Returns:
            Base embeddings [batch, hidden_dim].
        """
        embeddings, _ = self.forward(input_ids, attention_mask)
        return embeddings


def load_baseline(
    model_id: str,
    max_seq_length: int = 128,
    **kwargs,
) -> BaselineEncoder:
    """Load a baseline model by ID.
    
    Args:
        model_id: Model identifier ('minilm', 'e5-base', 'bge-base').
        max_seq_length: Maximum sequence length.
        **kwargs: Additional arguments for BaselineEncoder.
        
    Returns:
        Initialized baseline encoder.
    """
    # Model ID to HuggingFace model name mapping
    MODEL_MAP = {
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "e5-base": "intfloat/e5-base-v2",
        "bge-base": "BAAI/bge-base-en-v1.5",
    }
    
    if model_id not in MODEL_MAP:
        raise ValueError(
            f"Unknown model ID: {model_id}. "
            f"Available: {', '.join(MODEL_MAP.keys())}"
        )
    
    model_name = MODEL_MAP[model_id]
    logger.info(f"Loading baseline model: {model_id} -> {model_name}")
    
    return BaselineEncoder(
        model_name=model_name,
        max_seq_length=max_seq_length,
        **kwargs,
    )


def load_minilm_baseline(max_seq_length: int = 128) -> BaselineEncoder:
    """Load MiniLM baseline model.
    
    Args:
        max_seq_length: Maximum sequence length.
        
    Returns:
        MiniLM baseline encoder.
    """
    return load_baseline("minilm", max_seq_length=max_seq_length)


def load_e5_base_baseline(max_seq_length: int = 128) -> BaselineEncoder:
    """Load E5-base baseline model.
    
    Args:
        max_seq_length: Maximum sequence length.
        
    Returns:
        E5-base baseline encoder.
    """
    return load_baseline("e5-base", max_seq_length=max_seq_length)


def load_bge_base_baseline(max_seq_length: int = 128) -> BaselineEncoder:
    """Load BGE-base baseline model.
    
    Args:
        max_seq_length: Maximum sequence length.
        
    Returns:
        BGE-base baseline encoder.
    """
    return load_baseline("bge-base", max_seq_length=max_seq_length)
