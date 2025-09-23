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
    
    def encode_base(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text to embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            
        Returns:
            Embeddings [batch_size, hidden_dim].
        """
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        embeddings = self.pool_embeddings(outputs.last_hidden_state, attention_mask)
        return embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (ignores timestamps for baseline).
        
        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            timestamps: Ignored for baseline models.
            
        Returns:
            Tuple of (embeddings, embeddings) for API compatibility.
        """
        if timestamps is not None:
            logger.debug("Baseline model ignoring timestamps")
        
        embeddings = self.encode_base(input_ids, attention_mask)
        return embeddings, embeddings  # Return twice for API compatibility
    
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
            timestamps: Ignored for baseline models.
            
        Returns:
            Embeddings tensor [num_texts, embedding_dim].
        """
        if timestamps is not None:
            logger.debug("Baseline model ignoring timestamps in encode_texts")
        
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
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            embeddings = self.encode_base(
                inputs["input_ids"],
                inputs["attention_mask"]
            )
            
            all_embeddings.append(embeddings.cpu())
        
        # Concatenate all batches
        return torch.cat(all_embeddings, dim=0)
    
    def count_extra_parameters(self) -> int:
        """Count extra parameters (always 0 for frozen baselines).
        
        Returns:
            Number of extra trainable parameters (0).
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_summary(self) -> Dict[str, int]:
        """Get parameter count summary.
        
        Returns:
            Dictionary with parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
            "hidden_dim": self.hidden_dim,
            "extra_params": trainable,  # For baseline, all trainable params are "extra"
        }


class MiniLMBaseline(BaselineEncoder):
    """All-MiniLM-L6-v2 baseline model.
    
    - Model: sentence-transformers/all-MiniLM-L6-v2
    - Parameters: ~22M
    - Hidden dimension: 384
    - Max sequence length: 256 tokens
    """
    
    def __init__(
        self,
        pooling_strategy: str = "mean",
        freeze: bool = True,
        max_seq_length: int = 128,
    ) -> None:
        """Initialize MiniLM baseline.
        
        Args:
            pooling_strategy: Pooling method for token embeddings.
            freeze: Whether to freeze encoder weights.
            max_seq_length: Maximum sequence length for tokenization.
        """
        super().__init__(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            pooling_strategy=pooling_strategy,
            freeze=freeze,
            max_seq_length=max_seq_length,
        )


class E5BaseBaseline(BaselineEncoder):
    """E5-base baseline model.
    
    - Model: intfloat/e5-base
    - Parameters: ~110M
    - Hidden dimension: 768
    - Max sequence length: 512 tokens
    - Requires 'query: ' and 'passage: ' prefixes for optimal performance
    """
    
    def __init__(
        self,
        pooling_strategy: str = "mean",
        freeze: bool = True,
        max_seq_length: int = 128,
        add_prefix: bool = True,
    ) -> None:
        """Initialize E5-base baseline.
        
        Args:
            pooling_strategy: Pooling method for token embeddings.
            freeze: Whether to freeze encoder weights.
            max_seq_length: Maximum sequence length for tokenization.
            add_prefix: Whether to add E5 prefixes to texts.
        """
        super().__init__(
            model_name="intfloat/e5-base",
            pooling_strategy=pooling_strategy,
            freeze=freeze,
            max_seq_length=max_seq_length,
        )
        self.add_prefix = add_prefix
    
    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        timestamps: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Encode text strings to embeddings with E5 prefixes.
        
        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding.
            timestamps: Ignored for baseline models.
            
        Returns:
            Embeddings tensor [num_texts, embedding_dim].
        """
        # Add E5 prefixes if enabled
        if self.add_prefix:
            texts = [f"query: {text}" for text in texts]
        
        return super().encode_texts(texts, batch_size, timestamps)


class BGEBaseBaseline(BaselineEncoder):
    """BGE-base baseline model.
    
    - Model: BAAI/bge-base-en-v1.5
    - Parameters: ~109M
    - Hidden dimension: 768
    - Max sequence length: 512 tokens
    - Optimized for retrieval tasks
    """
    
    def __init__(
        self,
        pooling_strategy: str = "cls",  # BGE uses CLS pooling by default
        freeze: bool = True,
        max_seq_length: int = 128,
    ) -> None:
        """Initialize BGE-base baseline.
        
        Args:
            pooling_strategy: Pooling method for token embeddings.
            freeze: Whether to freeze encoder weights.
            max_seq_length: Maximum sequence length for tokenization.
        """
        super().__init__(
            model_name="BAAI/bge-base-en-v1.5",
            pooling_strategy=pooling_strategy,
            freeze=freeze,
            max_seq_length=max_seq_length,
        )


def load_baseline(
    model_type: str = "minilm",
    **kwargs
) -> BaselineEncoder:
    """Factory function to load baseline models.
    
    Args:
        model_type: Type of baseline ('minilm', 'e5-base', 'bge-base').
        **kwargs: Additional arguments for model initialization.
        
    Returns:
        Baseline encoder model.
        
    Raises:
        ValueError: If model_type is not recognized.
    """
    model_type = model_type.lower()
    
    if model_type == "minilm":
        return MiniLMBaseline(**kwargs)
    elif model_type in ["e5", "e5-base"]:
        return E5BaseBaseline(**kwargs)
    elif model_type in ["bge", "bge-base"]:
        return BGEBaseBaseline(**kwargs)
    else:
        raise ValueError(
            f"Unknown baseline model: {model_type}. "
            f"Choose from: minilm, e5-base, bge-base"
        )


class BaselineComparison:
    """Utility class for comparing baseline models."""
    
    def __init__(self) -> None:
        """Initialize comparison utility."""
        self.models = {}
        logger.info("Initialized baseline comparison utility")
    
    def add_model(self, name: str, model: BaselineEncoder) -> None:
        """Add a model for comparison.
        
        Args:
            name: Identifier for the model.
            model: BaselineEncoder instance.
        """
        self.models[name] = model
        summary = model.get_parameter_summary()
        logger.info(
            f"Added {name}: {summary['total_params']:,} params, "
            f"hidden_dim={summary['hidden_dim']}"
        )
    
    def compare_parameters(self) -> Dict[str, Dict[str, int]]:
        """Compare parameter counts across models.
        
        Returns:
            Dictionary mapping model names to parameter summaries.
        """
        return {
            name: model.get_parameter_summary()
            for name, model in self.models.items()
        }
    
    @torch.no_grad()
    def compare_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> Dict[str, torch.Tensor]:
        """Generate embeddings from all models for comparison.
        
        Args:
            texts: List of text strings to encode.
            batch_size: Batch size for encoding.
            
        Returns:
            Dictionary mapping model names to embeddings.
        """
        embeddings = {}
        
        for name, model in self.models.items():
            emb = model.encode_texts(texts, batch_size)
            embeddings[name] = emb
            logger.debug(f"Generated embeddings from {name}: shape {emb.shape}")
        
        return embeddings
    
    def compute_similarity_matrix(
        self,
        embeddings_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute pairwise cosine similarities between model embeddings.
        
        Args:
            embeddings_dict: Dictionary of model embeddings.
            
        Returns:
            Similarity matrix [n_models, n_models].
        """
        model_names = list(embeddings_dict.keys())
        n_models = len(model_names)
        
        # Stack embeddings
        embeddings_list = [embeddings_dict[name] for name in model_names]
        
        # Compute average embeddings per model
        avg_embeddings = torch.stack([emb.mean(0) for emb in embeddings_list])
        
        # Normalize
        avg_embeddings = F.normalize(avg_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(avg_embeddings, avg_embeddings.T)
        
        return similarity_matrix


def create_baseline_suite() -> BaselineComparison:
    """Create a suite of baseline models for comparison.
    
    Returns:
        BaselineComparison instance with loaded models.
    """
    suite = BaselineComparison()
    
    # Add standard baselines
    suite.add_model("minilm", MiniLMBaseline())
    suite.add_model("e5-base", E5BaseBaseline())
    suite.add_model("bge-base", BGEBaseBaseline())
    
    logger.info("Created baseline suite with 3 models")
    
    return suite
