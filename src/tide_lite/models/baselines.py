"""Baseline encoder models for comparison with TIDE-Lite.

This module provides frozen encoder baselines with the same embedding API
as TIDE-Lite but without temporal modulation.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
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
    ) -> None:
        """Initialize baseline encoder.
        
        Args:
            model_name: HuggingFace model identifier.
            pooling_strategy: How to pool token embeddings ('mean', 'cls', 'max').
            freeze: Whether to freeze encoder parameters.
        """
        super().__init__()
        
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        
        # Load pre-trained model
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get hidden dimension from config
        self.hidden_dim = self.encoder.config.hidden_size
        
        # Freeze if specified
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(f"Initialized frozen baseline: {model_name}")
        else:
            logger.info(f"Initialized trainable baseline: {model_name}")
    
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
        }


def load_minilm_baseline(
    pooling_strategy: str = "mean",
    freeze: bool = True,
) -> BaselineEncoder:
    """Load all-MiniLM-L6-v2 baseline model.
    
    Args:
        pooling_strategy: Pooling method for token embeddings.
        freeze: Whether to freeze encoder weights.
        
    Returns:
        Baseline encoder model.
        
    Note:
        - Model: sentence-transformers/all-MiniLM-L6-v2
        - Parameters: ~22M
        - Hidden dimension: 384
        - Max sequence length: 256 tokens
    """
    return BaselineEncoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        pooling_strategy=pooling_strategy,
        freeze=freeze,
    )


def load_e5_base_baseline(
    pooling_strategy: str = "mean",
    freeze: bool = True,
) -> BaselineEncoder:
    """Load E5-base baseline model.
    
    Args:
        pooling_strategy: Pooling method for token embeddings.
        freeze: Whether to freeze encoder weights.
        
    Returns:
        Baseline encoder model.
        
    Note:
        - Model: intfloat/e5-base
        - Parameters: ~110M
        - Hidden dimension: 768
        - Max sequence length: 512 tokens
        - Requires 'query: ' and 'passage: ' prefixes for optimal performance
    """
    return BaselineEncoder(
        model_name="intfloat/e5-base",
        pooling_strategy=pooling_strategy,
        freeze=freeze,
    )


def load_bge_base_baseline(
    pooling_strategy: str = "cls",  # BGE uses CLS pooling by default
    freeze: bool = True,
) -> BaselineEncoder:
    """Load BGE-base baseline model.
    
    Args:
        pooling_strategy: Pooling method for token embeddings.
        freeze: Whether to freeze encoder weights.
        
    Returns:
        Baseline encoder model.
        
    Note:
        - Model: BAAI/bge-base-en
        - Parameters: ~109M
        - Hidden dimension: 768
        - Max sequence length: 512 tokens
        - Optimized for retrieval tasks
    """
    return BaselineEncoder(
        model_name="BAAI/bge-base-en",
        pooling_strategy=pooling_strategy,
        freeze=freeze,
    )


class BaselineComparison:
    """Utility class for comparing baseline models.
    
    Provides methods to compare embeddings, performance, and efficiency
    across different baseline encoders.
    """
    
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
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Generate embeddings from all models for comparison.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            
        Returns:
            Dictionary mapping model names to embeddings.
        """
        embeddings = {}
        
        for name, model in self.models.items():
            model.eval()
            emb = model.encode_base(input_ids, attention_mask)
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
    suite.add_model("minilm", load_minilm_baseline())
    suite.add_model("e5-base", load_e5_base_baseline())
    suite.add_model("bge-base", load_bge_base_baseline())
    
    logger.info("Created baseline suite with 3 models")
    
    return suite
