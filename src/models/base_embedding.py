import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class BaseEmbedding(nn.Module, ABC):
    """Abstract base class for all embedding models.

    This class defines the interface that all embedding models must implement,
    ensuring consistency across different embedding strategies.
    """

    def __init__(
            self,
            input_dim: int,
            embedding_dim: int,
            dropout: float = 0.1,
            normalize: bool = True,
            **kwargs
    ):
        """Initialize base embedding model.

        Args:
            input_dim: Size of input vocabulary or features
            embedding_dim: Dimension of output embeddings
            dropout: Dropout probability for regularization
            normalize: Whether to L2-normalize embeddings
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.normalize = normalize

        # Store configuration for serialization
        self.config = {
            'input_dim': input_dim,
            'embedding_dim': embedding_dim,
            'dropout': dropout,
            'normalize': normalize,
            **kwargs
        }

        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")

    @abstractmethod
    def forward(
            self,
            inputs: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        """Compute embeddings for inputs.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length)
            context: Optional context tensor for adaptive embeddings
            **kwargs: Additional forward-pass parameters

        Returns:
            Embedding tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        pass

    def compute_similarity(
            self,
            embeddings1: torch.Tensor,
            embeddings2: torch.Tensor,
            metric: str = 'cosine'
    ) -> torch.Tensor:
        """Compute similarity between embedding pairs.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            metric: Similarity metric ('cosine', 'euclidean', 'dot')

        Returns:
            Similarity scores
        """
        if metric == 'cosine':
            # Normalize embeddings for cosine similarity
            norm1 = torch.nn.functional.normalize(embeddings1, p=2, dim=-1)
            norm2 = torch.nn.functional.normalize(embeddings2, p=2, dim=-1)
            return torch.sum(norm1 * norm2, dim=-1)
        elif metric == 'euclidean':
            return -torch.norm(embeddings1 - embeddings2, p=2, dim=-1)
        elif metric == 'dot':
            return torch.sum(embeddings1 * embeddings2, dim=-1)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def pool_embeddings(
            self,
            embeddings: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            pooling: str = 'mean'
    ) -> torch.Tensor:
        """Pool sequence embeddings into fixed-size representation.

        Args:
            embeddings: Embeddings of shape (batch, seq_len, dim)
            mask: Attention mask for valid positions
            pooling: Pooling strategy ('mean', 'max', 'cls', 'weighted')

        Returns:
            Pooled embeddings of shape (batch, dim)
        """
        if pooling == 'mean':
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand(embeddings.size())
                sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            return torch.mean(embeddings, dim=1)
        elif pooling == 'max':
            return torch.max(embeddings, dim=1)[0]
        elif pooling == 'cls':
            return embeddings[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, **override_config):
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        config = checkpoint['config']
        config.update(override_config)

        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
        return model