import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math
from .base_embedding import BaseEmbedding


class MatryoshkaEmbedding(BaseEmbedding):
    """Matryoshka embedding with nested representations.

    Implements the Matryoshka Representation Learning technique that allows
    truncation of embeddings to smaller dimensions while maintaining performance.
    Based on the approach used in Voyage-3-Large and EmbeddingGemma.
    """

    def __init__(
            self,
            input_dim: int,
            embedding_dim: int,
            nested_dims: List[int] = [64, 128, 256, 512, 768],
            base_model: str = 'transformer',
            num_layers: int = 12,
            num_heads: int = 12,
            dropout: float = 0.1,
            **kwargs
    ):
        """Initialize Matryoshka embedding model.

        Args:
            input_dim: Vocabulary size
            embedding_dim: Maximum embedding dimension
            nested_dims: List of nested dimensions for training
            base_model: Base architecture ('transformer', 'lstm')
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__(input_dim, embedding_dim, dropout, **kwargs)

        self.nested_dims = sorted(nested_dims)
        self.base_model = base_model

        # Token embeddings
        self.token_embedding = nn.Embedding(input_dim, embedding_dim)
        self.position_embedding = nn.Embedding(512, embedding_dim)

        # Base encoder
        if base_model == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        else:  # LSTM
            self.encoder = nn.LSTM(
                embedding_dim,
                embedding_dim // 2,
                num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )

        # Projection heads for different dimensions
        self.projection_heads = nn.ModuleDict({
            str(dim): nn.Sequential(
                nn.Linear(embedding_dim, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            )
            for dim in self.nested_dims
        })

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
            self,
            inputs: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            target_dim: Optional[int] = None,
            return_all_dims: bool = False,
            **kwargs
    ) -> torch.Tensor:
        """Compute Matryoshka embeddings.

        Args:
            inputs: Input token IDs (batch_size, seq_len)
            context: Optional context tensor
            target_dim: Specific dimension to return
            return_all_dims: Return embeddings at all dimensions

        Returns:
            Embeddings at requested dimension(s)
        """
        batch_size, seq_len = inputs.shape
        device = inputs.device

        # Create position indices
        positions = torch.arange(seq_len, device=device).expand(batch_size, seq_len)

        # Compute base embeddings
        token_embeds = self.token_embedding(inputs)
        pos_embeds = self.position_embedding(positions)
        embeddings = self.dropout(token_embeds + pos_embeds)

        # Apply encoder
        if self.base_model == 'transformer':
            # Create attention mask for padding tokens
            mask = (inputs != 0).float()
            embeddings = self.encoder(
                embeddings,
                src_key_padding_mask=(mask == 0)
            )
        else:  # LSTM
            embeddings, _ = self.encoder(embeddings)

        # Pool to get sentence representation
        pooled = self.pool_embeddings(embeddings, mask=(inputs != 0))

        # Apply projection heads
        if return_all_dims:
            results = {}
            for dim in self.nested_dims:
                projected = self.projection_heads[str(dim)](pooled)
                if self.normalize:
                    projected = F.normalize(projected, p=2, dim=-1)
                results[dim] = projected
            return results
        else:
            # Return specific dimension or maximum
            if target_dim is None:
                target_dim = self.embedding_dim

            # Find closest available dimension
            available_dim = min(self.nested_dims, key=lambda x: abs(x - target_dim))
            projected = self.projection_heads[str(available_dim)](pooled)

            if self.normalize:
                projected = F.normalize(projected, p=2, dim=-1)

            # Truncate if needed
            if target_dim < available_dim:
                projected = projected[..., :target_dim]

            return projected

    def compute_matryoshka_loss(
            self,
            embeddings_dict: dict,
            positive_pairs: torch.Tensor,
            negative_pairs: torch.Tensor,
            temperature: float = 0.05
    ) -> torch.Tensor:
        """Compute multi-scale contrastive loss.

        Args:
            embeddings_dict: Dictionary of embeddings at different dimensions
            positive_pairs: Indices of positive pairs
            negative_pairs: Indices of negative pairs
            temperature: Temperature for contrastive loss

        Returns:
            Combined loss across all dimensions
        """
        total_loss = 0
        weights = {dim: math.sqrt(dim / max(self.nested_dims))
                   for dim in self.nested_dims}

        for dim, embeddings in embeddings_dict.items():
            # Compute similarities
            sim_pos = self.compute_similarity(
                embeddings[positive_pairs[:, 0]],
                embeddings[positive_pairs[:, 1]]
            ) / temperature

            sim_neg = self.compute_similarity(
                embeddings[negative_pairs[:, 0]],
                embeddings[negative_pairs[:, 1]]
            ) / temperature

            # Contrastive loss
            loss = -torch.log(
                torch.exp(sim_pos) /
                (torch.exp(sim_pos) + torch.exp(sim_neg).sum(dim=-1))
            ).mean()

            total_loss += weights[dim] * loss

        return total_loss
