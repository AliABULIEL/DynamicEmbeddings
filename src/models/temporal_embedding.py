import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from einops import rearrange, repeat
from .base_embedding import BaseEmbedding


class StateSpaceLayer(nn.Module):
    """State Space Model layer for efficient temporal modeling.

    Based on the S4/S5 architecture for handling long-range dependencies
    with linear complexity, as demonstrated in GraphSSM.
    """

    def __init__(
            self,
            d_model: int,
            state_dim: int = 64,
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dropout: float = 0.0
    ):
        """Initialize SSM layer."""
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # Discretization parameters
        self.dt_min = dt_min
        self.dt_max = dt_max

        # State matrices
        self.A = nn.Parameter(torch.randn(d_model, state_dim, state_dim))
        self.B = nn.Parameter(torch.randn(d_model, state_dim, 1))
        self.C = nn.Parameter(torch.randn(d_model, 1, state_dim))
        self.D = nn.Parameter(torch.randn(d_model))

        # Discretization network
        self.dt_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Softplus()
        )

        self.dropout = nn.Dropout(dropout)
        self._init_parameters()

    def _init_parameters(self):
        """Initialize SSM parameters."""
        # Initialize A as diagonal-dominant for stability
        nn.init.xavier_uniform_(self.A)
        with torch.no_grad():
            self.A.diagonal(dim1=-2, dim2=-1).add_(1.0)

        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.zeros_(self.D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply state space transformation.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of same shape
        """
        batch, seq_len, _ = x.shape

        # Compute discretization timesteps
        dt = self.dt_proj(x)
        dt = self.dt_min + (self.dt_max - self.dt_min) * dt

        # Discretize continuous parameters
        A_discrete = torch.matrix_exp(self.A.unsqueeze(0) * dt.unsqueeze(-1).unsqueeze(-1))
        B_discrete = self.B.unsqueeze(0) * dt.unsqueeze(-1).unsqueeze(-1)

        # Initialize state
        state = torch.zeros(batch, self.d_model, self.state_dim, 1, device=x.device)
        outputs = []

        # Recurrent computation
        for t in range(seq_len):
            # Update state
            state = torch.matmul(A_discrete[:, t], state) + \
                    torch.matmul(B_discrete[:, t], x[:, t:t + 1, :].unsqueeze(-1))

            # Compute output
            y = torch.matmul(self.C.unsqueeze(0), state).squeeze(-1)
            y = y + self.D.unsqueeze(0) * x[:, t:t + 1, :]
            outputs.append(y)

        output = torch.cat(outputs, dim=1)
        return self.dropout(output)


class TemporalEmbedding(BaseEmbedding):
    """Temporal embedding with state space models and time-aware attention.

    Implements temporal dynamics modeling for evolving representations,
    inspired by TeAST and DyGSSM approaches.
    """

    def __init__(
            self,
            input_dim: int,
            embedding_dim: int,
            num_ssm_layers: int = 4,
            state_dim: int = 64,
            time_encoding_dim: int = 32,
            max_temporal_shift: int = 100,
            dropout: float = 0.1,
            **kwargs
    ):
        """Initialize temporal embedding model.

        Args:
            input_dim: Input vocabulary size
            embedding_dim: Embedding dimension
            num_ssm_layers: Number of SSM layers
            state_dim: Hidden state dimension for SSM
            time_encoding_dim: Dimension of time encodings
            max_temporal_shift: Maximum time difference to encode
            dropout: Dropout probability
        """
        super().__init__(input_dim, embedding_dim, dropout, **kwargs)

        # Base embeddings
        self.token_embedding = nn.Embedding(input_dim, embedding_dim)

        # Time encoding
        self.time_encoding_dim = time_encoding_dim
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_encoding_dim * 2),
            nn.GELU(),
            nn.Linear(time_encoding_dim * 2, time_encoding_dim)
        )

        # Combine token and time
        self.fusion = nn.Linear(embedding_dim + time_encoding_dim, embedding_dim)

        # State space layers
        self.ssm_layers = nn.ModuleList([
            StateSpaceLayer(embedding_dim, state_dim, dropout=dropout)
            for _ in range(num_ssm_layers)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim)
            for _ in range(num_ssm_layers)
        ])

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        # Temporal attention for aggregation
        self.temporal_attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_time(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps into continuous representations.

        Args:
            timestamps: Timestamps tensor of shape (batch, seq_len)

        Returns:
            Time encodings of shape (batch, seq_len, time_encoding_dim)
        """
        # Normalize timestamps
        timestamps = timestamps.unsqueeze(-1).float()
        time_encoded = self.time_encoder(timestamps)
        return time_encoded

    def forward(
            self,
            inputs: torch.Tensor,
            timestamps: Optional[torch.Tensor] = None,
            context: Optional[torch.Tensor] = None,
            return_sequence: bool = False,
            **kwargs
    ) -> torch.Tensor:
        """Compute temporal embeddings.

        Args:
            inputs: Input tokens (batch, seq_len)
            timestamps: Optional timestamps for each input
            context: Optional context tensor
            return_sequence: Return full sequence or pooled representation

        Returns:
            Temporal embeddings
        """
        batch_size, seq_len = inputs.shape

        # Get token embeddings
        token_embeds = self.token_embedding(inputs)

        # Add temporal information if provided
        if timestamps is not None:
            time_embeds = self.encode_time(timestamps)
            combined = torch.cat([token_embeds, time_embeds], dim=-1)
            embeddings = self.fusion(combined)
        else:
            embeddings = token_embeds

        # Apply SSM layers with residual connections
        for i, (ssm_layer, layer_norm) in enumerate(zip(self.ssm_layers, self.layer_norms)):
            residual = embeddings
            embeddings = ssm_layer(embeddings)
            embeddings = layer_norm(embeddings + residual)

        # Apply temporal attention if we have timestamps
        if timestamps is not None:
            # Create attention mask based on temporal proximity
            time_diff = timestamps.unsqueeze(1) - timestamps.unsqueeze(2)
            temporal_mask = (time_diff.abs() < 10).float()  # Within 10 time units

            attn_output, _ = self.temporal_attention(
                embeddings, embeddings, embeddings,
                attn_mask=temporal_mask
            )
            embeddings = embeddings + attn_output

        # Output projection
        embeddings = self.output_projection(embeddings)

        # Apply normalization if specified
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Return sequence or pooled representation
        if return_sequence:
            return embeddings
        else:
            # Pool with attention to padding
            mask = (inputs != 0).float()
            return self.pool_embeddings(embeddings, mask, pooling='mean')

    def compute_temporal_consistency_loss(
            self,
            embeddings_t1: torch.Tensor,
            embeddings_t2: torch.Tensor,
            time_diff: torch.Tensor,
            temperature: float = 1.0
    ) -> torch.Tensor:
        """Compute temporal consistency loss.

        Ensures smooth evolution of embeddings over time.

        Args:
            embeddings_t1: Embeddings at time t1
            embeddings_t2: Embeddings at time t2
            time_diff: Time difference between t1 and t2
            temperature: Temperature scaling

        Returns:
            Temporal consistency loss
        """
        # Expected similarity based on time difference
        expected_sim = torch.exp(-time_diff / temperature)

        # Actual similarity
        actual_sim = self.compute_similarity(embeddings_t1, embeddings_t2, metric='cosine')

        # MSE loss between expected and actual similarity
        loss = F.mse_loss(actual_sim, expected_sim)

        return loss