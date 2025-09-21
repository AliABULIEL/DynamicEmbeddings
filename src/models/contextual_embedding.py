import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from .base_embedding import BaseEmbedding


class ContextualEmbedding(BaseEmbedding):
    """Contextual document embeddings with two-stage processing.

    Implements context-aware embeddings that adapt based on surrounding
    context, inspired by the Contextual Document Embeddings approach.
    """

    def __init__(
            self,
            input_dim: int,
            embedding_dim: int,
            context_window: int = 128,
            num_context_layers: int = 6,
            num_heads: int = 8,
            use_position_agnostic: bool = True,
            dropout: float = 0.1,
            **kwargs
    ):
        """Initialize contextual embedding model.

        Args:
            input_dim: Vocabulary size
            embedding_dim: Embedding dimension
            context_window: Size of context window
            num_context_layers: Number of context processing layers
            num_heads: Number of attention heads
            use_position_agnostic: Remove positional bias
            dropout: Dropout probability
        """
        super().__init__(input_dim, embedding_dim, dropout, **kwargs)

        self.context_window = context_window
        self.use_position_agnostic = use_position_agnostic

        # Token embeddings
        self.token_embedding = nn.Embedding(input_dim, embedding_dim)

        # Position embeddings (optional)
        if not use_position_agnostic:
            self.position_embedding = nn.Embedding(512, embedding_dim)
        else:
            self.position_embedding = None

        # Context encoder
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_context_layers // 2
        )

        # Document encoder
        self.document_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_context_layers // 2
        )

        # Cross-attention between context and document
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Context gate
        self.context_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def extract_context_windows(
            self,
            inputs: torch.Tensor,
            window_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract context windows around each position.

        Args:
            inputs: Input tensor (batch, seq_len)
            window_size: Context window size

        Returns:
            Context windows and masks
        """
        batch_size, seq_len = inputs.shape
        device = inputs.device

        # Pad sequence for context extraction
        half_window = window_size // 2
        padded = F.pad(inputs, (half_window, half_window), value=0)

        # Extract windows
        windows = []
        masks = []

        for i in range(seq_len):
            window = padded[:, i:i + window_size]
            mask = (window != 0).float()
            windows.append(window)
            masks.append(mask)

        windows = torch.stack(windows, dim=1)  # (batch, seq_len, window_size)
        masks = torch.stack(masks, dim=1)

        return windows, masks

    def forward(
            self,
            inputs: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            document_boundaries: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        """Compute contextual embeddings.

        Args:
            inputs: Input tokens (batch, seq_len)
            context: Optional external context
            document_boundaries: Optional document boundary markers

        Returns:
            Contextual embeddings
        """
        batch_size, seq_len = inputs.shape
        device = inputs.device

        # Get base embeddings
        embeddings = self.token_embedding(inputs)

        # Add positional information if not position-agnostic
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=device).expand(batch_size, seq_len)
            pos_embeds = self.position_embedding(positions)
            embeddings = embeddings + pos_embeds

        # Stage 1: Process local context
        context_windows, context_masks = self.extract_context_windows(
            inputs, self.context_window
        )

        # Reshape for processing
        context_windows_flat = context_windows.view(-1, self.context_window)
        context_embeds = self.token_embedding(context_windows_flat)
        context_embeds = context_embeds.view(batch_size, seq_len, self.context_window, -1)

        # Process each context window
        context_representations = []
        for i in range(seq_len):
            ctx_embed = context_embeds[:, i]  # (batch, window_size, dim)
            ctx_mask = context_masks[:, i]  # (batch, window_size)

            # Encode context
            ctx_encoded = self.context_encoder(
                ctx_embed,
                src_key_padding_mask=(ctx_mask == 0)
            )

            # Pool context
            pooled_ctx = self.pool_embeddings(ctx_encoded, ctx_mask, pooling='mean')
            context_representations.append(pooled_ctx)

        context_representations = torch.stack(context_representations, dim=1)

        # Stage 2: Document-level processing
        mask = (inputs != 0).float()

        # Encode document
        doc_encoded = self.document_encoder(
            embeddings,
            src_key_padding_mask=(mask == 0)
        )

        # Cross-attention between document and context
        attended, _ = self.cross_attention(
            doc_encoded,
            context_representations,
            context_representations,
            key_padding_mask=(mask == 0)
        )

        # Gated fusion
        gate = self.context_gate(torch.cat([doc_encoded, attended], dim=-1))
        fused = gate * attended + (1 - gate) * doc_encoded

        # Output projection
        output = self.output_projection(fused)

        # Apply normalization
        if self.normalize:
            output = F.normalize(output, p=2, dim=-1)

        # Pool if needed
        return self.pool_embeddings(output, mask, pooling='mean')

    def compute_adversarial_loss(
            self,
            embeddings: torch.Tensor,
            position_predictor: nn.Module,
            positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute adversarial loss for position-agnostic training.

        Args:
            embeddings: Document embeddings
            position_predictor: Network trying to predict positions
            positions: True positions

        Returns:
            Adversarial loss
        """
        # Try to make embeddings position-agnostic
        predicted_positions = position_predictor(embeddings.detach())

        # Adversarial loss - maximize confusion
        loss = -F.cross_entropy(predicted_positions, positions)

        return loss