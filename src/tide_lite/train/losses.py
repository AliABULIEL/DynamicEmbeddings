"""Loss functions for TIDE-Lite training.

This module provides loss functions for training temporal embeddings:
- Cosine regression loss for similarity learning
- Temporal consistency loss for temporal smoothness
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def cosine_regression_loss(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    gold_scores_0to5: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity regression loss for STS-B.
    
    Computes MSE between cosine similarities and normalized gold scores.
    Gold scores are scaled from [0, 5] to [0, 1] range.
    
    Args:
        emb1: First embeddings [batch_size, embedding_dim].
        emb2: Second embeddings [batch_size, embedding_dim].
        gold_scores_0to5: Gold similarity scores in [0, 5] range [batch_size].
        
    Returns:
        Scalar loss value.
    """
    # Normalize embeddings
    emb1_norm = F.normalize(emb1, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = torch.sum(emb1_norm * emb2_norm, dim=1)
    
    # Scale gold scores from [0, 5] to [0, 1]
    gold_scores_normalized = gold_scores_0to5 / 5.0
    
    # MSE loss
    loss = F.mse_loss(cosine_sim, gold_scores_normalized)
    
    return loss


def temporal_consistency_loss(
    embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    tau_seconds: float = 86400.0,  # 1 day default
) -> torch.Tensor:
    """Temporal consistency loss for smooth evolution over time.
    
    Encourages embeddings to change smoothly over time using exponential decay:
    L = sum_{i,j} sim(e_i, e_j) * exp(-|t_i - t_j| / tau)
    
    Args:
        embeddings: Embeddings [batch_size, embedding_dim].
        timestamps: Unix timestamps in seconds [batch_size].
        tau_seconds: Time constant for exponential decay (seconds).
        
    Returns:
        Scalar loss value (negative for minimization).
    """
    batch_size = embeddings.shape[0]
    
    if batch_size < 2:
        # Need at least 2 samples for pairwise comparison
        return torch.tensor(0.0, device=embeddings.device)
    
    # Strict timestamp validation
    if not isinstance(timestamps, torch.Tensor):
        raise ValueError(f"timestamps must be a torch.Tensor, got {type(timestamps)}")
    
    if timestamps.dtype not in [torch.float32, torch.float64]:
        raise ValueError(f"timestamps must be float tensor, got {timestamps.dtype}")
    
    if timestamps.device != embeddings.device:
        raise ValueError(
            f"timestamps and embeddings must be on same device. "
            f"Got timestamps on {timestamps.device}, embeddings on {embeddings.device}"
        )
    
    # Ensure timestamps are 1D
    if timestamps.dim() == 2 and timestamps.shape[1] == 1:
        timestamps = timestamps.squeeze(1)
    elif timestamps.dim() != 1:
        raise ValueError(
            f"timestamps must be 1D tensor of shape [batch_size]. "
            f"Got shape {timestamps.shape} with batch_size={batch_size}"
        )
    
    if timestamps.shape[0] != batch_size:
        raise ValueError(
            f"timestamps batch size ({timestamps.shape[0]}) must match "
            f"embeddings batch size ({batch_size})"
        )
    
    # Normalize embeddings for cosine similarity
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    # Compute pairwise cosine similarities
    # Shape: [batch_size, batch_size]
    similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.t())
    
    # Compute pairwise time differences with safe broadcasting
    # Shape: [batch_size, 1] - [1, batch_size] = [batch_size, batch_size]
    time_diff = torch.abs(
        timestamps.unsqueeze(1) - timestamps.unsqueeze(0)
    )
    
    # Compute temporal weights with exponential decay
    # Higher weight for temporally close pairs
    temporal_weights = torch.exp(-time_diff / tau_seconds)
    
    # Mask diagonal (self-similarity)
    mask = torch.eye(batch_size, device=embeddings.device).bool()
    temporal_weights = temporal_weights.masked_fill(mask, 0)
    similarity_matrix = similarity_matrix.masked_fill(mask, 0)
    
    # Weighted temporal consistency
    # We want to maximize similarity for temporally close pairs
    weighted_similarity = similarity_matrix * temporal_weights
    
    # Normalize by number of pairs
    num_pairs = batch_size * (batch_size - 1)
    if num_pairs > 0:
        loss = -weighted_similarity.sum() / num_pairs
    else:
        loss = torch.tensor(0.0, device=embeddings.device)
    
    return loss


def preservation_loss(
    temporal_embeddings: torch.Tensor,
    base_embeddings: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Preservation loss to prevent excessive deviation from base embeddings.
    
    Args:
        temporal_embeddings: Modulated embeddings [batch_size, embedding_dim].
        base_embeddings: Original embeddings [batch_size, embedding_dim].
        alpha: Weight for L2 regularization.
        
    Returns:
        Scalar loss value.
    """
    # L2 distance between temporal and base embeddings
    diff = temporal_embeddings - base_embeddings
    loss = alpha * torch.mean(torch.sum(diff ** 2, dim=1))
    
    return loss


def combined_tide_loss(
    temporal_emb1: torch.Tensor,
    temporal_emb2: torch.Tensor,
    base_emb1: torch.Tensor,
    base_emb2: torch.Tensor,
    timestamps1: torch.Tensor,
    timestamps2: torch.Tensor,
    gold_scores: torch.Tensor,
    temporal_weight: float = 0.1,
    preservation_weight: float = 0.05,
    tau_seconds: float = 86400.0,
) -> dict:
    """Combined loss for TIDE-Lite training.
    
    Combines:
    - Cosine regression loss (primary task)
    - Temporal consistency loss (temporal smoothness)
    - Preservation loss (prevent excessive deviation)
    
    Args:
        temporal_emb1: First temporal embeddings [batch_size, embedding_dim].
        temporal_emb2: Second temporal embeddings [batch_size, embedding_dim].
        base_emb1: First base embeddings [batch_size, embedding_dim].
        base_emb2: Second base embeddings [batch_size, embedding_dim].
        timestamps1: First timestamps [batch_size].
        timestamps2: Second timestamps [batch_size].
        gold_scores: Gold similarity scores [batch_size].
        temporal_weight: Weight for temporal consistency loss.
        preservation_weight: Weight for preservation loss.
        tau_seconds: Time constant for temporal consistency.
        
    Returns:
        Dictionary with:
            - total: Combined loss
            - cosine_loss: Cosine regression component
            - temporal_loss: Temporal consistency component
            - preservation_loss: Preservation component
    """
    # Primary task: cosine regression
    cosine_loss = cosine_regression_loss(temporal_emb1, temporal_emb2, gold_scores)
    
    # Temporal consistency for both sets of embeddings
    temporal_loss1 = temporal_consistency_loss(temporal_emb1, timestamps1, tau_seconds)
    temporal_loss2 = temporal_consistency_loss(temporal_emb2, timestamps2, tau_seconds)
    temporal_loss = (temporal_loss1 + temporal_loss2) / 2
    
    # Preservation loss
    preserve_loss1 = preservation_loss(temporal_emb1, base_emb1)
    preserve_loss2 = preservation_loss(temporal_emb2, base_emb2)
    preserve_loss = (preserve_loss1 + preserve_loss2) / 2
    
    # Combined loss
    total_loss = (
        cosine_loss +
        temporal_weight * temporal_loss +
        preservation_weight * preserve_loss
    )
    
    return {
        "total": total_loss,
        "cosine_loss": cosine_loss.detach(),
        "temporal_loss": temporal_loss.detach(),
        "preservation_loss": preserve_loss.detach(),
    }


class TIDELiteLoss:
    """Loss wrapper for TIDE-Lite training with configurable components."""
    
    def __init__(
        self,
        temporal_weight: float = 0.1,
        preservation_weight: float = 0.05,
        tau_seconds: float = 86400.0,
    ) -> None:
        """Initialize loss configuration.
        
        Args:
            temporal_weight: Weight for temporal consistency loss.
            preservation_weight: Weight for preservation loss.
            tau_seconds: Time constant for temporal consistency.
        """
        self.temporal_weight = temporal_weight
        self.preservation_weight = preservation_weight
        self.tau_seconds = tau_seconds
        
        logger.info(
            f"Initialized TIDELiteLoss with weights: "
            f"temporal={temporal_weight}, preservation={preservation_weight}, "
            f"tau={tau_seconds}s"
        )
    
    def __call__(
        self,
        temporal_emb1: torch.Tensor,
        temporal_emb2: torch.Tensor,
        base_emb1: torch.Tensor,
        base_emb2: torch.Tensor,
        timestamps1: torch.Tensor,
        timestamps2: torch.Tensor,
        gold_scores: torch.Tensor,
    ) -> dict:
        """Compute combined loss.
        
        Args:
            temporal_emb1: First temporal embeddings [batch_size, embedding_dim].
            temporal_emb2: Second temporal embeddings [batch_size, embedding_dim].
            base_emb1: First base embeddings [batch_size, embedding_dim].
            base_emb2: Second base embeddings [batch_size, embedding_dim].
            timestamps1: First timestamps [batch_size].
            timestamps2: Second timestamps [batch_size].
            gold_scores: Gold similarity scores [batch_size].
            
        Returns:
            Dictionary with loss components.
        """
        return combined_tide_loss(
            temporal_emb1=temporal_emb1,
            temporal_emb2=temporal_emb2,
            base_emb1=base_emb1,
            base_emb2=base_emb2,
            timestamps1=timestamps1,
            timestamps2=timestamps2,
            gold_scores=gold_scores,
            temporal_weight=self.temporal_weight,
            preservation_weight=self.preservation_weight,
            tau_seconds=self.tau_seconds,
        )
