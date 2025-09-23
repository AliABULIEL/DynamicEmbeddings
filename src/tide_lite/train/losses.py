"""Loss functions for TIDE-Lite training.

This module provides loss functions for similarity regression
and temporal consistency objectives.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.common import cosine_similarity_matrix

logger = logging.getLogger(__name__)


def cosine_regression_loss(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
    gold_scores: torch.Tensor,
    score_range: Tuple[float, float] = (0.0, 5.0),
) -> torch.Tensor:
    """Compute cosine similarity regression loss for STS-B.
    
    This loss optimizes embeddings to match gold similarity scores
    through cosine similarity. The gold scores are normalized to [-1, 1]
    to match the cosine similarity range.
    
    Args:
        embeddings1: First set of embeddings [batch_size, hidden_dim].
        embeddings2: Second set of embeddings [batch_size, hidden_dim].
        gold_scores: Gold similarity scores [batch_size], typically in [0, 5].
        score_range: Original range of gold scores for normalization.
        
    Returns:
        Mean squared error loss between predicted and gold similarities.
        
    Note:
        STS-B scores range from 0-5, which we linearly map to cosine range [-1, 1].
        This preserves relative similarities while matching the natural output range.
    """
    # Normalize embeddings for cosine similarity
    embeddings1_norm = F.normalize(embeddings1, p=2, dim=1)
    embeddings2_norm = F.normalize(embeddings2, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = torch.sum(embeddings1_norm * embeddings2_norm, dim=1)
    
    # Normalize gold scores to [-1, 1]
    min_score, max_score = score_range
    normalized_gold = 2.0 * (gold_scores - min_score) / (max_score - min_score) - 1.0
    normalized_gold = torch.clamp(normalized_gold, -1.0, 1.0)
    
    # MSE loss between predicted and gold similarities
    loss = F.mse_loss(cosine_sim, normalized_gold)
    
    # Log statistics for debugging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Cosine loss: {loss.item():.4f}, "
            f"Sim range: [{cosine_sim.min().item():.3f}, {cosine_sim.max().item():.3f}], "
            f"Gold range: [{normalized_gold.min().item():.3f}, {normalized_gold.max().item():.3f}]"
        )
    
    return loss


def temporal_consistency_loss(
    embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    tau_seconds: float = 86400.0,
    distance_metric: str = "cosine",
    weight_type: str = "exponential",
) -> torch.Tensor:
    """Compute temporal consistency loss for embeddings.
    
    This loss encourages embeddings from similar times to be similar,
    with similarity weighted by temporal distance. The intuition is that
    content from nearby times should have related representations.
    
    Args:
        embeddings: Embedding vectors [batch_size, hidden_dim].
        timestamps: Unix timestamps in seconds [batch_size].
        tau_seconds: Time constant for exponential decay (default: 1 day).
        distance_metric: Distance metric ('cosine' or 'l2').
        weight_type: Weighting scheme ('exponential', 'linear', 'step').
        
    Returns:
        Temporal consistency loss scalar.
        
    Note:
        The loss encourages:
        - High similarity for embeddings close in time
        - Lower similarity for embeddings far apart in time
        - Smooth transitions in embedding space over time
    """
    batch_size = embeddings.shape[0]
    
    if batch_size < 2:
        # Need at least 2 samples for pairwise comparison
        return torch.tensor(0.0, device=embeddings.device)
    
    # Compute pairwise temporal distances
    time_diff = torch.abs(
        timestamps.unsqueeze(1) - timestamps.unsqueeze(0)
    )  # [batch, batch]
    
    # Compute temporal weights based on time difference
    if weight_type == "exponential":
        # Exponential decay: nearby times have high weight
        temporal_weights = torch.exp(-time_diff / tau_seconds)
    elif weight_type == "linear":
        # Linear decay up to tau_seconds
        temporal_weights = torch.clamp(1.0 - time_diff / tau_seconds, min=0.0)
    elif weight_type == "step":
        # Binary: 1 if within tau_seconds, 0 otherwise
        temporal_weights = (time_diff < tau_seconds).float()
    else:
        raise ValueError(f"Unknown weight type: {weight_type}")
    
    # Zero out diagonal (self-comparisons)
    temporal_weights.fill_diagonal_(0.0)
    
    # Normalize weights
    weight_sum = temporal_weights.sum()
    if weight_sum > 0:
        temporal_weights = temporal_weights / weight_sum
    else:
        # All samples too far apart
        return torch.tensor(0.0, device=embeddings.device)
    
    # Compute pairwise embedding similarities/distances
    if distance_metric == "cosine":
        # Cosine similarity (higher is more similar)
        embedding_sim = cosine_similarity_matrix(embeddings)
        
        # Loss encourages high similarity when temporal weight is high
        # We want sim=1 when weight=1, sim=0 when weight=0
        loss = torch.sum(temporal_weights * (1.0 - embedding_sim)) 
        
    elif distance_metric == "l2":
        # L2 distance (lower is more similar)
        embedding_dist = torch.cdist(embeddings, embeddings, p=2)
        
        # Loss encourages low distance when temporal weight is high
        loss = torch.sum(temporal_weights * embedding_dist)
        
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    # Log statistics
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Temporal loss: {loss.item():.4f}, "
            f"Time range: {time_diff.max().item()/3600:.1f} hours, "
            f"Effective pairs: {(temporal_weights > 0.01).sum().item()}"
        )
    
    return loss


def combined_tide_loss(
    temporal_embeddings: torch.Tensor,
    base_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    target_embeddings: Optional[torch.Tensor] = None,
    target_scores: Optional[torch.Tensor] = None,
    alpha: float = 0.1,
    beta: float = 0.05,
    tau_seconds: float = 86400.0,
) -> Tuple[torch.Tensor, dict]:
    """Combined loss for TIDE-Lite training.
    
    Combines multiple objectives:
    1. Task loss (similarity regression or embedding matching)
    2. Temporal consistency loss
    3. Optional: base embedding preservation regularization
    
    Args:
        temporal_embeddings: Temporally modulated embeddings [batch, hidden_dim].
        base_embeddings: Original embeddings without modulation [batch, hidden_dim].
        timestamps: Unix timestamps [batch].
        target_embeddings: Target embeddings for matching (optional).
        target_scores: Target similarity scores for regression (optional).
        alpha: Weight for temporal consistency loss.
        beta: Weight for base preservation regularization.
        tau_seconds: Time constant for temporal consistency.
        
    Returns:
        Tuple of:
            - Combined loss scalar
            - Dictionary of individual loss components
            
    Raises:
        ValueError: If neither target_embeddings nor target_scores provided.
    """
    loss_components = {}
    
    # Task loss (either regression or embedding matching)
    if target_scores is not None:
        # For paired data (STS-B style)
        if temporal_embeddings.shape[0] % 2 != 0:
            raise ValueError("Batch size must be even for paired similarity tasks")
        
        batch_size = temporal_embeddings.shape[0] // 2
        emb1 = temporal_embeddings[:batch_size]
        emb2 = temporal_embeddings[batch_size:]
        
        task_loss = cosine_regression_loss(emb1, emb2, target_scores)
        loss_components["task_loss"] = task_loss.item()
        
    elif target_embeddings is not None:
        # Direct embedding matching (e.g., distillation)
        task_loss = F.mse_loss(temporal_embeddings, target_embeddings)
        loss_components["task_loss"] = task_loss.item()
        
    else:
        raise ValueError("Must provide either target_scores or target_embeddings")
    
    # Temporal consistency loss
    temporal_loss = temporal_consistency_loss(
        temporal_embeddings,
        timestamps,
        tau_seconds=tau_seconds,
    )
    loss_components["temporal_loss"] = temporal_loss.item()
    
    # Base preservation regularization (prevents excessive modulation)
    if beta > 0:
        preservation_loss = F.mse_loss(temporal_embeddings, base_embeddings)
        loss_components["preservation_loss"] = preservation_loss.item()
    else:
        preservation_loss = 0.0
        loss_components["preservation_loss"] = 0.0
    
    # Combine losses
    total_loss = task_loss + alpha * temporal_loss + beta * preservation_loss
    loss_components["total_loss"] = total_loss.item()
    
    logger.debug(
        f"Loss breakdown - Task: {loss_components['task_loss']:.4f}, "
        f"Temporal: {loss_components['temporal_loss']:.4f}, "
        f"Preservation: {loss_components['preservation_loss']:.4f}"
    )
    
    return total_loss, loss_components


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning embeddings from positive/negative pairs.
    
    This can be used as an alternative to regression loss when
    we have binary similarity labels instead of continuous scores.
    """
    
    def __init__(
        self,
        margin: float = 0.5,
        distance_metric: str = "cosine",
    ) -> None:
        """Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs.
            distance_metric: Distance metric ('cosine' or 'l2').
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        logger.info(f"Initialized contrastive loss with margin={margin}")
    
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            embeddings1: First embeddings [batch_size, hidden_dim].
            embeddings2: Second embeddings [batch_size, hidden_dim].
            labels: Binary labels (1=similar, 0=dissimilar) [batch_size].
            
        Returns:
            Contrastive loss scalar.
        """
        if self.distance_metric == "cosine":
            # Cosine distance (1 - cosine_similarity)
            emb1_norm = F.normalize(embeddings1, p=2, dim=1)
            emb2_norm = F.normalize(embeddings2, p=2, dim=1)
            distances = 1.0 - torch.sum(emb1_norm * emb2_norm, dim=1)
        elif self.distance_metric == "l2":
            # L2 distance
            distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Contrastive loss
        positive_loss = labels * distances.pow(2)
        negative_loss = (1 - labels) * F.relu(self.margin - distances).pow(2)
        
        loss = torch.mean(positive_loss + negative_loss) / 2
        
        return loss


class TripletLoss(nn.Module):
    """Triplet loss for learning embeddings from anchor-positive-negative triplets.
    
    Useful when we have relative similarity information rather than
    absolute similarity scores.
    """
    
    def __init__(
        self,
        margin: float = 0.2,
        distance_metric: str = "cosine",
    ) -> None:
        """Initialize triplet loss.
        
        Args:
            margin: Margin between positive and negative distances.
            distance_metric: Distance metric ('cosine' or 'l2').
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        logger.info(f"Initialized triplet loss with margin={margin}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, hidden_dim].
            positive: Positive embeddings [batch_size, hidden_dim].
            negative: Negative embeddings [batch_size, hidden_dim].
            
        Returns:
            Triplet loss scalar.
        """
        if self.distance_metric == "cosine":
            # Cosine distance
            anchor_norm = F.normalize(anchor, p=2, dim=1)
            positive_norm = F.normalize(positive, p=2, dim=1)
            negative_norm = F.normalize(negative, p=2, dim=1)
            
            pos_dist = 1.0 - torch.sum(anchor_norm * positive_norm, dim=1)
            neg_dist = 1.0 - torch.sum(anchor_norm * negative_norm, dim=1)
        elif self.distance_metric == "l2":
            # L2 distance
            pos_dist = torch.norm(anchor - positive, p=2, dim=1)
            neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Triplet loss with margin
        loss = torch.mean(F.relu(pos_dist - neg_dist + self.margin))
        
        return loss
