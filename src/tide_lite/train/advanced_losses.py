"""Advanced temporal loss functions for dynamic embeddings.

This module provides sophisticated loss functions that model real
temporal dynamics instead of assuming smooth exponential decay.
"""

import logging
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SemanticDriftLoss(nn.Module):
    """Loss function that models realistic semantic drift patterns.
    
    Instead of exponential decay, this models different types of drift:
    - Sudden shifts (events, discoveries)
    - Gradual evolution (language change)
    - Cyclic patterns (seasonal topics)
    - Plateau effects (stable periods)
    """
    
    def __init__(
        self,
        drift_model: str = "adaptive",
        min_drift: float = 0.0,
        max_drift: float = 1.0,
    ):
        """Initialize semantic drift loss.
        
        Args:
            drift_model: Type of drift model ('adaptive', 'piecewise', 'learned')
            min_drift: Minimum expected drift
            max_drift: Maximum expected drift
        """
        super().__init__()
        self.drift_model = drift_model
        self.min_drift = min_drift
        self.max_drift = max_drift
        
        if drift_model == "learned":
            # Learn drift patterns from data
            self.drift_predictor = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
    
    def compute_expected_drift(
        self,
        time_delta: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Compute expected semantic drift based on time difference.
        
        Args:
            time_delta: Time differences in seconds [batch_size, batch_size]
            metadata: Optional metadata about text pairs (e.g., domain, topic)
        
        Returns:
            Expected drift values between 0 and 1
        """
        # Convert to days for easier reasoning
        days_delta = time_delta / 86400.0
        
        if self.drift_model == "adaptive":
            # Adaptive model with different phases
            # Quick initial drift, then slower evolution
            quick_drift = torch.tanh(days_delta / 7.0) * 0.3  # 30% drift in first week
            slow_drift = torch.log1p(days_delta / 365.0) * 0.5  # Logarithmic long-term
            seasonal = torch.sin(days_delta * 2 * torch.pi / 365.0) * 0.1  # Seasonal variation
            
            drift = quick_drift + slow_drift + torch.abs(seasonal)
            
        elif self.drift_model == "piecewise":
            # Piecewise linear with plateaus
            drift = torch.zeros_like(days_delta)
            
            # Different drift rates for different time scales
            hour_mask = days_delta < 1/24
            day_mask = (days_delta >= 1/24) & (days_delta < 1)
            week_mask = (days_delta >= 1) & (days_delta < 7)
            month_mask = (days_delta >= 7) & (days_delta < 30)
            year_mask = days_delta >= 30
            
            drift[hour_mask] = days_delta[hour_mask] * 24 * 0.01  # 1% per hour initially
            drift[day_mask] = 0.01 + (days_delta[day_mask] - 1/24) * 0.05  # 5% per day
            drift[week_mask] = 0.06 + (days_delta[week_mask] - 1) * 0.02  # 2% per day after
            drift[month_mask] = 0.18 + (days_delta[month_mask] - 7) * 0.005  # 0.5% per day
            drift[year_mask] = 0.30 + torch.log1p((days_delta[year_mask] - 30) / 365) * 0.3
            
        elif self.drift_model == "learned":
            # Use neural network to predict drift
            time_input = (days_delta / 365.0).unsqueeze(-1)  # Normalize to years
            drift = self.drift_predictor(time_input).squeeze(-1)
        
        else:
            # Fallback to simple log model
            drift = torch.log1p(days_delta / 30.0) / 3.0
        
        # Clamp to valid range
        drift = torch.clamp(drift, self.min_drift, self.max_drift)
        
        return drift
    
    def forward(
        self,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Compute semantic drift loss.
        
        Args:
            embeddings: Embedding vectors [batch_size, hidden_dim]
            timestamps: Unix timestamps [batch_size]
            metadata: Optional metadata about texts
        
        Returns:
            Semantic drift loss
        """
        batch_size = embeddings.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Compute pairwise time differences
        time_diff = torch.abs(
            timestamps.unsqueeze(1) - timestamps.unsqueeze(0)
        )
        
        # Compute expected drift
        expected_drift = self.compute_expected_drift(time_diff, metadata)
        
        # Compute actual embedding distances
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        cosine_sim = torch.matmul(embeddings_norm, embeddings_norm.T)
        actual_drift = (1.0 - cosine_sim) / 2.0  # Scale to [0, 1]
        
        # Mask diagonal (self-comparisons)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        expected_drift = expected_drift * mask
        actual_drift = actual_drift * mask
        
        # Compute loss: penalize deviation from expected drift
        drift_loss = F.mse_loss(actual_drift[mask], expected_drift[mask])
        
        # Log statistics
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Drift loss: {drift_loss.item():.4f}, "
                f"Expected drift: {expected_drift[mask].mean().item():.3f}, "
                f"Actual drift: {actual_drift[mask].mean().item():.3f}"
            )
        
        return drift_loss


class EventAwareTemporalLoss(nn.Module):
    """Temporal loss that accounts for discrete events and discontinuities.
    
    Real temporal evolution isn't smooth - it has jumps corresponding to:
    - Major events (pandemics, wars, discoveries)
    - Viral phenomena (memes, trends)
    - Technological breakthroughs
    - Cultural shifts
    """
    
    def __init__(
        self,
        event_impact_scale: float = 0.5,
        event_decay_rate: float = 0.1,
    ):
        """Initialize event-aware temporal loss.
        
        Args:
            event_impact_scale: Maximum impact of an event on embeddings
            event_decay_rate: How quickly event impact decays
        """
        super().__init__()
        self.event_impact_scale = event_impact_scale
        self.event_decay_rate = event_decay_rate
        
        # Known events that cause discontinuities
        # In practice, these would be learned or provided
        self.major_events = [
            # (timestamp, impact_magnitude, decay_rate)
            (datetime(2020, 3, 11).timestamp(), 0.8, 0.05),  # COVID pandemic
            (datetime(2022, 11, 30).timestamp(), 0.6, 0.2),   # ChatGPT launch
            (datetime(2022, 2, 24).timestamp(), 0.7, 0.1),    # Ukraine invasion
        ]
    
    def compute_event_impact(
        self,
        timestamp1: torch.Tensor,
        timestamp2: torch.Tensor
    ) -> torch.Tensor:
        """Compute impact of events on embedding similarity.
        
        Args:
            timestamp1: First timestamp [batch_size]
            timestamp2: Second timestamp [batch_size]
        
        Returns:
            Event impact factor [batch_size]
        """
        impact = torch.zeros_like(timestamp1)
        
        for event_time, magnitude, decay in self.major_events:
            # Check if event occurred between the two timestamps
            event_between = (
                (timestamp1 < event_time) & (timestamp2 > event_time) |
                (timestamp2 < event_time) & (timestamp1 > event_time)
            )
            
            # Compute time since event for both timestamps
            time_since_1 = torch.abs(timestamp1 - event_time)
            time_since_2 = torch.abs(timestamp2 - event_time)
            
            # Event creates a discontinuity
            event_effect = magnitude * torch.exp(-decay * time_since_1 / 86400)
            event_effect += magnitude * torch.exp(-decay * time_since_2 / 86400)
            
            # Add discontinuity bonus if event is between timestamps
            event_effect[event_between] += magnitude * 0.5
            
            impact = torch.maximum(impact, event_effect)
        
        return torch.clamp(impact, 0, self.event_impact_scale)
    
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        timestamps1: torch.Tensor,
        timestamps2: torch.Tensor
    ) -> torch.Tensor:
        """Compute event-aware temporal loss.
        
        Args:
            embeddings1: First set of embeddings [batch_size, hidden_dim]
            embeddings2: Second set of embeddings [batch_size, hidden_dim]
            timestamps1: First timestamps [batch_size]
            timestamps2: Second timestamps [batch_size]
        
        Returns:
            Event-aware temporal loss
        """
        # Base temporal difference
        time_diff = torch.abs(timestamps2 - timestamps1)
        
        # Compute event impacts
        event_impact = self.compute_event_impact(timestamps1, timestamps2)
        
        # Expected distance includes both gradual drift and event impacts
        gradual_drift = torch.log1p(time_diff / 86400 / 30) * 0.2  # 20% per month log
        expected_distance = gradual_drift + event_impact
        expected_distance = torch.clamp(expected_distance, 0, 1)
        
        # Actual embedding distance
        emb1_norm = F.normalize(embeddings1, p=2, dim=1)
        emb2_norm = F.normalize(embeddings2, p=2, dim=1)
        actual_distance = (1 - torch.sum(emb1_norm * emb2_norm, dim=1)) / 2
        
        # Loss: match expected temporal evolution
        loss = F.mse_loss(actual_distance, expected_distance)
        
        return loss


class CyclicTemporalLoss(nn.Module):
    """Model cyclic/seasonal patterns in temporal evolution.
    
    Many concepts have seasonal or cyclic patterns:
    - Holiday-related topics
    - Sports seasons
    - Academic cycles
    - Weather-related discussions
    """
    
    def __init__(
        self,
        cycles: List[float] = [365.25, 7.0],  # Yearly and weekly cycles
        cycle_strength: float = 0.2
    ):
        """Initialize cyclic temporal loss.
        
        Args:
            cycles: List of cycle periods in days
            cycle_strength: Maximum impact of cycles
        """
        super().__init__()
        self.cycles = cycles
        self.cycle_strength = cycle_strength
    
    def forward(
        self,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """Compute cyclic temporal loss.
        
        Args:
            embeddings: Embeddings [batch_size, hidden_dim]
            timestamps: Unix timestamps [batch_size]
        
        Returns:
            Cyclic temporal loss
        """
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Convert timestamps to days
        days = timestamps / 86400.0
        
        # Compute cyclic similarity for each cycle period
        cyclic_sim = torch.zeros((batch_size, batch_size), device=embeddings.device)
        
        for cycle_days in self.cycles:
            # Compute phase difference in cycle
            phase_diff = torch.abs(
                torch.remainder(days.unsqueeze(1), cycle_days) -
                torch.remainder(days.unsqueeze(0), cycle_days)
            )
            
            # Minimum phase difference (accounting for wrap-around)
            phase_diff = torch.minimum(phase_diff, cycle_days - phase_diff)
            
            # Convert to similarity (similar phase = high similarity)
            phase_sim = 1.0 - (phase_diff / (cycle_days / 2))
            cyclic_sim += phase_sim / len(self.cycles)
        
        # Scale cyclic similarity
        cyclic_sim = cyclic_sim * self.cycle_strength
        
        # Compute actual embedding similarity
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        actual_sim = torch.matmul(emb_norm, emb_norm.T)
        
        # Mask diagonal
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        
        # Loss: embeddings with similar cyclic phase should be more similar
        # Weight by cyclic strength (only apply where cycles matter)
        weight = cyclic_sim[mask].abs()
        target_sim = cyclic_sim[mask]
        actual = actual_sim[mask]
        
        loss = torch.mean(weight * (actual - target_sim) ** 2)
        
        return loss


from datetime import datetime

def advanced_temporal_loss(
    temporal_embeddings: torch.Tensor,
    base_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    target_scores: Optional[torch.Tensor] = None,
    metadata: Optional[Dict[str, Any]] = None,
    alpha_drift: float = 0.1,
    alpha_event: float = 0.05,
    alpha_cyclic: float = 0.02,
    beta_preservation: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined advanced temporal loss function.
    
    Args:
        temporal_embeddings: Temporally modulated embeddings
        base_embeddings: Original base embeddings
        timestamps: Unix timestamps
        target_scores: Target similarity scores (for task loss)
        metadata: Optional metadata about texts
        alpha_drift: Weight for semantic drift loss
        alpha_event: Weight for event-aware loss
        alpha_cyclic: Weight for cyclic patterns
        beta_preservation: Weight for base preservation
    
    Returns:
        Total loss and component dictionary
    """
    loss_components = {}
    
    # Task loss (if targets provided)
    if target_scores is not None:
        batch_size = temporal_embeddings.shape[0] // 2
        emb1 = temporal_embeddings[:batch_size]
        emb2 = temporal_embeddings[batch_size:]
        
        from .losses import cosine_regression_loss
        task_loss = cosine_regression_loss(emb1, emb2, target_scores)
        loss_components["task_loss"] = task_loss.item()
    else:
        task_loss = 0.0
        loss_components["task_loss"] = 0.0
    
    # Semantic drift loss
    drift_loss_fn = SemanticDriftLoss(drift_model="adaptive")
    drift_loss = drift_loss_fn(temporal_embeddings, timestamps, metadata)
    loss_components["drift_loss"] = drift_loss.item()
    
    # Event-aware loss (for paired data)
    if target_scores is not None:
        event_loss_fn = EventAwareTemporalLoss()
        ts1 = timestamps[:batch_size]
        ts2 = timestamps[batch_size:]
        event_loss = event_loss_fn(emb1, emb2, ts1, ts2)
        loss_components["event_loss"] = event_loss.item()
    else:
        event_loss = 0.0
        loss_components["event_loss"] = 0.0
    
    # Cyclic temporal loss
    cyclic_loss_fn = CyclicTemporalLoss()
    cyclic_loss = cyclic_loss_fn(temporal_embeddings, timestamps)
    loss_components["cyclic_loss"] = cyclic_loss.item()
    
    # Base preservation (prevent over-modulation)
    preservation_loss = F.mse_loss(temporal_embeddings, base_embeddings)
    loss_components["preservation_loss"] = preservation_loss.item()
    
    # Combine losses
    total_loss = (
        task_loss +
        alpha_drift * drift_loss +
        alpha_event * event_loss +
        alpha_cyclic * cyclic_loss +
        beta_preservation * preservation_loss
    )
    
    loss_components["total_loss"] = total_loss.item()
    
    return total_loss, loss_components
