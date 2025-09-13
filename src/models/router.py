"""
MoE Router with learned routing
Based on "Your Mixture-of-Experts LLM Is Secretly an Embedding Model"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LearnedMoERouter(nn.Module):
    """
    Learned routing mechanism for Mixture of Experts
    Key insights from research:
    1. Routing weights themselves are useful features
    2. Sparse routing (top-k) is more efficient
    3. Load balancing is important for training
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Lightweight encoder for routing decisions
        logger.info(f"Loading router encoder: {config.encoder_model}")
        self.encoder = SentenceTransformer(config.encoder_model)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder_dim = config.encoder_dim
        self.hidden_dim = config.hidden_dim
        self.num_experts = len(config.experts) if hasattr(config, 'experts') else 5
        self.top_k = config.top_k

        # Routing network with residual connections
        self.router_layers = nn.ModuleList()

        # First layer
        self.router_layers.append(nn.Sequential(
            nn.Linear(self.encoder_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        ))

        # Additional layers with residual connections
        for _ in range(config.num_layers - 1):
            self.router_layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ))

        # Final routing head
        self.routing_head = nn.Linear(self.hidden_dim, self.num_experts)

        # Learnable temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1) * config.temperature)

        # Load balancing parameters
        self.use_load_balancing = config.use_load_balancing
        self.load_balance_alpha = config.load_balance_alpha

        # Noise parameters for exploration
        self.routing_noise = config.routing_noise
        self.noise_epsilon = config.noise_epsilon

        # Statistics tracking
        self.register_buffer('expert_usage', torch.zeros(self.num_experts))
        self.register_buffer('routing_entropy', torch.zeros(1))

    def forward(self,
                texts: List[str],
                return_auxiliary: bool = False,
                force_expert: Optional[int] = None) -> Union[torch.Tensor, Tuple]:
        """
        Route texts to experts

        Args:
            texts: List of input texts
            return_auxiliary: Whether to return auxiliary losses and stats
            force_expert: Force routing to specific expert (for analysis)

        Returns:
            routing_weights: Tensor of shape [batch_size, num_experts]
            auxiliary_output: Dict with auxiliary losses and statistics (if requested)
        """
        batch_size = len(texts)

        # Encode texts with lightweight model
        with torch.no_grad():
            text_features = self.encoder.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

        # Pass through routing network
        hidden = text_features
        for i, layer in enumerate(self.router_layers):
            if i == 0:
                hidden = layer(hidden)
            else:
                # Residual connection for deeper layers
                hidden = hidden + layer(hidden)

        # Get routing logits
        routing_logits = self.routing_head(hidden)

        # Add noise during training for exploration
        if self.training and self.routing_noise > 0:
            noise = torch.randn_like(routing_logits) * self.routing_noise
            routing_logits = routing_logits + noise

        # Force specific expert if requested (for analysis)
        if force_expert is not None:
            routing_logits = torch.full_like(routing_logits, -1e9)
            routing_logits[:, force_expert] = 1e9

        # Apply temperature-scaled softmax
        temperature = self.temperature.clamp(min=0.1, max=10.0)
        routing_weights = F.softmax(routing_logits / temperature, dim=-1)

        # Apply top-k sparsity
        if self.top_k < self.num_experts:
            topk_vals, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)

            # Create sparse weights
            sparse_weights = torch.zeros_like(routing_weights)
            sparse_weights.scatter_(1, topk_indices, topk_vals)

            # Renormalize
            routing_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-10)

        # Update statistics
        with torch.no_grad():
            # Expert usage (for monitoring load balance)
            self.expert_usage = 0.9 * self.expert_usage + 0.1 * routing_weights.mean(dim=0)

            # Routing entropy (for monitoring routing diversity)
            entropy = -torch.sum(routing_weights * torch.log(routing_weights + 1e-10), dim=-1)
            self.routing_entropy = 0.9 * self.routing_entropy + 0.1 * entropy.mean()

        if return_auxiliary:
            auxiliary_output = {}

            # Load balancing loss
            if self.training and self.use_load_balancing:
                # Encourage uniform expert usage
                expert_usage = routing_weights.mean(dim=0)
                uniform_distribution = torch.ones_like(expert_usage) / self.num_experts

                # KL divergence from uniform distribution
                load_balance_loss = self.load_balance_alpha * F.kl_div(
                    torch.log(expert_usage + 1e-10),
                    uniform_distribution,
                    reduction='batchmean'
                )
                auxiliary_output['load_balance_loss'] = load_balance_loss

            # Diversity loss (encourage different routing for different samples)
            if self.training:
                # Compute pairwise similarity of routing weights
                routing_similarity = torch.matmul(routing_weights, routing_weights.t())

                # Penalize high similarity (except diagonal)
                mask = 1 - torch.eye(batch_size, device=routing_weights.device)
                diversity_loss = (routing_similarity * mask).mean() * 0.01
                auxiliary_output['diversity_loss'] = diversity_loss

            # Statistics for monitoring
            auxiliary_output['stats'] = {
                'expert_usage': self.expert_usage.detach().cpu().numpy(),
                'routing_entropy': self.routing_entropy.item(),
                'temperature': temperature.item(),
                'sparsity': (routing_weights > 0.01).float().mean().item()
            }

            return routing_weights, auxiliary_output

        return routing_weights

    def get_routing_analysis(self, texts: List[str]) -> Dict:
        """
        Analyze routing patterns for given texts
        Useful for understanding model behavior
        """
        self.eval()

        with torch.no_grad():
            routing_weights = self.forward(texts)

            # Get top experts for each text
            top_experts = torch.topk(routing_weights, k=min(3, self.num_experts), dim=-1)

            # Calculate entropy
            entropy = -torch.sum(routing_weights * torch.log(routing_weights + 1e-10), dim=-1)

            # Find dominant expert
            dominant_expert = torch.argmax(routing_weights, dim=-1)

            analysis = {
                'routing_weights': routing_weights.cpu().numpy(),
                'top_experts': {
                    'indices': top_experts.indices.cpu().numpy(),
                    'weights': top_experts.values.cpu().numpy()
                },
                'entropy': entropy.cpu().numpy(),
                'dominant_expert': dominant_expert.cpu().numpy(),
                'expert_usage': self.expert_usage.cpu().numpy(),
                'average_entropy': self.routing_entropy.item()
            }

        return analysis

    def reset_statistics(self):
        """Reset tracking statistics"""
        self.expert_usage.zero_()
        self.routing_entropy.zero_()
        logger.info("Router statistics reset")