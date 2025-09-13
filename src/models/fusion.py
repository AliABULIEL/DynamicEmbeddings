"""
AdapterFusion layers for combining expert embeddings
Based on "AdapterFusion: Non-Destructive Task Composition"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class AdapterFusionLayer(nn.Module):
    """
    Single AdapterFusion layer
    Combines expert embeddings using attention mechanism with gating
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim if hasattr(config, 'hidden_dim') else 768
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout
        self.use_gating = config.use_gating

        # Ensure hidden_dim is divisible by num_heads
        assert self.hidden_dim % self.num_heads == 0
        self.head_dim = self.hidden_dim // self.num_heads

        # Multi-head attention components
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        # Gating mechanism (optional)
        if self.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Sigmoid()
            )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self,
                expert_embeddings: torch.Tensor,
                routing_weights: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse expert embeddings using attention mechanism

        Args:
            expert_embeddings: [batch_size, num_experts, hidden_dim]
            routing_weights: [batch_size, num_experts]
            mask: Optional attention mask

        Returns:
            fused_embedding: [batch_size, hidden_dim]
        """
        batch_size, num_experts, hidden_dim = expert_embeddings.shape

        # Initial weighted combination using routing weights
        weighted_avg = torch.einsum('be,bed->bd', routing_weights, expert_embeddings)

        # Multi-head attention
        # Query from weighted average (what we're looking for)
        queries = self.query_proj(weighted_avg)  # [B, D]
        queries = queries.view(batch_size, 1, self.num_heads, self.head_dim)
        queries = queries.transpose(1, 2)  # [B, H, 1, D/H]

        # Keys and values from expert embeddings
        keys = self.key_proj(expert_embeddings)  # [B, E, D]
        keys = keys.view(batch_size, num_experts, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)  # [B, H, E, D/H]

        values = self.value_proj(expert_embeddings)  # [B, E, D]
        values = values.view(batch_size, num_experts, self.num_heads, self.head_dim)
        values = values.transpose(1, 2)  # [B, H, E, D/H]

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1))  # [B, H, 1, E]
        scores = scores / (self.head_dim ** 0.5)

        # Add routing weights as attention bias
        # This ensures attention is influenced by routing decisions
        routing_bias = routing_weights.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, E]
        routing_bias = torch.log(routing_bias + 1e-10)  # Log space for stability
        scores = scores + routing_bias

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, values)  # [B, H, 1, D/H]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, 1, H, D/H]
        attn_output = attn_output.view(batch_size, self.hidden_dim)  # [B, D]
        attn_output = self.output_proj(attn_output)

        # Gating mechanism (if enabled)
        if self.use_gating:
            # Decide how much of the attention output to use
            gate_input = torch.cat([weighted_avg, attn_output], dim=-1)
            gate = self.gate(gate_input)
            attn_output = gate * attn_output

        # Residual connection and layer norm
        output = self.layer_norm1(weighted_avg + self.dropout(attn_output))

        # Feed-forward network with residual
        ffn_output = self.ffn(output)
        output = self.layer_norm2(output + ffn_output)

        return output


class HierarchicalAdapterFusion(nn.Module):
    """
    Hierarchical AdapterFusion with multiple layers
    Progressively refines the fusion of expert embeddings
    """

    def __init__(self, config):
        super().__init__()

        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim if hasattr(config, 'hidden_dim') else 768

        # Stack of fusion layers
        self.fusion_layers = nn.ModuleList([
            AdapterFusionLayer(config) for _ in range(self.num_layers)
        ])

        # Optional: Cross-layer connections
        self.use_cross_connections = getattr(config, 'use_cross_connections', False)
        if self.use_cross_connections:
            self.cross_connections = nn.ModuleList([
                nn.Linear(self.hidden_dim * 2, self.hidden_dim)
                for _ in range(self.num_layers - 1)
            ])

        # Final projection (optional)
        self.final_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.final_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self,
                expert_embeddings: torch.Tensor,
                routing_weights: torch.Tensor,
                return_intermediate: bool = False) -> Union[torch.Tensor, Dict]:
        """
        Hierarchical fusion of expert embeddings

        Args:
            expert_embeddings: [batch_size, num_experts, hidden_dim]
            routing_weights: [batch_size, num_experts]
            return_intermediate: Whether to return intermediate representations

        Returns:
            fused_embedding: [batch_size, hidden_dim]
            or dict with intermediate outputs if requested
        """
        intermediate_outputs = []

        # First layer
        fused = self.fusion_layers[0](expert_embeddings, routing_weights)
        intermediate_outputs.append(fused)

        # Subsequent layers
        for i in range(1, self.num_layers):
            # Create input for next layer
            # Option 1: Use fused output as query, original experts as context
            # Option 2: Update expert embeddings based on previous fusion

            # Here we update the "query" while keeping experts fixed
            # This allows the model to progressively refine its understanding

            if self.use_cross_connections and i > 0:
                # Add cross-layer connection
                prev_fused = intermediate_outputs[-1]
                cross_input = torch.cat([fused, prev_fused], dim=-1)
                cross_output = self.cross_connections[i - 1](cross_input)
                fused = fused + cross_output

            # Apply next fusion layer
            fused = self.fusion_layers[i](expert_embeddings, routing_weights)
            intermediate_outputs.append(fused)

        # Final projection
        output = self.final_projection(fused)
        output = self.final_norm(output)

        if return_intermediate:
            return {
                'final': output,
                'intermediate': intermediate_outputs,
                'routing_weights': routing_weights
            }

        return output