"""
Main Dynamic Embedding Model
Combines experts, router, and fusion components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import logging

from .experts import ExpertPool
from .router import LearnedMoERouter
from .fusion import HierarchicalAdapterFusion

logger = logging.getLogger(__name__)


class DynamicEmbeddingModel(nn.Module):
    """
    Complete Dynamic Embedding Model
    Combines MoE routing with AdapterFusion for adaptive embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize components
        logger.info("Initializing Dynamic Embedding Model...")

        # Expert pool
        self.experts = ExpertPool(config.expert.experts)

        # MoE Router
        self.router = LearnedMoERouter(config.router)

        # Hierarchical AdapterFusion
        self.fusion = HierarchicalAdapterFusion(config.fusion)

        # Optional: Projection layers for different output dimensions
        self.projection_dim = config.expert.projection_dim
        if self.projection_dim != config.expert.hidden_dim:
            self.output_projection = nn.Sequential(
                nn.Linear(config.expert.hidden_dim, self.projection_dim),
                nn.LayerNorm(self.projection_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.projection_dim, config.expert.hidden_dim)
            )
        else:
            self.output_projection = nn.Identity()

        # Contrastive learning head (for training)
        self.contrastive_head = nn.Sequential(
            nn.Linear(config.expert.hidden_dim, config.expert.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.expert.hidden_dim, config.expert.hidden_dim)
        )

        # Cache for efficiency
        self.use_cache = False
        self.cache = {}

        # Move to device
        self.device = torch.device(config.training.device)
        self.to(self.device)

        logger.info(f"Model initialized with {self.experts.num_experts} experts")
        logger.info(f"Total parameters: {self.count_parameters():,}")
        logger.info(f"Trainable parameters: {self.count_parameters(trainable=True):,}")

    def forward(self,
                texts: Union[str, List[str]],
                return_details: bool = False,
                use_cache: bool = None) -> Union[torch.Tensor, Dict]:
        """
        Forward pass through the model

        Args:
            texts: Input text(s)
            return_details: Whether to return detailed outputs
            use_cache: Whether to use caching (overrides self.use_cache)

        Returns:
            embeddings: Tensor of shape [batch_size, hidden_dim]
            or dict with detailed outputs if requested
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        # Check cache
        use_cache = use_cache if use_cache is not None else self.use_cache
        cache_key = str(hash(tuple(texts))) if use_cache else None

        if use_cache and cache_key in self.cache:
            logger.debug(f"Cache hit for {len(texts)} texts")
            return self.cache[cache_key]

        # Get routing weights and auxiliary outputs
        if self.training:
            routing_weights, auxiliary = self.router(texts, return_auxiliary=True)
        else:
            routing_weights = self.router(texts)
            auxiliary = None

        # Get embeddings from all experts
        expert_embeddings = self.experts(texts)

        # Apply hierarchical fusion
        if return_details:
            fusion_output = self.fusion(
                expert_embeddings,
                routing_weights,
                return_intermediate=True
            )
            fused_embedding = fusion_output['final']
        else:
            fused_embedding = self.fusion(expert_embeddings, routing_weights)

        # Apply output projection
        output_embedding = self.output_projection(fused_embedding)

        # Cache result
        if use_cache and cache_key:
            self.cache[cache_key] = output_embedding
            # Limit cache size
            if len(self.cache) > 10000:
                self.cache = dict(list(self.cache.items())[-5000:])

        if return_details:
            return {
                'embedding': output_embedding,
                'routing_weights': routing_weights,
                'expert_embeddings': expert_embeddings,
                'fusion_intermediate': fusion_output.get('intermediate', []),
                'auxiliary': auxiliary
            }

        return output_embedding

    def encode(self,
               texts: Union[str, List[str]],
               batch_size: int = 32,
               show_progress: bool = False,
               normalize: bool = False) -> np.ndarray:
        """
        Encode texts to embeddings (compatible with sentence-transformers API)

        Args:
            texts: Input text(s)
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings

        Returns:
            embeddings: Numpy array of shape [n_texts, hidden_dim]
        """
        if isinstance(texts, str):
            texts = [texts]

        self.eval()
        all_embeddings = []

        # Process in batches
        from tqdm import tqdm
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", total=len(texts) // batch_size)

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]
                embeddings = self.forward(batch_texts)

                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def get_expert_embeddings(self, texts: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Get embeddings from each expert separately (for analysis)

        Args:
            texts: Input text(s)

        Returns:
            dict mapping expert names to embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        self.eval()
        expert_embeddings = {}

        with torch.no_grad():
            for name, expert in self.experts.experts.items():
                embeddings = expert(texts)
                expert_embeddings[name] = embeddings.cpu().numpy()

        return expert_embeddings

    def analyze_routing(self, texts: Union[str, List[str]]) -> Dict:
        """
        Analyze routing patterns for given texts

        Args:
            texts: Input text(s)

        Returns:
            dict with routing analysis
        """
        if isinstance(texts, str):
            texts = [texts]

        return self.router.get_routing_analysis(texts)

    def count_parameters(self, trainable: bool = False) -> int:
        """Count model parameters"""
        if trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save(self, path: Union[str, Path]):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'expert_info': self.experts.get_expert_info(),
            'router_stats': {
                'expert_usage': self.router.expert_usage.cpu().numpy(),
                'routing_entropy': self.router.routing_entropy.item()
            }
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path], strict: bool = True):
        """Load model checkpoint"""
        path = Path(path)

        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        # Restore router statistics
        if 'router_stats' in checkpoint:
            self.router.expert_usage = torch.tensor(
                checkpoint['router_stats']['expert_usage'],
                device=self.device
            )
            self.router.routing_entropy = torch.tensor(
                checkpoint['router_stats']['routing_entropy'],
                device=self.device
            )

        logger.info(f"Model loaded from {path}")

    def enable_cache(self, max_size: int = 10000):
        """Enable embedding cache"""
        self.use_cache = True
        self.cache = {}
        self.max_cache_size = max_size
        logger.info(f"Embedding cache enabled (max size: {max_size})")

    def clear_cache(self):
        """Clear embedding cache"""
        self.cache = {}
        if hasattr(self.experts, 'clear_cache'):
            self.experts.clear_cache()
        logger.info("All caches cleared")

    def freeze_experts(self):
        """Freeze expert encoders (useful for training only router/fusion)"""
        for param in self.experts.parameters():
            param.requires_grad = False
        logger.info("Expert encoders frozen")

    def unfreeze_experts(self):
        """Unfreeze expert encoders"""
        for param in self.experts.parameters():
            param.requires_grad = True
        logger.info("Expert encoders unfrozen")

    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        return {
            'num_experts': self.experts.num_experts,
            'expert_names': self.experts.expert_names,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': self.count_parameters(trainable=True),
            'router_config': {
                'top_k': self.router.top_k,
                'temperature': self.router.temperature.item(),
                'hidden_dim': self.router.hidden_dim
            },
            'fusion_config': {
                'num_layers': self.fusion.num_layers,
                'hidden_dim': self.fusion.hidden_dim
            },
            'device': str(self.device)
        }