"""
Expert encoder implementations
Each expert specializes in a specific domain
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class ExpertEncoder(nn.Module):
    """
    Wrapper for different types of expert models
    Handles both sentence-transformers and regular transformers
    """

    def __init__(self, expert_config: Dict):
        super().__init__()
        self.config = expert_config
        self.model_name = expert_config['model']
        self.model_type = expert_config['type']
        self.output_dim = expert_config['dim']

        # Load model based on type
        if self.model_type == 'sentence-transformer':
            self.model = SentenceTransformer(self.model_name)
            self.model.eval()
            self.tokenizer = None
        else:
            # For transformer models, load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Projection layer to standardize dimensions
        if self.output_dim != 768:
            self.projection = nn.Linear(self.output_dim, 768)
        else:
            self.projection = None

        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, texts: List[str], return_attention: bool = False) -> torch.Tensor:
        """
        Encode texts to embeddings
        Args:
            texts: List of input texts
            return_attention: Whether to return attention weights (for analysis)
        Returns:
            embeddings: Tensor of shape [batch_size, hidden_dim]
        """
        if self.model_type == 'sentence-transformer':
            # Use sentence-transformers encoding
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=False,  # We'll normalize later if needed
                    batch_size=len(texts)
                )
        else:
            # Use transformer model with mean pooling
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )

            # Move to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

                # Mean pooling over tokens
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']

                # Expand attention mask for broadcasting
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                    last_hidden_state.size()
                ).float()

                # Sum embeddings and mask
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                # Calculate mean
                embeddings = sum_embeddings / sum_mask

        # Apply projection if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings)

        return embeddings

    def get_model_info(self) -> Dict:
        """Get information about the expert model"""
        return {
            'name': self.model_name,
            'type': self.model_type,
            'output_dim': self.output_dim,
            'description': self.config.get('description', 'No description'),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class ExpertPool(nn.Module):
    """
    Pool of expert encoders
    Manages multiple experts and provides unified interface
    """

    def __init__(self, expert_configs: Dict[str, Dict]):
        super().__init__()
        self.expert_configs = expert_configs
        self.expert_names = list(expert_configs.keys())
        self.num_experts = len(self.expert_names)

        # Initialize expert encoders
        logger.info(f"Initializing {self.num_experts} expert encoders...")
        self.experts = nn.ModuleDict()

        for name, config in expert_configs.items():
            logger.info(f"  Loading {name}: {config['model']}")
            try:
                self.experts[name] = ExpertEncoder(config)
                logger.info(f"    ✓ Successfully loaded {name}")
            except Exception as e:
                logger.error(f"    ✗ Failed to load {name}: {e}")
                raise

        # Cache for embeddings (optional optimization)
        self.use_cache = False
        self.cache = {}

    def forward(self, texts: List[str], expert_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get embeddings from all or selected experts
        Args:
            texts: List of input texts
            expert_indices: Optional tensor indicating which experts to use
        Returns:
            expert_embeddings: Tensor of shape [batch_size, num_experts, hidden_dim]
        """
        batch_size = len(texts)

        # Check cache
        cache_key = str(hash(tuple(texts)))
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        # Collect embeddings from each expert
        expert_embeddings = []

        for i, name in enumerate(self.expert_names):
            # Skip if expert not selected
            if expert_indices is not None and i not in expert_indices:
                # Add zeros for non-selected experts
                device = next(self.experts[name].parameters()).device
                zeros = torch.zeros(batch_size, 768).to(device)
                expert_embeddings.append(zeros)
            else:
                # Get embeddings from expert
                embeddings = self.experts[name](texts)
                expert_embeddings.append(embeddings)

        # Stack embeddings: [batch_size, num_experts, hidden_dim]
        expert_embeddings = torch.stack(expert_embeddings, dim=1)

        # Cache if enabled
        if self.use_cache:
            self.cache[cache_key] = expert_embeddings

        return expert_embeddings

    def get_expert_by_name(self, name: str) -> ExpertEncoder:
        """Get a specific expert by name"""
        if name not in self.experts:
            raise ValueError(f"Expert '{name}' not found. Available: {list(self.experts.keys())}")
        return self.experts[name]

    def get_expert_info(self) -> Dict:
        """Get information about all experts"""
        info = {}
        for name, expert in self.experts.items():
            info[name] = expert.get_model_info()
        return info

    def enable_cache(self, max_size: int = 1000):
        """Enable embedding cache for efficiency"""
        self.use_cache = True
        self.cache = {}
        self.max_cache_size = max_size
        logger.info(f"Embedding cache enabled (max size: {max_size})")

    def clear_cache(self):
        """Clear embedding cache"""
        self.cache = {}
        logger.info("Embedding cache cleared")

    def to(self, device):
        """Move all experts to device"""
        super().to(device)
        for expert in self.experts.values():
            expert.to(device)
        return self