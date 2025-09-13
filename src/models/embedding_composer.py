"""
Enhanced embedding composition strategies with improvements
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Literal, Union, Tuple
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from config.settings import DOMAINS, EMBEDDING_DIM
from src.models.domain_classifier import DomainClassifier
from src.models.domain_embedders import DomainEmbedderManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

CompositionMethod = Literal['weighted_sum', 'attention', 'max_pooling', 'learned_gate', 'attention_based']


class EmbeddingAligner:
    """
    Improved alignment using learned projections
    """
    def __init__(self, embedding_dim=768):
        self.projection_matrices = {}
        self.embedding_dim = embedding_dim
        self.fitted = False

    def fit_projections(self, source_embeddings_dict, target_domain='news'):
        """
        Learn projections from each domain to target domain
        """
        if target_domain not in source_embeddings_dict:
            logger.warning(f"Target domain {target_domain} not found")
            return

        target_embs = source_embeddings_dict[target_domain]

        for domain, source_embs in source_embeddings_dict.items():
            if domain != target_domain:
                # Learn linear transformation
                model = Ridge(alpha=1.0)
                model.fit(source_embs.reshape(-1, self.embedding_dim),
                         target_embs.reshape(-1, self.embedding_dim))
                self.projection_matrices[domain] = model

        self.fitted = True

    def align_embeddings(self, embeddings_dict, reference_domain='news'):
        """
        Project all embeddings to reference domain space
        """
        aligned = {}

        for domain, emb in embeddings_dict.items():
            if domain == reference_domain:
                aligned[domain] = emb
            elif domain in self.projection_matrices and self.fitted:
                # Apply learned projection
                aligned[domain] = self.projection_matrices[domain].predict(
                    emb.reshape(1, -1))[0]
            else:
                # Fallback to simple normalization
                norm_emb = emb / (np.linalg.norm(emb) + 1e-9)
                ref_norm = np.linalg.norm(embeddings_dict[reference_domain])
                aligned[domain] = norm_emb * ref_norm

        return aligned


class EmbeddingComposer:
    """
    Enhanced composer with task-specific strategies
    """

    def __init__(self):
        """Initialize the composer with classifier and embedders"""
        logger.info("Initializing EmbeddingComposer...")

        self.classifier = DomainClassifier()
        self.embedder_manager = DomainEmbedderManager(load_all=True)
        self.domains = DOMAINS

        # Enhanced components
        self.aligner = EmbeddingAligner()
        self.pca = None
        self.optimal_weights = None
        self.attention_weights = None

        logger.info("EmbeddingComposer initialized successfully")

    def compose_for_task(self, text: str, task: str = 'classification') -> np.ndarray:
        """
        Task-specific composition strategy

        Args:
            text: Input text
            task: 'classification' or 'similarity'

        Returns:
            Composed embedding optimized for task
        """
        if task == 'similarity':
            # For similarity, use single best domain
            domain_probs = self.classifier.classify(text)
            best_domain = self.domains[np.argmax(domain_probs)]
            return self.embedder_manager.get_embedding(text, best_domain)
        elif task == 'classification':
            # For classification, use top-4 composition (best in experiments)
            return self.compose_topk(text, k=4, method='weighted_sum')
        else:
            # Default to standard composition
            return self.compose(text, method='weighted_sum')

    def compose_topk(self,
                     text: str,
                     k: int = 2,
                     method: str = 'weighted_sum') -> np.ndarray:
        """
        Compose using only top-k domains

        Args:
            text: Input text
            k: Number of top domains to use
            method: Composition method

        Returns:
            Composed embedding from top-k domains
        """
        # Get domain probabilities
        domain_probs = self.classifier.classify(text)

        # Get top-k domains
        top_indices = np.argsort(domain_probs)[-k:]

        # Zero out non-top domains
        masked_probs = np.zeros_like(domain_probs)
        masked_probs[top_indices] = domain_probs[top_indices]
        masked_probs = masked_probs / (masked_probs.sum() + 1e-10)

        # Get embeddings
        domain_embeddings = self.embedder_manager.get_all_embeddings(text)

        # Apply smart filtering for problematic domains
        domain_embeddings = self._filter_problematic_domains(
            text, domain_embeddings, masked_probs)

        # Compose with masked probabilities
        if method == 'weighted_sum':
            return self._weighted_sum(domain_embeddings, masked_probs)
        elif method == 'attention':
            return self._attention_based(domain_embeddings, masked_probs)
        elif method == 'attention_based':
            return self.attention_compose(text, domain_embeddings, masked_probs)
        else:
            return self._weighted_sum(domain_embeddings, masked_probs)

    def attention_compose(self, text: str,
                         domain_embeddings: Dict[str, np.ndarray] = None,
                         domain_probs: np.ndarray = None) -> np.ndarray:
        """
        Use self-attention mechanism for composition
        """
        if domain_embeddings is None:
            domain_embeddings = self.embedder_manager.get_all_embeddings(text)
        if domain_probs is None:
            domain_probs = self.classifier.classify(text)

        # Convert to tensor
        embeddings = [domain_embeddings[d] for d in self.domains]
        emb_matrix = torch.FloatTensor(embeddings)  # [n_domains, 768]

        # Compute attention scores using scaled dot-product
        d_k = emb_matrix.shape[1]
        scores = torch.matmul(emb_matrix, emb_matrix.T) / np.sqrt(d_k)

        # Apply domain probabilities as initial weights
        prob_tensor = torch.FloatTensor(domain_probs).unsqueeze(1)
        scores = scores * prob_tensor

        # Softmax to get attention weights
        attention_weights = F.softmax(scores.sum(dim=1), dim=0)

        # Apply attention
        final = torch.matmul(attention_weights, emb_matrix)

        return final.numpy()

    def compose_aligned(self, text: str) -> np.ndarray:
        """
        Compose with improved alignment
        """
        domain_probs = self.classifier.classify(text)
        domain_embeddings = self.embedder_manager.get_all_embeddings(text)

        # Apply filtering
        domain_embeddings = self._filter_problematic_domains(
            text, domain_embeddings, domain_probs)

        # Align embeddings
        aligned_embeddings = self.aligner.align_embeddings(domain_embeddings)

        return self._weighted_sum(aligned_embeddings, domain_probs)

    def ensemble_compose(self, text: str) -> np.ndarray:
        """
        Use dimensionality reduction on concatenated embeddings
        """
        # Get all embeddings
        all_embeddings = []
        for domain in self.domains:
            try:
                emb = self.embedder_manager.get_embedding(text, domain)
                all_embeddings.append(emb)
            except:
                # Use zeros if domain fails
                all_embeddings.append(np.zeros(EMBEDDING_DIM))

        # Concatenate
        concatenated = np.concatenate(all_embeddings)

        # Initialize PCA if needed
        if self.pca is None:
            self.pca = PCA(n_components=EMBEDDING_DIM)
            # Fit on a batch of random data for initialization
            random_data = np.random.randn(100, len(all_embeddings) * EMBEDDING_DIM)
            self.pca.fit(random_data)

        # Reduce dimensions
        reduced = self.pca.transform(concatenated.reshape(1, -1))[0]

        return reduced

    def _filter_problematic_domains(self, text: str,
                                   domain_embeddings: Dict[str, np.ndarray],
                                   domain_probs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Filter out problematic domains based on text characteristics
        """
        # Social domain filtering (BERTweet issues)
        social_indicators = ['@', '#', 'RT', 'http', '😀', '😂', 'LOL', 'lol', 'OMG', 'omg']
        has_social_markers = any(indicator in text for indicator in social_indicators)

        if 'social' in self.domains:
            social_idx = self.domains.index('social')

            # If low social probability AND no social markers, reduce weight
            if domain_probs[social_idx] < 0.3 and not has_social_markers:
                domain_probs[social_idx] = 0.01
                # Renormalize
                domain_probs = domain_probs / (domain_probs.sum() + 1e-10)

        return domain_embeddings

    def compose(self,
                text: str,
                method: CompositionMethod = 'weighted_sum',
                return_details: bool = False) -> Union[np.ndarray, tuple]:
        """
        Create composite embedding for text with improvements
        """
        # Get domain probabilities
        domain_probs = self.classifier.classify(text)

        # Get embeddings from each domain with error handling
        try:
            domain_embeddings = self.embedder_manager.get_all_embeddings(text)
        except Exception as e:
            logger.warning(f"Error getting embeddings: {str(e)[:100]}")
            domain_embeddings = self._get_embeddings_with_fallback(text)

        # Filter problematic domains
        domain_embeddings = self._filter_problematic_domains(
            text, domain_embeddings, domain_probs)

        # Validate embeddings
        domain_embeddings, domain_probs = self._validate_embeddings(
            domain_embeddings, domain_probs)

        # Compose embeddings
        if method == 'weighted_sum':
            final_embedding = self._weighted_sum(domain_embeddings, domain_probs)
        elif method == 'attention':
            final_embedding = self._attention_based(domain_embeddings, domain_probs)
        elif method == 'max_pooling':
            final_embedding = self._max_pooling(domain_embeddings, domain_probs)
        elif method == 'learned_gate':
            final_embedding = self._learned_gate(domain_embeddings, domain_probs)
        elif method == 'attention_based':
            final_embedding = self.attention_compose(text, domain_embeddings, domain_probs)
        else:
            raise ValueError(f"Unknown composition method: {method}")

        # Final validation
        final_embedding = self._validate_final_embedding(final_embedding, domain_embeddings)

        if return_details:
            return final_embedding, domain_probs, domain_embeddings

        return final_embedding

    def _get_embeddings_with_fallback(self, text: str) -> Dict[str, np.ndarray]:
        """
        Get embeddings with fallback for failed domains
        """
        domain_embeddings = {}

        for domain in self.domains:
            try:
                domain_embeddings[domain] = self.embedder_manager.get_embedding(text, domain)
            except Exception as e:
                logger.warning(f"Failed to get {domain} embedding: {e}")
                domain_embeddings[domain] = np.zeros(EMBEDDING_DIM)

        return domain_embeddings

    def _validate_embeddings(self, domain_embeddings: Dict[str, np.ndarray],
                            domain_probs: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Validate and clean embeddings
        """
        for i, domain in enumerate(self.domains):
            if domain in domain_embeddings:
                emb = domain_embeddings[domain]
                # Check for NaN or Inf
                if np.isnan(emb).any() or np.isinf(emb).any():
                    logger.warning(f"Invalid embedding for {domain}, using zeros")
                    domain_embeddings[domain] = np.zeros(EMBEDDING_DIM)
                    domain_probs[i] = 0

        # Renormalize
        if domain_probs.sum() > 0:
            domain_probs = domain_probs / domain_probs.sum()
        else:
            domain_probs = np.ones(len(self.domains)) / len(self.domains)

        return domain_embeddings, domain_probs

    def _validate_final_embedding(self, final_embedding: np.ndarray,
                                 domain_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Validate final embedding and provide fallback
        """
        if np.isnan(final_embedding).any() or np.isinf(final_embedding).any():
            logger.warning("Final embedding has NaN/Inf values, using fallback")

            valid_embeddings = [emb for emb in domain_embeddings.values()
                              if not np.isnan(emb).any() and not np.isinf(emb).any()]

            if valid_embeddings:
                final_embedding = np.mean(valid_embeddings, axis=0)
            else:
                final_embedding = np.random.randn(EMBEDDING_DIM) * 0.1

        return final_embedding

    def compose_batch(self,
                      texts: List[str],
                      method: CompositionMethod = 'weighted_sum',
                      batch_size: int = 32,
                      task: str = 'classification') -> np.ndarray:
        """
        Compose embeddings for multiple texts with task-specific strategy
        """
        logger.info(f"Composing embeddings for {len(texts)} texts using {method} for {task}")

        all_embeddings = []

        for text_idx, text in enumerate(texts):
            if task == 'classification':
                # Use top-4 for classification
                embedding = self.compose_topk(text, k=4, method=method)
            elif task == 'similarity':
                # Use best single domain for similarity
                embedding = self.compose_for_task(text, task='similarity')
            else:
                embedding = self.compose(text, method=method)

            all_embeddings.append(embedding)

            if (text_idx + 1) % 100 == 0:
                logger.info(f"Processed {text_idx + 1}/{len(texts)} texts")

        return np.array(all_embeddings)

    def _weighted_sum(self,
                      embeddings: Dict[str, np.ndarray],
                      probs: np.ndarray) -> np.ndarray:
        """
        Simple weighted average of embeddings
        """
        final_embedding = np.zeros(EMBEDDING_DIM)

        for i, domain in enumerate(self.domains):
            if domain in embeddings:
                final_embedding += probs[i] * embeddings[domain]

        return final_embedding

    def _attention_based(self,
                         embeddings: Dict[str, np.ndarray],
                         probs: np.ndarray) -> np.ndarray:
        """
        Attention-based composition
        """
        # Stack embeddings
        emb_matrix = np.stack([embeddings[d] for d in self.domains if d in embeddings])

        # Convert to torch for attention computation
        emb_tensor = torch.FloatTensor(emb_matrix)
        prob_tensor = torch.FloatTensor(probs[:len(emb_matrix)])

        # Compute attention scores
        attention_scores = prob_tensor.unsqueeze(0)

        # Apply attention
        weighted_emb = torch.matmul(attention_scores, emb_tensor)

        return weighted_emb.squeeze().numpy()

    def _max_pooling(self,
                     embeddings: Dict[str, np.ndarray],
                     probs: np.ndarray) -> np.ndarray:
        """
        Max pooling over domain embeddings
        """
        # Only consider domains with probability > threshold
        threshold = 0.1
        active_domains = [d for i, d in enumerate(self.domains)
                         if i < len(probs) and probs[i] > threshold and d in embeddings]

        if not active_domains:
            active_domains = [d for d in self.domains if d in embeddings]

        if not active_domains:
            return np.zeros(EMBEDDING_DIM)

        # Stack active embeddings
        active_embeddings = np.stack([embeddings[d] for d in active_domains])

        # Take element-wise maximum
        return np.max(active_embeddings, axis=0)

    def _learned_gate(self,
                      embeddings: Dict[str, np.ndarray],
                      probs: np.ndarray) -> np.ndarray:
        """
        Gated composition
        """
        # Use probabilities as gates with a threshold
        gates = np.where(probs > 0.2, probs, 0)
        gates = gates / (gates.sum() + 1e-10)

        final_embedding = np.zeros(EMBEDDING_DIM)

        for i, domain in enumerate(self.domains):
            if domain in embeddings and i < len(gates):
                final_embedding += gates[i] * embeddings[domain]

        return final_embedding


class MoERouter(nn.Module):
    """
    Simple router for MoE that works with your existing setup
    """

    def __init__(self, input_dim=768, num_experts=5):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )

    def forward(self, x):
        return F.softmax(self.router(x), dim=-1)


class MoEEmbeddingComposer(EmbeddingComposer):
    """
    MoE version of your EmbeddingComposer - inherits everything and adds MoE logic
    """

    def __init__(self):
        super().__init__()
        # Initialize router for your 5 domains
        self.router = MoERouter(EMBEDDING_DIM, len(self.domains))
        self.router.eval()  # Keep in eval mode initially

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(EMBEDDING_DIM + len(self.domains), EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        )
        self.fusion.eval()

        logger.info("Initialized MoE composer with routing")

    def compose_moe(self, text: str, return_details: bool = False):
        """
        MoE composition using routing weights as features
        """
        # Get embeddings from all domains (reuse your existing method)
        domain_embeddings = self.embedder_manager.get_all_embeddings(text)

        # Stack embeddings for routing
        emb_list = [domain_embeddings[d] for d in self.domains]
        emb_tensor = torch.FloatTensor(emb_list)  # [5, 768]

        # Get initial embedding (use news as it performed best)
        initial_emb = torch.FloatTensor(domain_embeddings['news']).unsqueeze(0)

        # Get routing weights
        with torch.no_grad():
            routing_weights = self.router(initial_emb).squeeze(0)  # [5]

        # Apply soft routing (all experts contribute)
        weighted_embeddings = []
        for i, domain in enumerate(self.domains):
            weight = routing_weights[i].item()
            weighted_emb = weight * domain_embeddings[domain]
            weighted_embeddings.append(weighted_emb)

        # Sum weighted embeddings
        combined = np.sum(weighted_embeddings, axis=0)

        # Use routing weights as additional features (key insight)
        routing_features = routing_weights.numpy()

        # Concatenate embeddings with routing weights
        enhanced = np.concatenate([combined, routing_features])

        # Pass through fusion layer
        with torch.no_grad():
            enhanced_tensor = torch.FloatTensor(enhanced).unsqueeze(0)
            final_embedding = self.fusion(enhanced_tensor).squeeze(0).numpy()

        if return_details:
            return final_embedding, routing_features, domain_embeddings

        return final_embedding

    def compose_batch_moe(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Batch processing with MoE
        """
        logger.info(f"MoE composing embeddings for {len(texts)} texts")

        all_embeddings = []

        for text_idx, text in enumerate(texts):
            embedding = self.compose_moe(text)
            all_embeddings.append(embedding)

            if (text_idx + 1) % 100 == 0:
                logger.info(f"Processed {text_idx + 1}/{len(texts)} texts")

        return np.array(all_embeddings)