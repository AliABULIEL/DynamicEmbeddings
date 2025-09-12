"""
Embedding composition strategies
"""
import numpy as np
from typing import Dict, List, Optional, Literal
import torch
import torch.nn.functional as F
from config.settings import DOMAINS, EMBEDDING_DIM
from src.models.domain_classifier import DomainClassifier
from src.models.domain_embedders import DomainEmbedderManager
from src.utils.logger import get_logger
from typing import Dict, List, Optional, Union


logger = get_logger(__name__)

CompositionMethod = Literal['weighted_sum', 'attention', 'max_pooling', 'learned_gate']


class EmbeddingComposer:
    """
    Composes embeddings from multiple domain-specific models
    """

    def __init__(self):
        """Initialize the composer with classifier and embedders"""
        logger.info("Initializing EmbeddingComposer...")

        self.classifier = DomainClassifier()
        self.embedder_manager = DomainEmbedderManager(load_all=True)
        self.domains = DOMAINS

        # For attention-based composition (optional)
        self.attention_weights = None

        logger.info("EmbeddingComposer initialized successfully")

    def compose(self,
                text: str,
                method: CompositionMethod = 'weighted_sum',
                return_details: bool = False) -> Union[np.ndarray, tuple]:
        """
        Create composite embedding for text

        Args:
            text: Input text
            method: Composition method to use
            return_details: If True, return (embedding, domain_probs, domain_embeddings)

        Returns:
            Composite embedding or tuple with details
        """
        # Step 1: Get domain probabilities
        domain_probs = self.classifier.classify(text)

        # Step 2: Smart filtering for social domain
        # BERTweet expects tweets - if text doesn't look like social media and has low social prob, zero it out
        social_indicators = ['@', '#', 'RT', 'http', '😀', '😂', 'LOL', 'lol', 'OMG', 'omg']
        has_social_markers = any(indicator in text for indicator in social_indicators)

        # Find social domain index (assuming it's the last one)
        social_idx = self.domains.index('social') if 'social' in self.domains else -1

        if social_idx >= 0:
            # If low social probability AND no social markers, reduce social weight significantly
            if domain_probs[social_idx] < 0.3 and not has_social_markers:
                # Store original for logging if needed
                original_social_prob = domain_probs[social_idx]

                # Reduce social weight to near-zero
                domain_probs[social_idx] = 0.01

                # Renormalize to sum to 1
                domain_probs = domain_probs / domain_probs.sum()

                # Optional logging
                if original_social_prob > 0.1:
                    logger.debug(
                        f"Reduced social weight from {original_social_prob:.3f} to {domain_probs[social_idx]:.3f} for formal text")

        # Step 3: Get embeddings from each domain with error handling
        try:
            domain_embeddings = self.embedder_manager.get_all_embeddings(text)
        except Exception as e:
            logger.warning(f"Error getting embeddings for text: {str(e)[:100]}")

            # Fallback: try getting embeddings individually with error handling
            domain_embeddings = {}
            for domain in self.domains:
                try:
                    domain_embeddings[domain] = self.embedder_manager.get_embedding(text, domain)
                except Exception as domain_error:
                    logger.warning(f"Failed to get {domain} embedding: {domain_error}")
                    # Use zeros as fallback for failed domain
                    domain_embeddings[domain] = np.zeros(EMBEDDING_DIM)
                    # Also zero out this domain's probability
                    domain_idx = self.domains.index(domain)
                    domain_probs[domain_idx] = 0

            # Renormalize probabilities after zeroing failed domains
            if domain_probs.sum() > 0:
                domain_probs = domain_probs / domain_probs.sum()
            else:
                # If all failed, use uniform distribution
                domain_probs = np.ones(len(self.domains)) / len(self.domains)

        # Step 4: Validate embeddings
        for domain in self.domains:
            if domain in domain_embeddings:
                emb = domain_embeddings[domain]
                # Check for NaN or Inf
                if np.isnan(emb).any() or np.isinf(emb).any():
                    logger.warning(f"Invalid embedding for {domain}, using zeros")
                    domain_embeddings[domain] = np.zeros(EMBEDDING_DIM)
                    # Zero out this domain's weight
                    domain_idx = self.domains.index(domain)
                    domain_probs[domain_idx] = 0

        # Renormalize after validation
        if domain_probs.sum() > 0:
            domain_probs = domain_probs / domain_probs.sum()
        else:
            domain_probs = np.ones(len(self.domains)) / len(self.domains)

        # Step 5: Compose embeddings
        if method == 'weighted_sum':
            final_embedding = self._weighted_sum(domain_embeddings, domain_probs)
        elif method == 'attention':
            final_embedding = self._attention_based(domain_embeddings, domain_probs)
        elif method == 'max_pooling':
            final_embedding = self._max_pooling(domain_embeddings, domain_probs)
        elif method == 'learned_gate':
            final_embedding = self._learned_gate(domain_embeddings, domain_probs)
        else:
            raise ValueError(f"Unknown composition method: {method}")

        # Step 6: Final validation
        if np.isnan(final_embedding).any() or np.isinf(final_embedding).any():
            logger.warning("Final embedding has NaN/Inf values, using fallback")
            # Use average of successful embeddings as fallback
            valid_embeddings = [emb for emb in domain_embeddings.values()
                                if not np.isnan(emb).any() and not np.isinf(emb).any()]
            if valid_embeddings:
                final_embedding = np.mean(valid_embeddings, axis=0)
            else:
                # Last resort: random embedding
                final_embedding = np.random.randn(EMBEDDING_DIM) * 0.1

        if return_details:
            return final_embedding, domain_probs, domain_embeddings

        return final_embedding

    def compose_batch(self,
                      texts: List[str],
                      method: CompositionMethod = 'weighted_sum',
                      batch_size: int = 32) -> np.ndarray:
        """
        Compose embeddings for multiple texts

        Args:
            texts: List of texts
            method: Composition method
            batch_size: Batch size for processing

        Returns:
            Array of composite embeddings
        """
        logger.info(f"Composing embeddings for {len(texts)} texts using {method}")

        # Get domain probabilities for all texts
        all_domain_probs = self.classifier.classify_batch(texts, batch_size)

        # Get embeddings for each domain
        all_embeddings = []

        for text_idx, text in enumerate(texts):
            domain_embeddings = self.embedder_manager.get_all_embeddings(text)
            domain_probs = all_domain_probs[text_idx]

            # Compose based on method
            if method == 'weighted_sum':
                embedding = self._weighted_sum(domain_embeddings, domain_probs)
            elif method == 'attention':
                embedding = self._attention_based(domain_embeddings, domain_probs)
            elif method == 'max_pooling':
                embedding = self._max_pooling(domain_embeddings, domain_probs)
            else:
                embedding = self._learned_gate(domain_embeddings, domain_probs)

            all_embeddings.append(embedding)

            if (text_idx + 1) % 100 == 0:
                logger.info(f"Processed {text_idx + 1}/{len(texts)} texts")

        return np.array(all_embeddings)

    def _weighted_sum(self,
                      embeddings: Dict[str, np.ndarray],
                      probs: np.ndarray) -> np.ndarray:
        """
        Simple weighted average of embeddings

        Args:
            embeddings: Dict of domain embeddings
            probs: Domain probabilities

        Returns:
            Weighted embedding
        """
        final_embedding = np.zeros(EMBEDDING_DIM)

        for i, domain in enumerate(self.domains):
            final_embedding += probs[i] * embeddings[domain]

        return final_embedding

    def _attention_based(self,
                         embeddings: Dict[str, np.ndarray],
                         probs: np.ndarray) -> np.ndarray:
        """
        Attention-based composition

        Args:
            embeddings: Dict of domain embeddings
            probs: Domain probabilities

        Returns:
            Attention-weighted embedding
        """
        # Stack embeddings
        emb_matrix = np.stack([embeddings[d] for d in self.domains])  # [n_domains, embedding_dim]

        # Convert to torch for attention computation
        emb_tensor = torch.FloatTensor(emb_matrix)
        prob_tensor = torch.FloatTensor(probs)

        # Compute attention scores (using probabilities as initial weights)
        # You could also learn these weights or use self-attention
        attention_scores = prob_tensor.unsqueeze(0)  # [1, n_domains]

        # Apply attention
        weighted_emb = torch.matmul(attention_scores, emb_tensor)  # [1, embedding_dim]

        return weighted_emb.squeeze().numpy()

    def _max_pooling(self,
                     embeddings: Dict[str, np.ndarray],
                     probs: np.ndarray) -> np.ndarray:
        """
        Max pooling over domain embeddings

        Args:
            embeddings: Dict of domain embeddings
            probs: Domain probabilities (used for filtering)

        Returns:
            Max-pooled embedding
        """
        # Only consider domains with probability > threshold
        threshold = 0.1
        active_domains = [d for i, d in enumerate(self.domains) if probs[i] > threshold]

        if not active_domains:
            # If no domain passes threshold, use all
            active_domains = self.domains

        # Stack active embeddings
        active_embeddings = np.stack([embeddings[d] for d in active_domains])

        # Take element-wise maximum
        return np.max(active_embeddings, axis=0)

    def _learned_gate(self,
                      embeddings: Dict[str, np.ndarray],
                      probs: np.ndarray) -> np.ndarray:
        """
        Gated composition (simplified version without learning)

        Args:
            embeddings: Dict of domain embeddings
            probs: Domain probabilities

        Returns:
            Gated embedding
        """
        # This is a simplified version
        # In a full implementation, you'd have learnable gate parameters

        # Use probabilities as gates with a threshold
        gates = np.where(probs > 0.2, probs, 0)
        gates = gates / (gates.sum() + 1e-10)  # Renormalize

        final_embedding = np.zeros(EMBEDDING_DIM)

        for i, domain in enumerate(self.domains):
            final_embedding += gates[i] * embeddings[domain]

        return final_embedding