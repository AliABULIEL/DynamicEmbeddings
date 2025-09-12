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

        # Step 2: Get embeddings from each domain
        domain_embeddings = self.embedder_manager.get_all_embeddings(text)

        # Step 3: Compose embeddings
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