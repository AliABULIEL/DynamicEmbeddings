"""
Zero-shot domain classification using pre-trained models
"""
import numpy as np
from typing import List, Dict, Optional
from transformers import pipeline
import torch
from config.settings import DOMAINS
from config.model_configs import ZERO_SHOT_MODELS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DomainClassifier:
    """
    Zero-shot domain classifier using pre-trained NLI models
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the domain classifier

        Args:
            model_name: HuggingFace model name for zero-shot classification
        """
        self.model_name = model_name or ZERO_SHOT_MODELS['default']
        self.domains = DOMAINS

        # Initialize the classifier pipeline
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading zero-shot classifier: {self.model_name}")

        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device
        )

        logger.info(f"Classifier loaded successfully on device: {device}")

    def classify(self, text: str, return_dict: bool = False) -> np.ndarray:
        """
        Classify text into domain probabilities

        Args:
            text: Input text to classify
            return_dict: If True, return dict with domain names as keys

        Returns:
            Array of probabilities for each domain or dict if return_dict=True
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        # Perform zero-shot classification
        result = self.classifier(
            text,
            candidate_labels=self.domains,
            multi_label=False,  # Ensures probabilities sum to 1
            hypothesis_template="This text is about {}."
        )

        # Convert to probability array in consistent order
        prob_dict = dict(zip(result['labels'], result['scores']))

        if return_dict:
            return {domain: prob_dict[domain] for domain in self.domains}

        # Return as numpy array in consistent domain order
        probs = np.array([prob_dict[domain] for domain in self.domains])

        # Ensure probabilities sum to 1 (they should already)
        probs = probs / probs.sum()

        return probs

    def classify_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Classify multiple texts in batches

        Args:
            texts: List of texts to classify
            batch_size: Batch size for processing

        Returns:
            Array of shape (n_texts, n_domains) with probabilities
        """
        logger.info(f"Classifying {len(texts)} texts in batches of {batch_size}")

        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_probs = [self.classify(text) for text in batch]
            all_probs.extend(batch_probs)

            if (i + batch_size) % 100 == 0:
                logger.info(f"Processed {i + batch_size}/{len(texts)} texts")

        return np.array(all_probs)

    def get_dominant_domain(self, text: str) -> tuple:
        """
        Get the most likely domain for a text

        Args:
            text: Input text

        Returns:
            Tuple of (domain_name, probability)
        """
        probs = self.classify(text)
        max_idx = np.argmax(probs)
        return self.domains[max_idx], probs[max_idx]

    def get_multi_domain_texts(self, texts: List[str],
                               entropy_threshold: float = 1.0) -> List[int]:
        """
        Identify texts that are multi-domain based on entropy

        Args:
            texts: List of texts
            entropy_threshold: Threshold for considering text as multi-domain

        Returns:
            Indices of multi-domain texts
        """
        probs = self.classify_batch(texts)

        # Calculate entropy for each text
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

        # Find texts with high entropy (multi-domain)
        multi_domain_indices = np.where(entropy > entropy_threshold)[0]

        return multi_domain_indices.tolist()