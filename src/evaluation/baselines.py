"""
Baseline models for comparison
"""
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from config.model_configs import BASELINE_MODELS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineModels:
    """
    Manages baseline models for comparison
    """

    def __init__(self, load_all: bool = False):
        """
        Initialize baseline models

        Args:
            load_all: If True, load all models at initialization
        """
        self.model_configs = BASELINE_MODELS
        self.models = {}

        if load_all:
            self._load_all_models()

    def _load_all_models(self):
        """Load all baseline models"""
        logger.info("Loading baseline models...")

        for name, model_path in self.model_configs.items():
            logger.info(f"Loading {name}: {model_path}")
            self.models[name] = SentenceTransformer(model_path)

    def load_model(self, model_name: str):
        """Load a specific baseline model"""
        if model_name in self.models:
            return self.models[model_name]

        model_path = self.model_configs[model_name]
        logger.info(f"Loading baseline model {model_name}: {model_path}")

        self.models[model_name] = SentenceTransformer(model_path)
        return self.models[model_name]

    def get_embedding(self, text: str, model_name: str) -> np.ndarray:
        """
        Get embedding from a baseline model

        Args:
            text: Input text
            model_name: Name of baseline model

        Returns:
            Embedding vector
        """
        if model_name not in self.models:
            self.load_model(model_name)

        return self.models[model_name].encode(text, convert_to_numpy=True)

    def get_batch_embeddings(self, texts: List[str],
                             model_name: str) -> np.ndarray:
        """
        Get embeddings for multiple texts

        Args:
            texts: List of texts
            model_name: Name of baseline model

        Returns:
            Array of embeddings
        """
        if model_name not in self.models:
            self.load_model(model_name)

        return self.models[model_name].encode(texts, convert_to_numpy=True)


# Add to src/evaluation/baselines.py

# In src/evaluation/baselines.py, fix the get_multi_embedding method:

class MultiEmbeddingBaseline:
    """
    Concatenate multiple embeddings as baseline
    """

    def __init__(self):
        # Use 2-3 strong general models
        self.models = {
            'mpnet': SentenceTransformer('all-mpnet-base-v2'),  # 768 dim
            'roberta': SentenceTransformer('roberta-base-nli-mean-tokens'),  # 768 dim
            'distilbert': SentenceTransformer('distilbert-base-nli-mean-tokens')  # 768 dim
        }

    def get_multi_embedding(self, text: str, method='concat'):
        """
        Get multi-model embedding
        """
        embeddings = []
        for name, model in self.models.items():
            emb = model.encode(text, convert_to_numpy=True)
            # Ensure it's 1D array
            if emb.ndim > 1:
                emb = emb.squeeze()
            embeddings.append(emb)

        if method == 'concat':
            # Concatenate all embeddings
            return np.concatenate(embeddings)
        elif method == 'average':
            # Stack then average (fix for inhomogeneous shape error)
            embeddings_array = np.stack(embeddings)
            return np.mean(embeddings_array, axis=0)
        elif method == 'max':
            # Max pool across embeddings
            embeddings_array = np.stack(embeddings)
            return np.max(embeddings_array, axis=0)