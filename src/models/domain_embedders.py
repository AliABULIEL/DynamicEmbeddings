"""
Simplified domain embedders - only load what's needed
"""
import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DomainEmbedderManager:
    """
    Simplified - only load models on demand
    """

    def __init__(self, load_all: bool = False):
        self.models = {}
        # Only load MPNet by default (it's the news model and best performer)
        if not load_all:
            self.models['news'] = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            logger.info("Loaded MPNet as news model")

    def get_embedding(self, text: str, domain: str = 'news') -> np.ndarray:
        """Get embedding - default to news/MPNet"""
        if domain not in self.models:
            # Just use MPNet for everything
            domain = 'news'

        return self.models[domain].encode(text, convert_to_numpy=True)

    def get_batch_embeddings(self, texts: List[str], domain: str = 'news') -> np.ndarray:
        """Batch embeddings"""
        if domain not in self.models:
            domain = 'news'

        return self.models[domain].encode(texts, convert_to_numpy=True)

    def get_all_embeddings(self, text: str) -> Dict[str, np.ndarray]:
        """Only return news embedding"""
        return {'news': self.get_embedding(text, 'news')}

# Remove TransformerEmbedder class - not needed