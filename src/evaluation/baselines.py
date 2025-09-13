"""
Simplified baselines - only keep what we actually use
"""
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineModels:
    """
    Simplified baselines - only MPNet and E5
    """

    def __init__(self):
        self.models = {}
        # Only load MPNet by default (best performer)
        self.models['mpnet'] = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        logger.info("Loaded MPNet baseline")

    def get_embedding(self, text: str, model_name: str = 'mpnet') -> np.ndarray:
        """Get embedding from baseline model"""
        if model_name not in self.models:
            if model_name == 'e5':
                self.models['e5'] = SentenceTransformer('intfloat/e5-large-v2')
            else:
                # Default to MPNet
                model_name = 'mpnet'

        return self.models[model_name].encode(text, convert_to_numpy=True)

    def get_batch_embeddings(self, texts: List[str], model_name: str = 'mpnet') -> np.ndarray:
        """Get batch embeddings"""
        if model_name not in self.models:
            if model_name == 'e5':
                self.models['e5'] = SentenceTransformer('intfloat/e5-large-v2')
            else:
                model_name = 'mpnet'

        return self.models[model_name].encode(texts, convert_to_numpy=True)

# Remove MultiEmbeddingBaseline and AdvancedBaselines - they don't help