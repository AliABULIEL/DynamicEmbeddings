"""
Enhanced baseline models for comparison
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
        """
        if model_name not in self.models:
            self.load_model(model_name)

        return self.models[model_name].encode(text, convert_to_numpy=True)

    def get_batch_embeddings(self, texts: List[str],
                             model_name: str) -> np.ndarray:
        """
        Get embeddings for multiple texts
        """
        if model_name not in self.models:
            self.load_model(model_name)

        return self.models[model_name].encode(texts, convert_to_numpy=True)


class MultiEmbeddingBaseline:
    """
    Enhanced multi-embedding baseline with proper handling
    """

    def __init__(self):
        # Use models with same 768 dimensions for consistency
        self.models = {
            'mpnet': SentenceTransformer('all-mpnet-base-v2'),           # 768 dim
            'distilbert': SentenceTransformer('distilbert-base-nli-mean-tokens'),  # 768 dim
            'roberta': SentenceTransformer('roberta-base-nli-mean-tokens')  # 768 dim
        }

        logger.info("Loaded multi-embedding baseline models")

    def get_multi_embedding(self, text: str, method='concat'):
        """
        Get multi-model embedding with improved handling
        """
        embeddings = []

        for name, model in self.models.items():
            try:
                emb = model.encode(text, convert_to_numpy=True)

                # Ensure it's 1D array
                if emb.ndim > 1:
                    emb = emb.squeeze()

                # Ensure consistent dimension (768)
                if emb.shape[0] != 768:
                    logger.warning(f"Model {name} produced {emb.shape[0]} dims, padding/truncating to 768")
                    if emb.shape[0] < 768:
                        emb = np.pad(emb, (0, 768 - emb.shape[0]), mode='constant')
                    else:
                        emb = emb[:768]

                embeddings.append(emb)

            except Exception as e:
                logger.error(f"Failed to get embedding from {name}: {e}")
                embeddings.append(np.zeros(768))

        if method == 'concat':
            # Concatenate all embeddings
            return np.concatenate(embeddings)
        elif method == 'average':
            # Stack then average
            embeddings_array = np.stack(embeddings)
            return np.mean(embeddings_array, axis=0)
        elif method == 'max':
            # Max pool across embeddings
            embeddings_array = np.stack(embeddings)
            return np.max(embeddings_array, axis=0)
        elif method == 'weighted':
            # Weighted average (MPNet gets higher weight as it's generally better)
            weights = [0.5, 0.25, 0.25]  # MPNet, DistilBERT, RoBERTa
            weighted_sum = sum(w * emb for w, emb in zip(weights, embeddings))
            return weighted_sum
        else:
            raise ValueError(f"Unknown method: {method}")


class AdvancedBaseline:
    """
    Additional advanced baseline methods
    """

    def __init__(self):
        # Load state-of-the-art models
        self.models = {
            'e5_large': SentenceTransformer('intfloat/e5-large-v2'),
            'gte_large': SentenceTransformer('thenlper/gte-large'),
            'bge_large': SentenceTransformer('BAAI/bge-large-en-v1.5')
        }

        logger.info("Loaded advanced baseline models")

    def get_best_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding from the best performing model (E5-large)
        """
        # E5 requires specific formatting
        formatted_text = f"query: {text}" if len(text) < 100 else f"passage: {text}"
        return self.models['e5_large'].encode(formatted_text, convert_to_numpy=True)

    def get_ensemble_embedding(self, text: str) -> np.ndarray:
        """
        Ensemble of top models
        """
        embeddings = []

        # E5
        formatted_text = f"query: {text}" if len(text) < 100 else f"passage: {text}"
        embeddings.append(self.models['e5_large'].encode(formatted_text, convert_to_numpy=True))

        # GTE
        embeddings.append(self.models['gte_large'].encode(text, convert_to_numpy=True))

        # BGE
        embeddings.append(self.models['bge_large'].encode(text, convert_to_numpy=True))

        # Average ensemble
        return np.mean(embeddings, axis=0)