# In src/evaluation/baselines.py, fix the get_multi_embedding method:
import numpy as np
from sentence_transformers import SentenceTransformer


class MultiEmbeddingBaseline:
    """
    Concatenate multiple embeddings as baseline
    """

    def __init__(self):
        # Use 2-3 strong general models
        self.models = {
            'mpnet': SentenceTransformer('all-mpnet-base-v2'),
            'minilm': SentenceTransformer('all-MiniLM-L6-v2'),
            'roberta': SentenceTransformer('roberta-base-nli-mean-tokens')
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