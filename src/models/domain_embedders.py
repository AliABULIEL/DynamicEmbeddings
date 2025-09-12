"""
Domain-specific embedding models manager
"""
import numpy as np
from typing import Dict, List, Optional, Union
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from config.model_configs import DOMAIN_MODELS
from config.settings import EMBEDDING_DIM, MAX_SEQUENCE_LENGTH
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TransformerEmbedder:
    """
    Wrapper for transformer models to create sentence embeddings
    """

    def __init__(self, model_name: str):
        """Initialize a transformer model for embeddings"""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                try:
                    # Tokenize with better error handling
                    encoded = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=128,  # Reduce from 512
                        return_tensors='pt',
                        return_attention_mask=True,
                        add_special_tokens=True
                    )

                    # Move to device
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}

                    # Get embeddings
                    outputs = self.model(**encoded)

                    # Mean pooling
                    attention_mask = encoded['attention_mask']
                    embeddings = outputs.last_hidden_state

                    # Proper mean pooling with attention mask
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask

                    embeddings = embeddings.cpu().numpy()
                    all_embeddings.append(embeddings)

                except Exception as e:
                    # If a model fails, use zeros
                    print(f"Warning: {self.model_name} failed on batch: {e}")
                    batch_size_actual = len(batch_texts)
                    fallback = np.zeros((batch_size_actual, 768))
                    all_embeddings.append(fallback)

        return np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
    # def encode(self, texts: Union[str, List[str]],
    #            batch_size: int = 32) -> np.ndarray:
    #     """
    #     Encode texts into embeddings
    #
    #     Args:
    #         texts: Single text or list of texts
    #         batch_size: Batch size for encoding
    #
    #     Returns:
    #         Embeddings array
    #     """
    #     if isinstance(texts, str):
    #         texts = [texts]
    #
    #     all_embeddings = []
    #
    #     with torch.no_grad():
    #         for i in range(0, len(texts), batch_size):
    #             batch_texts = texts[i:i + batch_size]
    #
    #             # Tokenize
    #             encoded = self.tokenizer(
    #                 batch_texts,
    #                 padding=True,
    #                 truncation=True,
    #                 max_length=MAX_SEQUENCE_LENGTH,
    #                 return_tensors='pt'
    #             )
    #
    #             # Move to device
    #             encoded = {k: v.to(self.device) for k, v in encoded.items()}
    #
    #             # Get embeddings
    #             outputs = self.model(**encoded)
    #
    #             # Mean pooling over tokens
    #             embeddings = outputs.last_hidden_state.mean(dim=1)
    #
    #             # Move to CPU and convert to numpy
    #             embeddings = embeddings.cpu().numpy()
    #             all_embeddings.append(embeddings)
    #
    #     return np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]


class DomainEmbedderManager:
    """
    Manages all domain-specific embedding models
    """

    def __init__(self, load_all: bool = True):
        """
        Initialize the embedder manager

        Args:
            load_all: If True, load all models at initialization
        """
        self.domain_configs = DOMAIN_MODELS
        self.models = {}

        if load_all:
            self._load_all_models()

    def _load_all_models(self):
        """Load all domain-specific models"""
        logger.info("Loading all domain-specific models...")

        for domain, config in self.domain_configs.items():
            logger.info(f"Loading {domain} model: {config['model_name']}")

            if config['type'] == 'sentence-transformer':
                self.models[domain] = SentenceTransformer(config['model_name'])
            else:  # transformer type
                self.models[domain] = TransformerEmbedder(config['model_name'])

            logger.info(f"Successfully loaded {domain} model")

    def load_model(self, domain: str):
        """Load a specific domain model"""
        if domain in self.models:
            return self.models[domain]

        config = self.domain_configs[domain]
        logger.info(f"Loading {domain} model: {config['model_name']}")

        if config['type'] == 'sentence-transformer':
            self.models[domain] = SentenceTransformer(config['model_name'])
        else:
            self.models[domain] = TransformerEmbedder(config['model_name'])

        return self.models[domain]

    def get_embedding(self, text: str, domain: str) -> np.ndarray:
        """
        Get embedding for text using specific domain model

        Args:
            text: Input text
            domain: Domain name

        Returns:
            Embedding vector
        """
        if domain not in self.models:
            self.load_model(domain)

        model = self.models[domain]

        # Handle both SentenceTransformer and TransformerEmbedder
        if isinstance(model, SentenceTransformer):
            embedding = model.encode(text, convert_to_numpy=True)
        else:
            embedding = model.encode(text)

        # Ensure correct shape
        if embedding.ndim == 1:
            return embedding
        else:
            return embedding.squeeze()

    def get_all_embeddings(self, text: str) -> Dict[str, np.ndarray]:
        """
        Get embeddings from all domain models

        Args:
            text: Input text

        Returns:
            Dictionary mapping domain names to embeddings
        """
        embeddings = {}

        for domain in self.domain_configs.keys():
            embeddings[domain] = self.get_embedding(text, domain)

        return embeddings

    def get_batch_embeddings(self, texts: List[str],
                             domain: str) -> np.ndarray:
        """
        Get embeddings for multiple texts from a specific domain

        Args:
            texts: List of texts
            domain: Domain name

        Returns:
            Array of embeddings
        """
        if domain not in self.models:
            self.load_model(domain)

        model = self.models[domain]

        if isinstance(model, SentenceTransformer):
            embeddings = model.encode(texts, convert_to_numpy=True)
        else:
            embeddings = model.encode(texts)

        return embeddings