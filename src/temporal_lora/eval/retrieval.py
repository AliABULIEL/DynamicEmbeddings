"""FAISS retrieval utilities for benchmark."""

from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np


class FAISSRetriever:
    """Simple FAISS retrieval wrapper for benchmarking."""
    
    def __init__(self, embedding_dim: int):
        """Initialize retriever.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.doc_ids = []
    
    def build_index(self, embeddings: np.ndarray, doc_ids: List[str]):
        """Build FAISS index from embeddings.
        
        Args:
            embeddings: Document embeddings (n_docs, dim)
            doc_ids: Document IDs
        """
        # Ensure contiguous float32
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        # Build index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        self.doc_ids = doc_ids
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 100
    ) -> List[List[Tuple[str, float]]]:
        """Search for top-k documents.
        
        Args:
            query_embeddings: Query embeddings (n_queries, dim)
            k: Number of results per query
            
        Returns:
            List of [(doc_id, score), ...] for each query
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Ensure contiguous float32
        query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        
        # Search
        scores, indices = self.index.search(query_embeddings, k)
        
        # Convert to list of results
        results = []
        for i in range(len(query_embeddings)):
            query_results = [
                (self.doc_ids[idx], float(scores[i][j]))
                for j, idx in enumerate(indices[i])
                if idx < len(self.doc_ids)
            ]
            results.append(query_results)
        
        return results
