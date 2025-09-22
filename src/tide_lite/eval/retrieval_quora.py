"""Quora duplicate questions retrieval evaluation module.

This module evaluates retrieval performance on the Quora dataset
using FAISS indexing with nDCG@10 and Recall@10 metrics.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.collate import RetrievalCollator, TextBatcher
from ..data.datasets import DatasetConfig, load_quora
from ..models.tide_lite import TIDELite
from ..models.baselines import BaselineEncoder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval evaluation.
    
    Attributes:
        ndcg_at_10: Normalized Discounted Cumulative Gain at 10.
        recall_at_10: Recall at 10.
        recall_at_1: Recall at 1 (exact match).
        recall_at_5: Recall at 5.
        mean_reciprocal_rank: Mean Reciprocal Rank.
        avg_query_time_ms: Average query time in milliseconds.
        index_build_time_s: Time to build FAISS index in seconds.
        total_eval_time_s: Total evaluation time in seconds.
        num_queries: Number of queries evaluated.
        corpus_size: Size of document corpus.
    """
    ndcg_at_10: float
    recall_at_10: float
    recall_at_1: float
    recall_at_5: float
    mean_reciprocal_rank: float
    avg_query_time_ms: float
    index_build_time_s: float
    total_eval_time_s: float
    num_queries: int
    corpus_size: int


class QuoraRetrievalEvaluator:
    """Evaluator for Quora duplicate questions retrieval task.
    
    Uses FAISS for efficient similarity search and computes
    standard retrieval metrics.
    """
    
    def __init__(
        self,
        model: Union[TIDELite, BaselineEncoder],
        index_type: str = "Flat",
        device: Optional[torch.device] = None,
        use_temporal: bool = False,
    ) -> None:
        """Initialize Quora retrieval evaluator.
        
        Args:
            model: Model to evaluate (TIDE-Lite or baseline).
            index_type: FAISS index type ('Flat' or 'IVFFlat').
            device: Device for computation (auto-detect if None).
            use_temporal: Whether to use temporal modulation (not applicable for Quora).
        """
        self.model = model
        self.index_type = index_type
        self.use_temporal = use_temporal and isinstance(model, TIDELite)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # FAISS index (will be built during evaluation)
        self.index = None
        self.corpus_embeddings = None
        
        logger.info(
            f"Initialized Quora retrieval evaluator with model type: {type(model).__name__}, "
            f"index: {index_type}, device: {self.device}"
        )
    
    def prepare_data(
        self,
        max_corpus_size: Optional[int] = None,
        max_queries: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader, Dict]:
        """Prepare Quora data for retrieval evaluation.
        
        Args:
            max_corpus_size: Maximum number of documents (None for all).
            max_queries: Maximum number of queries (None for all).
            
        Returns:
            Tuple of (corpus_loader, query_loader, qrels).
        """
        # Load Quora dataset
        dataset_config = DatasetConfig(
            seed=42,
            cache_dir="./data",
            max_samples=max_corpus_size,
        )
        
        corpus, queries, qrels = load_quora(dataset_config)
        
        if max_queries:
            queries = queries.select(range(min(max_queries, len(queries))))
            # Filter qrels to match selected queries
            query_ids = set(queries["query_id"])
            qrels_list = []
            for item in qrels:
                if item["query_id"] in query_ids:
                    qrels_list.append(item)
            qrels = qrels.from_list(qrels_list[:max_queries])
        
        # Create tokenizer
        tokenizer = TextBatcher(
            model_name=getattr(self.model, "config", None).encoder_name 
            if hasattr(self.model, "config") else "sentence-transformers/all-MiniLM-L6-v2",
            max_length=128,
        )
        
        # Create collators
        corpus_collator = RetrievalCollator(tokenizer, is_corpus=True)
        query_collator = RetrievalCollator(tokenizer, is_corpus=False)
        
        # Create dataloaders
        corpus_loader = DataLoader(
            corpus,
            batch_size=128,
            shuffle=False,
            collate_fn=corpus_collator,
            num_workers=2,
            pin_memory=True,
        )
        
        query_loader = DataLoader(
            queries,
            batch_size=64,
            shuffle=False,
            collate_fn=query_collator,
            num_workers=2,
            pin_memory=True,
        )
        
        # Convert qrels to dict for efficient lookup
        qrels_dict = {}
        for item in qrels:
            query_id = item["query_id"]
            doc_id = item["doc_id"]
            relevance = item["relevance"]
            
            if query_id not in qrels_dict:
                qrels_dict[query_id] = []
            qrels_dict[query_id].append(doc_id)
        
        logger.info(
            f"Prepared Quora data - Corpus: {len(corpus)} docs, "
            f"Queries: {len(queries)}, Qrels: {len(qrels_dict)}"
        )
        
        return corpus_loader, query_loader, qrels_dict
    
    @torch.no_grad()
    def encode_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> np.ndarray:
        """Encode a batch of texts to embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            
        Returns:
            Embeddings as numpy array [batch_size, hidden_dim].
        """
        if hasattr(self.model, "encode_base"):
            embeddings = self.model.encode_base(input_ids, attention_mask)
        else:
            # Generic forward pass
            embeddings, _ = self.model(input_ids, attention_mask, None)
        
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def build_faiss_index(
        self,
        corpus_loader: DataLoader,
    ) -> Tuple[faiss.Index, np.ndarray, List[int]]:
        """Build FAISS index from corpus embeddings.
        
        Args:
            corpus_loader: DataLoader for corpus documents.
            
        Returns:
            Tuple of (index, embeddings, doc_ids).
        """
        logger.info(f"Building FAISS {self.index_type} index")
        
        all_embeddings = []
        all_doc_ids = []
        
        start_time = time.time()
        
        for batch in tqdm(corpus_loader, desc="Encoding corpus"):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            doc_ids = batch["doc_ids"].numpy()
            
            # Encode batch
            embeddings = self.encode_batch(input_ids, attention_mask)
            
            all_embeddings.append(embeddings)
            all_doc_ids.extend(doc_ids)
        
        # Concatenate all embeddings
        corpus_embeddings = np.vstack(all_embeddings).astype(np.float32)
        hidden_dim = corpus_embeddings.shape[1]
        
        # Create FAISS index
        if self.index_type == "Flat":
            # Exact search with cosine similarity (via inner product on normalized vectors)
            index = faiss.IndexFlatIP(hidden_dim)
        elif self.index_type == "IVFFlat":
            # Approximate search with inverted file index
            quantizer = faiss.IndexFlatIP(hidden_dim)
            n_clusters = min(100, len(corpus_embeddings) // 10)
            index = faiss.IndexIVFFlat(quantizer, hidden_dim, n_clusters)
            
            # Train index on corpus
            logger.info(f"Training IVFFlat index with {n_clusters} clusters")
            index.train(corpus_embeddings)
            
            # Set search parameters
            index.nprobe = 10  # Number of clusters to search
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Add embeddings to index
        index.add(corpus_embeddings)
        
        build_time = time.time() - start_time
        
        logger.info(
            f"Built index with {len(corpus_embeddings)} documents "
            f"in {build_time:.2f} seconds"
        )
        
        return index, corpus_embeddings, all_doc_ids
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search index for nearest neighbors.
        
        Args:
            query_embeddings: Query embeddings [n_queries, hidden_dim].
            k: Number of nearest neighbors to retrieve.
            
        Returns:
            Tuple of (distances, indices) both [n_queries, k].
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_faiss_index first.")
        
        # Search index
        distances, indices = self.index.search(query_embeddings, k)
        
        return distances, indices
    
    def compute_ndcg(
        self,
        retrieved_ids: List[List[int]],
        relevant_ids: List[List[int]],
        k: int = 10,
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain at k.
        
        Args:
            retrieved_ids: Retrieved document IDs per query.
            relevant_ids: Relevant document IDs per query.
            k: Cutoff position.
            
        Returns:
            Average nDCG@k across queries.
        """
        ndcg_scores = []
        
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            relevant_set = set(relevant)
            
            # Compute DCG
            dcg = 0.0
            for i, doc_id in enumerate(retrieved[:k]):
                if doc_id in relevant_set:
                    # Binary relevance: 1 if relevant, 0 otherwise
                    dcg += 1.0 / np.log2(i + 2)  # i+2 because positions start at 1
            
            # Compute IDCG (ideal DCG)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
            
            # Compute nDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def compute_recall(
        self,
        retrieved_ids: List[List[int]],
        relevant_ids: List[List[int]],
        k: int = 10,
    ) -> float:
        """Compute Recall at k.
        
        Args:
            retrieved_ids: Retrieved document IDs per query.
            relevant_ids: Relevant document IDs per query.
            k: Cutoff position.
            
        Returns:
            Average Recall@k across queries.
        """
        recall_scores = []
        
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            relevant_set = set(relevant)
            retrieved_set = set(retrieved[:k])
            
            # Compute recall
            if relevant_set:
                recall = len(relevant_set & retrieved_set) / len(relevant_set)
            else:
                recall = 0.0
            
            recall_scores.append(recall)
        
        return np.mean(recall_scores) if recall_scores else 0.0
    
    def compute_mrr(
        self,
        retrieved_ids: List[List[int]],
        relevant_ids: List[List[int]],
    ) -> float:
        """Compute Mean Reciprocal Rank.
        
        Args:
            retrieved_ids: Retrieved document IDs per query.
            relevant_ids: Relevant document IDs per query.
            
        Returns:
            Mean Reciprocal Rank across queries.
        """
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            relevant_set = set(relevant)
            
            # Find rank of first relevant document
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant_set:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def evaluate(
        self,
        corpus_loader: Optional[DataLoader] = None,
        query_loader: Optional[DataLoader] = None,
        qrels: Optional[Dict] = None,
        max_corpus_size: Optional[int] = None,
        max_queries: Optional[int] = None,
    ) -> RetrievalMetrics:
        """Evaluate model on Quora retrieval task.
        
        Args:
            corpus_loader: Optional corpus dataloader.
            query_loader: Optional query dataloader.
            qrels: Optional relevance judgments.
            max_corpus_size: Maximum corpus size if creating new loaders.
            max_queries: Maximum queries if creating new loaders.
            
        Returns:
            RetrievalMetrics with evaluation results.
        """
        eval_start = time.time()
        
        # Prepare data if not provided
        if corpus_loader is None or query_loader is None or qrels is None:
            corpus_loader, query_loader, qrels = self.prepare_data(
                max_corpus_size, max_queries
            )
        
        # Build FAISS index
        index_start = time.time()
        self.index, self.corpus_embeddings, doc_id_map = self.build_faiss_index(corpus_loader)
        index_build_time = time.time() - index_start
        
        # Evaluate queries
        logger.info("Evaluating queries")
        
        all_retrieved = []
        all_relevant = []
        query_times = []
        
        for batch in tqdm(query_loader, desc="Evaluating queries"):
            query_start = time.perf_counter()
            
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            query_ids = batch["query_ids"].numpy()
            
            # Encode queries
            query_embeddings = self.encode_batch(input_ids, attention_mask)
            
            # Search index
            distances, indices = self.search(query_embeddings, k=10)
            
            query_time = (time.perf_counter() - query_start) * 1000  # ms
            query_times.append(query_time / len(query_ids))
            
            # Map indices to document IDs
            for i, query_id in enumerate(query_ids):
                retrieved_doc_ids = [doc_id_map[idx] for idx in indices[i]]
                relevant_doc_ids = qrels.get(query_id, [])
                
                all_retrieved.append(retrieved_doc_ids)
                all_relevant.append(relevant_doc_ids)
        
        # Compute metrics
        ndcg_10 = self.compute_ndcg(all_retrieved, all_relevant, k=10)
        recall_10 = self.compute_recall(all_retrieved, all_relevant, k=10)
        recall_5 = self.compute_recall(all_retrieved, all_relevant, k=5)
        recall_1 = self.compute_recall(all_retrieved, all_relevant, k=1)
        mrr = self.compute_mrr(all_retrieved, all_relevant)
        
        total_time = time.time() - eval_start
        
        metrics = RetrievalMetrics(
            ndcg_at_10=float(ndcg_10),
            recall_at_10=float(recall_10),
            recall_at_1=float(recall_1),
            recall_at_5=float(recall_5),
            mean_reciprocal_rank=float(mrr),
            avg_query_time_ms=float(np.mean(query_times)),
            index_build_time_s=index_build_time,
            total_eval_time_s=total_time,
            num_queries=len(all_retrieved),
            corpus_size=len(self.corpus_embeddings),
        )
        
        logger.info(
            f"Retrieval evaluation complete - "
            f"nDCG@10: {metrics.ndcg_at_10:.4f}, "
            f"Recall@10: {metrics.recall_at_10:.4f}, "
            f"MRR: {metrics.mean_reciprocal_rank:.4f}"
        )
        
        return metrics
    
    def save_results(
        self,
        metrics: RetrievalMetrics,
        output_dir: Union[str, Path],
        model_name: str = "model",
    ) -> Path:
        """Save evaluation metrics to JSON file.
        
        Args:
            metrics: Evaluation metrics.
            output_dir: Directory to save results.
            model_name: Model identifier for filename.
            
        Returns:
            Path to saved metrics file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        metrics_file = output_dir / f"metrics_quora_{model_name}.json"
        
        # Prepare metrics dict
        metrics_dict = asdict(metrics)
        metrics_dict["model_name"] = model_name
        metrics_dict["task"] = "Quora Retrieval"
        metrics_dict["index_type"] = self.index_type
        
        # Save to JSON
        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Saved Quora retrieval metrics to {metrics_file}")
        
        return metrics_file
