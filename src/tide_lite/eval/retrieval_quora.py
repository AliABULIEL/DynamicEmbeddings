"""Quora retrieval evaluation with FAISS.

This module evaluates models on the Quora duplicate questions dataset
using FAISS for efficient similarity search.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    import warnings
    warnings.warn(
        "FAISS not installed. Install with: pip install faiss-cpu\n"
        "For GPU support: pip install faiss-gpu",
        ImportWarning,
        stacklevel=2
    )

from ..models.tide_lite import TIDELite
from ..models.baselines import load_baseline
from ..data.dataloaders import create_quora_dataloaders

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval evaluation.
    
    Attributes:
        ndcg_at_10: Normalized Discounted Cumulative Gain at 10.
        recall_at_10: Recall at 10.
        mrr_at_10: Mean Reciprocal Rank at 10.
        map_at_10: Mean Average Precision at 10.
        latency_median_ms: Median query latency in milliseconds.
        latency_p90_ms: 90th percentile query latency in milliseconds.
        latency_p99_ms: 99th percentile query latency in milliseconds.
        num_queries: Number of queries evaluated.
        num_docs: Number of documents in corpus.
        index_build_time_s: Time to build FAISS index in seconds.
    """
    ndcg_at_10: float
    recall_at_10: float
    mrr_at_10: float
    map_at_10: float
    latency_median_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    num_queries: int
    num_docs: int
    index_build_time_s: float


class QuoraRetrievalEvaluator:
    """Evaluator for Quora retrieval task using FAISS."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 128,
        max_seq_length: int = 128,
        faiss_index_type: str = "Flat",
        faiss_nlist: int = 100,
        faiss_nprobe: int = 10,
        use_gpu: bool = False,
    ) -> None:
        """Initialize Quora retrieval evaluator.
        
        Args:
            device: Device to use for embeddings (auto-detect if None).
            batch_size: Batch size for encoding.
            max_seq_length: Maximum sequence length (must be same for all models).
            faiss_index_type: Type of FAISS index ("Flat" or "IVF").
            faiss_nlist: Number of clusters for IVF index.
            faiss_nprobe: Number of probes for IVF search.
            use_gpu: Whether to use GPU for FAISS (if available).
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for retrieval evaluation. Install with: pip install faiss-cpu or faiss-gpu")
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.faiss_index_type = faiss_index_type
        self.faiss_nlist = faiss_nlist
        self.faiss_nprobe = faiss_nprobe
        self.use_gpu = use_gpu and torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources')
        
        logger.info(f"Initialized QuoraRetrievalEvaluator:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  FAISS index: {faiss_index_type}")
        logger.info(f"  FAISS GPU: {self.use_gpu}")
        logger.info(f"  Max seq length: {max_seq_length}")
    
    def load_model(
        self,
        model_id_or_path: str,
        model_type: str = "tide_lite",
    ) -> Union[TIDELite, torch.nn.Module]:
        """Load a model for evaluation.
        
        Args:
            model_id_or_path: Model identifier or path to saved model.
            model_type: Type of model ('tide_lite' or baseline name).
            
        Returns:
            Loaded model.
        """
        if model_type == "tide_lite":
            if Path(model_id_or_path).exists():
                model = TIDELite.from_pretrained(model_id_or_path)
                logger.info(f"Loaded TIDE-Lite from {model_id_or_path}")
            else:
                from ..models.tide_lite import TIDELiteConfig
                config = TIDELiteConfig(
                    encoder_name=model_id_or_path,
                    max_seq_length=self.max_seq_length,
                )
                model = TIDELite(config)
                logger.info(f"Initialized TIDE-Lite with encoder {model_id_or_path}")
        else:
            model = load_baseline(model_type, max_seq_length=self.max_seq_length)
            logger.info(f"Loaded baseline model: {model_type}")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    @torch.no_grad()
    def encode_corpus(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[np.ndarray, List[int]]:
        """Encode corpus documents.
        
        Args:
            model: Model to encode with.
            dataloader: DataLoader for corpus.
            
        Returns:
            Tuple of (embeddings, doc_ids).
        """
        all_embeddings = []
        all_doc_ids = []
        
        for batch in tqdm(dataloader, desc="Encoding corpus"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            doc_ids = batch["doc_ids"].tolist()
            
            # Get embeddings
            if hasattr(model, 'encode_base'):
                embeddings = model.encode_base(input_ids, attention_mask)
            else:
                embeddings, _ = model(input_ids, attention_mask)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_doc_ids.extend(doc_ids)
        
        embeddings = np.vstack(all_embeddings)
        return embeddings, all_doc_ids
    
    @torch.no_grad()
    def encode_queries(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> Tuple[np.ndarray, List[int]]:
        """Encode query texts.
        
        Args:
            model: Model to encode with.
            dataloader: DataLoader for queries.
            
        Returns:
            Tuple of (embeddings, query_ids).
        """
        all_embeddings = []
        all_query_ids = []
        
        for batch in tqdm(dataloader, desc="Encoding queries"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            query_ids = batch["query_ids"].tolist()
            
            # Get embeddings
            if hasattr(model, 'encode_base'):
                embeddings = model.encode_base(input_ids, attention_mask)
            else:
                embeddings, _ = model(input_ids, attention_mask)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_query_ids.extend(query_ids)
        
        embeddings = np.vstack(all_embeddings)
        return embeddings, all_query_ids
    
    def build_faiss_index(
        self,
        corpus_embeddings: np.ndarray,
        embedding_dim: int,
    ) -> faiss.Index:
        """Build FAISS index for corpus.
        
        Args:
            corpus_embeddings: Corpus embeddings [num_docs, embedding_dim].
            embedding_dim: Dimension of embeddings.
            
        Returns:
            FAISS index.
        """
        start_time = time.perf_counter()
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(corpus_embeddings)
        
        if self.faiss_index_type == "IVF":
            # IVF index for large-scale search
            quantizer = faiss.IndexFlatIP(embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, self.faiss_nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(corpus_embeddings)
            index.add(corpus_embeddings)
            index.nprobe = self.faiss_nprobe
            logger.info(f"Built IVF index with {self.faiss_nlist} clusters, nprobe={self.faiss_nprobe}")
        else:
            # Flat index for exact search
            index = faiss.IndexFlatIP(embedding_dim)
            index.add(corpus_embeddings)
            logger.info("Built Flat index for exact search")
        
        if self.use_gpu:
            # Move index to GPU if available
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("Moved FAISS index to GPU")
        
        build_time = time.perf_counter() - start_time
        logger.info(f"Index build time: {build_time:.2f} seconds")
        
        return index, build_time
    
    def search(
        self,
        index: faiss.Index,
        query_embeddings: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Search for nearest neighbors.
        
        Args:
            index: FAISS index.
            query_embeddings: Query embeddings.
            k: Number of neighbors to retrieve.
            
        Returns:
            Tuple of (distances, indices, latencies).
        """
        # Normalize queries for cosine similarity
        faiss.normalize_L2(query_embeddings)
        
        latencies = []
        all_distances = []
        all_indices = []
        
        # Search in batches to measure latency
        for i in range(len(query_embeddings)):
            query = query_embeddings[i:i+1]
            
            start_time = time.perf_counter()
            distances, indices = index.search(query, k)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            latencies.append(latency_ms)
            all_distances.append(distances)
            all_indices.append(indices)
        
        distances = np.vstack(all_distances)
        indices = np.vstack(all_indices)
        
        return distances, indices, latencies
    
    def compute_metrics(
        self,
        retrieved_indices: np.ndarray,
        query_ids: List[int],
        doc_ids: List[int],
        qrels: Dict[int, Dict[int, float]],
        k: int = 10,
    ) -> Dict[str, float]:
        """Compute retrieval metrics.
        
        Args:
            retrieved_indices: Retrieved document indices [num_queries, k].
            query_ids: Query IDs.
            doc_ids: Document IDs.
            qrels: Relevance judgments {query_id: {doc_id: relevance}}.
            k: Cutoff for metrics.
            
        Returns:
            Dictionary of metrics.
        """
        ndcg_scores = []
        recall_scores = []
        mrr_scores = []
        map_scores = []
        
        for i, query_id in enumerate(query_ids):
            if query_id not in qrels:
                continue
            
            relevant_docs = qrels[query_id]
            retrieved_docs = [doc_ids[idx] for idx in retrieved_indices[i][:k]]
            
            # NDCG@k
            dcg = 0.0
            idcg = 0.0
            for rank, doc_id in enumerate(retrieved_docs, 1):
                if doc_id in relevant_docs:
                    dcg += relevant_docs[doc_id] / np.log2(rank + 1)
            
            sorted_rels = sorted(relevant_docs.values(), reverse=True)
            for rank, rel in enumerate(sorted_rels[:k], 1):
                idcg += rel / np.log2(rank + 1)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
            
            # Recall@k
            num_relevant_retrieved = sum(1 for doc_id in retrieved_docs if doc_id in relevant_docs)
            num_relevant_total = len(relevant_docs)
            recall = num_relevant_retrieved / num_relevant_total if num_relevant_total > 0 else 0
            recall_scores.append(recall)
            
            # MRR@k
            rr = 0.0
            for rank, doc_id in enumerate(retrieved_docs, 1):
                if doc_id in relevant_docs:
                    rr = 1.0 / rank
                    break
            mrr_scores.append(rr)
            
            # MAP@k
            ap = 0.0
            num_relevant_found = 0
            for rank, doc_id in enumerate(retrieved_docs, 1):
                if doc_id in relevant_docs:
                    num_relevant_found += 1
                    ap += num_relevant_found / rank
            ap = ap / min(k, num_relevant_total) if num_relevant_total > 0 else 0
            map_scores.append(ap)
        
        return {
            "ndcg_at_10": np.mean(ndcg_scores),
            "recall_at_10": np.mean(recall_scores),
            "mrr_at_10": np.mean(mrr_scores),
            "map_at_10": np.mean(map_scores),
        }
    
    def evaluate(
        self,
        model_id_or_path: str,
        model_type: str = "tide_lite",
        max_corpus_size: Optional[int] = 10000,
        max_queries: Optional[int] = 1000,
        output_dir: Optional[str] = None,
        save_results: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, any]:
        """Evaluate a model on Quora retrieval.
        
        Args:
            model_id_or_path: Model identifier or path.
            model_type: Type of model.
            max_corpus_size: Maximum corpus size (for efficiency).
            max_queries: Maximum number of queries.
            output_dir: Directory to save results.
            save_results: Whether to save results to JSON.
            dry_run: If True, just print plan without execution.
            
        Returns:
            Dictionary with evaluation results.
        """
        if dry_run:
            logger.info("[DRY RUN] Would evaluate model on Quora retrieval:")
            logger.info(f"  Model: {model_id_or_path}")
            logger.info(f"  Type: {model_type}")
            logger.info(f"  Max corpus: {max_corpus_size}")
            logger.info(f"  Max queries: {max_queries}")
            logger.info(f"  FAISS index: {self.faiss_index_type}")
            logger.info(f"  Max seq length: {self.max_seq_length}")
            if save_results:
                model_name = Path(model_id_or_path).name if "/" in model_id_or_path else model_id_or_path
                output_file = f"results/metrics_quora_{model_name}.json"
                logger.info(f"  Would save to: {output_file}")
            return {
                "dry_run": True,
                "model": model_id_or_path,
                "faiss_index": self.faiss_index_type,
            }
        
        # Load model
        logger.info(f"Loading model: {model_id_or_path}")
        model = self.load_model(model_id_or_path, model_type)
        
        # Get embedding dimension
        if hasattr(model, 'embedding_dim'):
            embedding_dim = model.embedding_dim
        elif hasattr(model, 'hidden_dim'):
            embedding_dim = model.hidden_dim
        else:
            embedding_dim = 384  # Default for MiniLM
        
        # Load data
        logger.info("Loading Quora retrieval data")
        data_config = {
            "cache_dir": "./data",
            "seed": 42,
            "model_name": model.config.encoder_name if hasattr(model, 'config') else model.model_name,
        }
        
        corpus_loader, query_loader, qrels_dataset = create_quora_dataloaders(
            cfg=data_config,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            num_workers=2,
            max_corpus_size=max_corpus_size,
            max_queries=max_queries,
        )
        
        # Convert qrels to dict format
        qrels = {}
        for item in qrels_dataset:
            query_id = item["query_id"]
            doc_id = item["doc_id"]
            relevance = item["relevance"]
            
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = relevance
        
        # Encode corpus
        logger.info("Encoding corpus...")
        corpus_embeddings, doc_ids = self.encode_corpus(model, corpus_loader)
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        index, build_time = self.build_faiss_index(corpus_embeddings, embedding_dim)
        
        # Encode queries
        logger.info("Encoding queries...")
        query_embeddings, query_ids = self.encode_queries(model, query_loader)
        
        # Search
        logger.info("Searching...")
        distances, indices, latencies = self.search(index, query_embeddings, k=10)
        
        # Compute metrics
        logger.info("Computing metrics...")
        metrics = self.compute_metrics(indices, query_ids, doc_ids, qrels, k=10)
        
        # Add latency stats
        latencies_np = np.array(latencies)
        metrics.update({
            "latency_median_ms": float(np.median(latencies_np)),
            "latency_p90_ms": float(np.percentile(latencies_np, 90)),
            "latency_p99_ms": float(np.percentile(latencies_np, 99)),
            "num_queries": len(query_ids),
            "num_docs": len(doc_ids),
            "index_build_time_s": float(build_time),
        })
        
        # Create RetrievalMetrics object
        retrieval_metrics = RetrievalMetrics(**metrics)
        
        # Prepare results
        results = {
            "model": model_id_or_path,
            "model_type": model_type,
            "metrics": asdict(retrieval_metrics),
            "config": {
                "batch_size": self.batch_size,
                "max_seq_length": self.max_seq_length,
                "faiss_index_type": self.faiss_index_type,
                "faiss_nlist": self.faiss_nlist if self.faiss_index_type == "IVF" else None,
                "faiss_nprobe": self.faiss_nprobe if self.faiss_index_type == "IVF" else None,
                "use_gpu": self.use_gpu,
                "max_corpus_size": max_corpus_size,
                "max_queries": max_queries,
            },
        }
        
        # Print results
        logger.info("=" * 60)
        logger.info("Quora Retrieval Evaluation Results")
        logger.info("=" * 60)
        logger.info(f"Model: {model_id_or_path}")
        logger.info(f"Corpus: {retrieval_metrics.num_docs} docs")
        logger.info(f"Queries: {retrieval_metrics.num_queries}")
        logger.info("-" * 40)
        logger.info(f"nDCG@10: {retrieval_metrics.ndcg_at_10:.4f}")
        logger.info(f"Recall@10: {retrieval_metrics.recall_at_10:.4f}")
        logger.info(f"MRR@10: {retrieval_metrics.mrr_at_10:.4f}")
        logger.info(f"MAP@10: {retrieval_metrics.map_at_10:.4f}")
        logger.info("-" * 40)
        logger.info(f"Latency (median): {retrieval_metrics.latency_median_ms:.2f} ms")
        logger.info(f"Latency (p90): {retrieval_metrics.latency_p90_ms:.2f} ms")
        logger.info(f"Latency (p99): {retrieval_metrics.latency_p99_ms:.2f} ms")
        logger.info(f"Index build time: {retrieval_metrics.index_build_time_s:.2f} s")
        logger.info("=" * 60)
        
        # Save results
        if save_results:
            if output_dir is None:
                output_dir = "results"
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create filename based on model
            model_name = Path(model_id_or_path).name if "/" in model_id_or_path else model_id_or_path
            model_name = model_name.replace("/", "_")
            output_file = output_path / f"metrics_quora_{model_name}.json"
            
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to: {output_file}")
        
        return results


def evaluate_quora_retrieval(
    model_id_or_path: str,
    model_type: str = "tide_lite",
    max_corpus_size: Optional[int] = 10000,
    max_queries: Optional[int] = 1000,
    batch_size: int = 128,
    max_seq_length: int = 128,
    faiss_index_type: str = "Flat",
    output_dir: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, any]:
    """Convenience function to evaluate a model on Quora retrieval.
    
    Args:
        model_id_or_path: Model identifier or path.
        model_type: Type of model.
        max_corpus_size: Maximum corpus size.
        max_queries: Maximum number of queries.
        batch_size: Batch size for encoding.
        max_seq_length: Maximum sequence length (must be same for all models).
        faiss_index_type: Type of FAISS index.
        output_dir: Directory to save results.
        dry_run: If True, just print plan without execution.
        
    Returns:
        Dictionary with evaluation results.
    """
    evaluator = QuoraRetrievalEvaluator(
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        faiss_index_type=faiss_index_type,
    )
    
    return evaluator.evaluate(
        model_id_or_path=model_id_or_path,
        model_type=model_type,
        max_corpus_size=max_corpus_size,
        max_queries=max_queries,
        output_dir=output_dir,
        dry_run=dry_run,
    )
