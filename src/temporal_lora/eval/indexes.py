"""FAISS index building and querying utilities."""

from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build FAISS IndexFlatIP for inner product (cosine similarity).
    
    Args:
        embeddings: Normalized embeddings (n, dim).
        
    Returns:
        FAISS index.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def save_faiss_index(index: faiss.IndexFlatIP, output_path: Path) -> None:
    """Save FAISS index to disk.
    
    Args:
        index: FAISS index.
        output_path: Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    logger.info(f"Saved FAISS index to: {output_path}")


def load_faiss_index(index_path: Path) -> faiss.IndexFlatIP:
    """Load FAISS index from disk.
    
    Args:
        index_path: Path to FAISS index file.
        
    Returns:
        FAISS index.
    """
    index = faiss.read_index(str(index_path))
    return index


def query_index(
    index: faiss.IndexFlatIP,
    query_embeddings: np.ndarray,
    k: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Query FAISS index for top-k similar items.
    
    Args:
        index: FAISS index.
        query_embeddings: Query embeddings (n_queries, dim).
        k: Number of results to retrieve.
        
    Returns:
        Tuple of (scores, indices). Both are (n_queries, k).
    """
    scores, indices = index.search(query_embeddings.astype(np.float32), k)
    return scores, indices


def build_bucket_indexes(
    embeddings_dir: Path,
    output_dir: Path,
    buckets: List[str],
) -> Dict[str, Path]:
    """Build FAISS indexes for all buckets.
    
    Args:
        embeddings_dir: Directory containing bucket embeddings.
        output_dir: Output directory for indexes.
        buckets: List of bucket names.
        
    Returns:
        Dictionary mapping bucket -> index path.
    """
    from .encoder import load_embeddings
    
    index_paths = {}
    
    for bucket_name in buckets:
        logger.info(f"Building index for bucket: {bucket_name}")
        
        # Combine all splits for this bucket
        all_embeddings = []
        all_ids = []
        
        for split in ["train", "val", "test"]:
            split_dir = embeddings_dir / bucket_name / split
            if not split_dir.exists():
                continue
            
            embeddings, ids = load_embeddings(split_dir)
            all_embeddings.append(embeddings)
            all_ids.extend(ids)
        
        if not all_embeddings:
            logger.warning(f"No embeddings found for bucket: {bucket_name}")
            continue
        
        # Concatenate all embeddings
        combined_embeddings = np.vstack(all_embeddings)
        logger.info(f"Building index with {len(all_ids)} documents")
        
        # Build index
        index = build_faiss_index(combined_embeddings)
        
        # Save index
        index_path = output_dir / f"{bucket_name}.faiss"
        save_faiss_index(index, index_path)
        
        # Save corresponding IDs
        ids_path = output_dir / f"{bucket_name}_ids.txt"
        with open(ids_path, "w") as f:
            for doc_id in all_ids:
                f.write(f"{doc_id}\n")
        
        index_paths[bucket_name] = index_path
        logger.info(f"✓ Index built for {bucket_name}: {len(all_ids)} docs")
    
    return index_paths


def multi_index_search(
    indexes: Dict[str, faiss.IndexFlatIP],
    index_ids: Dict[str, List[str]],
    query_embeddings: np.ndarray,
    k: int = 100,
    merge_strategy: str = "softmax",
    temperature: float = 2.0,
) -> Tuple[List[str], np.ndarray]:
    """Search across multiple indexes and merge results.
    
    Args:
        indexes: Dictionary mapping bucket -> FAISS index.
        index_ids: Dictionary mapping bucket -> list of document IDs.
        query_embeddings: Query embeddings (n_queries, dim).
        k: Number of results per query.
        merge_strategy: How to merge scores (softmax, mean, max, rrf).
        temperature: Temperature for softmax merge.
        
    Returns:
        Tuple of (document_ids, scores) for each query.
        document_ids: List of k IDs per query.
        scores: Array of shape (n_queries, k).
    """
    n_queries = query_embeddings.shape[0]
    
    # Search each index
    all_results = {}
    for bucket_name, index in indexes.items():
        scores, indices = query_index(index, query_embeddings, k=k)
        
        # Map indices to document IDs
        bucket_ids = index_ids[bucket_name]
        doc_ids = []
        for query_indices in indices:
            doc_ids.append([bucket_ids[idx] for idx in query_indices])
        
        all_results[bucket_name] = {
            "doc_ids": doc_ids,
            "scores": scores,
        }
    
    # Merge results
    merged_ids = []
    merged_scores = []
    
    for query_idx in range(n_queries):
        # Collect all candidates
        candidates = {}  # doc_id -> list of scores from different indexes
        
        for bucket_name, results in all_results.items():
            query_doc_ids = results["doc_ids"][query_idx]
            query_scores = results["scores"][query_idx]
            
            for doc_id, score in zip(query_doc_ids, query_scores):
                if doc_id not in candidates:
                    candidates[doc_id] = []
                candidates[doc_id].append(score)
        
        # Merge scores
        final_scores = {}
        for doc_id, scores_list in candidates.items():
            if merge_strategy == "softmax":
                # Temperature-scaled softmax
                scores_array = np.array(scores_list) / temperature
                exp_scores = np.exp(scores_array - np.max(scores_array))
                weights = exp_scores / np.sum(exp_scores)
                final_scores[doc_id] = np.sum(weights * scores_list)
            elif merge_strategy == "mean":
                final_scores[doc_id] = np.mean(scores_list)
            elif merge_strategy == "max":
                final_scores[doc_id] = np.max(scores_list)
            elif merge_strategy == "rrf":
                # Reciprocal rank fusion
                rrf_score = sum(1.0 / (60 + rank + 1) for rank in range(len(scores_list)))
                final_scores[doc_id] = rrf_score
            else:
                raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Sort by score and take top-k
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        merged_ids.append([doc_id for doc_id, _ in sorted_items])
        merged_scores.append([score for _, score in sorted_items])
    
    return merged_ids, np.array(merged_scores)
