"""Hard temporal negatives generation for contrastive learning.

Samples hard negatives from adjacent time bins with high lexical overlap.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import InputExample

from ..utils.logging import get_logger
from ..utils.seeding import get_rng

logger = get_logger(__name__)

# Try to import rank_bm25, fallback to TF-IDF if not available
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    logger.warning("rank_bm25 not available, will use TF-IDF fallback for hard negatives")
    HAS_BM25 = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity


def tokenize_text(text: str) -> List[str]:
    """Simple tokenization by splitting on whitespace.
    
    Args:
        text: Input text.
        
    Returns:
        List of tokens.
    """
    return text.lower().split()


def get_adjacent_bins(
    current_bin: str,
    all_bins: List[str],
) -> List[str]:
    """Get adjacent time bins (t±1) for hard negative sampling.
    
    Args:
        current_bin: Current time bin name.
        all_bins: List of all bin names in chronological order.
        
    Returns:
        List of adjacent bin names.
    """
    try:
        current_idx = all_bins.index(current_bin)
    except ValueError:
        logger.warning(f"Bin {current_bin} not in bins list, using all other bins")
        return [b for b in all_bins if b != current_bin]
    
    adjacent = []
    
    # Previous bin
    if current_idx > 0:
        adjacent.append(all_bins[current_idx - 1])
    
    # Next bin
    if current_idx < len(all_bins) - 1:
        adjacent.append(all_bins[current_idx + 1])
    
    if not adjacent:
        # If no adjacent bins (only one bin), use all other bins
        adjacent = [b for b in all_bins if b != current_bin]
    
    logger.debug(f"Adjacent bins for '{current_bin}': {adjacent}")
    return adjacent


class HardNegativeSampler:
    """Samples hard negatives from adjacent time bins using BM25 or TF-IDF."""
    
    def __init__(
        self,
        data_dir: Path,
        all_bins: List[str],
        neg_k: int = 4,
        seed: int = 42,
    ):
        """Initialize hard negative sampler.
        
        Args:
            data_dir: Directory containing bucket data.
            all_bins: List of all bin names in chronological order.
            neg_k: Number of hard negatives to sample per positive.
            seed: Random seed.
        """
        self.data_dir = data_dir
        self.all_bins = all_bins
        self.neg_k = neg_k
        self.rng = get_rng(seed)
        
        # Load and index all data by bin
        self.bin_data: Dict[str, pd.DataFrame] = {}
        self._load_all_bins()
    
    def _load_all_bins(self) -> None:
        """Load train data for all bins."""
        logger.info(f"Loading data from {len(self.all_bins)} bins for hard negatives...")
        
        for bin_name in self.all_bins:
            bin_path = self.data_dir / bin_name / "train.parquet"
            
            if not bin_path.exists():
                logger.warning(f"Train data not found for bin: {bin_name}")
                continue
            
            df = pd.read_parquet(bin_path)
            
            # Combine text_a and text_b for full document representation
            df["full_text"] = df["text_a"] + " " + df["text_b"]
            
            self.bin_data[bin_name] = df
            logger.debug(f"Loaded {len(df)} examples from bin: {bin_name}")
        
        logger.info(f"✓ Loaded data from {len(self.bin_data)} bins")
    
    def sample_hard_negatives_bm25(
        self,
        query_text: str,
        candidate_bin: str,
        k: int,
        exclude_ids: set,
    ) -> List[Tuple[str, str]]:
        """Sample hard negatives using BM25 lexical matching.
        
        Args:
            query_text: Query text (positive example).
            candidate_bin: Bin to sample negatives from.
            k: Number of negatives to sample.
            exclude_ids: Paper IDs to exclude.
            
        Returns:
            List of (text_a, text_b) tuples for negatives.
        """
        if candidate_bin not in self.bin_data:
            logger.warning(f"No data for bin: {candidate_bin}")
            return []
        
        candidate_df = self.bin_data[candidate_bin]
        
        # Filter out excluded IDs
        candidate_df = candidate_df[~candidate_df["paper_id"].isin(exclude_ids)]
        
        if len(candidate_df) == 0:
            return []
        
        # Build BM25 index
        corpus_texts = candidate_df["full_text"].tolist()
        tokenized_corpus = [tokenize_text(text) for text in corpus_texts]
        
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Score query against corpus
        tokenized_query = tokenize_text(query_text)
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        # Extract negatives
        negatives = []
        for idx in top_k_indices:
            row = candidate_df.iloc[idx]
            negatives.append((row["text_a"], row["text_b"]))
        
        return negatives
    
    def sample_hard_negatives_tfidf(
        self,
        query_text: str,
        candidate_bin: str,
        k: int,
        exclude_ids: set,
    ) -> List[Tuple[str, str]]:
        """Sample hard negatives using TF-IDF + cosine similarity.
        
        Args:
            query_text: Query text (positive example).
            candidate_bin: Bin to sample negatives from.
            k: Number of negatives to sample.
            exclude_ids: Paper IDs to exclude.
            
        Returns:
            List of (text_a, text_b) tuples for negatives.
        """
        if candidate_bin not in self.bin_data:
            logger.warning(f"No data for bin: {candidate_bin}")
            return []
        
        candidate_df = self.bin_data[candidate_bin]
        
        # Filter out excluded IDs
        candidate_df = candidate_df[~candidate_df["paper_id"].isin(exclude_ids)]
        
        if len(candidate_df) == 0:
            return []
        
        # Build TF-IDF index
        corpus_texts = candidate_df["full_text"].tolist()
        all_texts = [query_text] + corpus_texts
        
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Compute cosine similarity
        query_vec = tfidf_matrix[0:1]
        corpus_vecs = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vec, corpus_vecs).flatten()
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Extract negatives
        negatives = []
        for idx in top_k_indices:
            row = candidate_df.iloc[idx]
            negatives.append((row["text_a"], row["text_b"]))
        
        return negatives
    
    def augment_with_hard_negatives(
        self,
        current_bin: str,
        train_examples: List[InputExample],
    ) -> List[InputExample]:
        """Augment training examples with hard temporal negatives.
        
        For each positive pair in the current bin, sample k negatives from
        adjacent bins (t±1) with high lexical overlap.
        
        Args:
            current_bin: Current time bin name.
            train_examples: Original training examples (positive pairs only).
            
        Returns:
            Augmented examples with hard negatives added.
        """
        logger.info(f"Augmenting {len(train_examples)} examples with hard negatives...")
        logger.info(f"Current bin: {current_bin}, neg_k: {self.neg_k}")
        
        # Get adjacent bins
        adjacent_bins = get_adjacent_bins(current_bin, self.all_bins)
        
        if not adjacent_bins:
            logger.warning("No adjacent bins found, returning original examples")
            return train_examples
        
        logger.info(f"Sampling from adjacent bins: {adjacent_bins}")
        
        # Collect positive IDs to exclude
        positive_ids = set()
        for example in train_examples:
            # InputExample.guid may contain paper_id
            if hasattr(example, "guid") and example.guid:
                positive_ids.add(example.guid)
        
        # Sample hard negatives for each positive example
        augmented_examples = []
        sample_method = self.sample_hard_negatives_bm25 if HAS_BM25 else self.sample_hard_negatives_tfidf
        method_name = "BM25" if HAS_BM25 else "TF-IDF"
        
        for i, example in enumerate(train_examples):
            # Add original positive example
            augmented_examples.append(example)
            
            # Combine texts for query
            query_text = example.texts[0] + " " + example.texts[1]
            
            # Sample negatives from each adjacent bin
            negatives_per_bin = max(1, self.neg_k // len(adjacent_bins))
            
            for adj_bin in adjacent_bins:
                negatives = sample_method(
                    query_text=query_text,
                    candidate_bin=adj_bin,
                    k=negatives_per_bin,
                    exclude_ids=positive_ids,
                )
                
                # Create negative examples
                for neg_a, neg_b in negatives:
                    # Label 0 indicates negative pair
                    neg_example = InputExample(
                        texts=[example.texts[0], neg_a],
                        label=0.0,
                        guid=f"{current_bin}_neg_{i}_{adj_bin}",
                    )
                    augmented_examples.append(neg_example)
            
            if (i + 1) % 100 == 0:
                logger.debug(f"Processed {i + 1}/{len(train_examples)} examples")
        
        logger.info(
            f"✓ Augmented {len(train_examples)} → {len(augmented_examples)} examples "
            f"using {method_name} (ratio: {len(augmented_examples) / len(train_examples):.1f}x)"
        )
        
        return augmented_examples


def add_hard_temporal_negatives(
    data_dir: Path,
    all_bins: List[str],
    bucket_name: str,
    train_examples: List[InputExample],
    neg_k: int = 4,
    seed: int = 42,
) -> List[InputExample]:
    """Add hard temporal negatives to training examples.
    
    This is the main entry point for hard negative augmentation.
    
    Args:
        data_dir: Directory containing bucket data.
        all_bins: List of all bin names in chronological order.
        bucket_name: Current bucket name.
        train_examples: Original training examples.
        neg_k: Number of hard negatives per positive.
        seed: Random seed.
        
    Returns:
        Augmented training examples.
    """
    sampler = HardNegativeSampler(
        data_dir=data_dir,
        all_bins=all_bins,
        neg_k=neg_k,
        seed=seed,
    )
    
    return sampler.augment_with_hard_negatives(bucket_name, train_examples)
