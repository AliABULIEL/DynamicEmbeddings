"""FAST Hard temporal negatives using Sentence Transformers built-in function.

Uses FAISS-accelerated similarity search - takes seconds instead of hours.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.util import mine_hard_negatives
from datasets import Dataset

from ..utils.logging import get_logger

logger = get_logger(__name__)


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
    
    return adjacent


def load_adjacent_bin_corpus(
    data_dir: Path,
    adjacent_bins: List[str],
) -> List[str]:
    """Load corpus texts from adjacent bins.
    
    Args:
        data_dir: Directory containing bucket data.
        adjacent_bins: List of adjacent bin names.
        
    Returns:
        List of corpus texts.
    """
    corpus = []
    
    for bin_name in adjacent_bins:
        bin_path = data_dir / bin_name / "train.parquet"
        
        if not bin_path.exists():
            logger.warning(f"Train data not found for bin: {bin_name}")
            continue
        
        df = pd.read_parquet(bin_path)
        
        # Combine text_a and text_b
        for _, row in df.iterrows():
            corpus.append(f"{row['text_a']} {row['text_b']}")
    
    return corpus


def add_hard_temporal_negatives(
    data_dir: Path,
    all_bins: List[str],
    bucket_name: str,
    train_examples: List[InputExample],
    neg_k: int = 4,
    seed: int = 42,
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[InputExample]:
    """Add hard temporal negatives using FAST built-in mining.
    
    Uses sentence-transformers' mine_hard_negatives() with FAISS.
    Takes seconds instead of hours!
    
    Args:
        data_dir: Directory containing bucket data.
        all_bins: List of all bin names in chronological order.
        bucket_name: Current bucket name.
        train_examples: Original training examples.
        neg_k: Number of hard negatives per positive.
        seed: Random seed.
        base_model_name: Model to use for embedding similarity.
        
    Returns:
        Augmented training examples.
    """
    logger.info(f"Augmenting {len(train_examples)} examples with hard negatives...")
    logger.info(f"Current bin: {bucket_name}, neg_k: {neg_k}")
    
    # Get adjacent bins
    adjacent_bins = get_adjacent_bins(bucket_name, all_bins)
    
    if not adjacent_bins:
        logger.warning("No adjacent bins found, returning original examples")
        return train_examples
    
    logger.info(f"Sampling from adjacent bins: {adjacent_bins}")
    
    # Convert training examples to dataset format
    queries = []
    positives = []
    
    for example in train_examples:
        # Anchor is text_a (title), positive is text_b (abstract)
        queries.append(example.texts[0])
        positives.append(example.texts[1])
    
    # Create dataset
    dataset = Dataset.from_dict({
        "query": queries,
        "positive": positives,
    })
    
    # Load corpus from adjacent bins
    logger.info("Loading corpus from adjacent time bins...")
    corpus = load_adjacent_bin_corpus(data_dir, adjacent_bins)
    
    if not corpus:
        logger.warning("No corpus texts found in adjacent bins")
        return train_examples
    
    logger.info(f"Corpus size: {len(corpus):,} texts from adjacent bins")
    
    # Load model for mining
    logger.info(f"Loading model: {base_model_name}")
    model = SentenceTransformer(base_model_name, device="cuda")
    
    # Mine hard negatives using built-in function
    logger.info("Mining hard negatives with FAISS (fast!)...")
    
    try:
        mined_dataset = mine_hard_negatives(
            dataset=dataset,
            model=model,
            corpus=corpus,
            range_min=0,  # Don't skip any candidates
            range_max=100,  # Consider top 100 most similar
            num_negatives=neg_k,
            sampling_strategy="top",  # Take the hardest negatives
            use_faiss=True,  # Use FAISS for speed!
            batch_size=32,
        )
        
        logger.info(f"✓ Mined {len(mined_dataset)} examples with hard negatives")
        
    except Exception as e:
        logger.error(f"Hard negative mining failed: {e}")
        logger.warning("Falling back to original examples without hard negatives")
        return train_examples
    
    # Convert back to InputExamples
    augmented_examples = []
    
    for i, item in enumerate(mined_dataset):
        # Add positive example
        pos_example = InputExample(
            texts=[item["query"], item["positive"]],
            label=1.0,
            guid=f"{bucket_name}_pos_{i}",
        )
        augmented_examples.append(pos_example)
        
        # Add negative examples
        if "negative" in item:
            # Single negative
            negatives = [item["negative"]]
        elif "negatives" in item:
            # Multiple negatives
            negatives = item["negatives"]
        else:
            negatives = []
        
        for neg_idx, neg_text in enumerate(negatives):
            neg_example = InputExample(
                texts=[item["query"], neg_text],
                label=0.0,
                guid=f"{bucket_name}_neg_{i}_{neg_idx}",
            )
            augmented_examples.append(neg_example)
    
    ratio = len(augmented_examples) / len(train_examples)
    logger.info(
        f"✓ Augmented {len(train_examples)} → {len(augmented_examples)} examples "
        f"(ratio: {ratio:.1f}x)"
    )
    
    return augmented_examples
