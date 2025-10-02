"""Text preprocessing utilities for cleaning and deduplication."""

import hashlib
import re
from typing import Dict, Any

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def clean_html(text: str) -> str:
    """Remove HTML tags from text.
    
    Args:
        text: Input text potentially containing HTML.
        
    Returns:
        Text with HTML tags removed.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters to single space.
    
    Args:
        text: Input text.
        
    Returns:
        Text with normalized whitespace.
    """
    # Replace multiple whitespace (including newlines) with single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to maximum number of tokens (simple whitespace split).
    
    Args:
        text: Input text.
        max_tokens: Maximum number of tokens to keep.
        
    Returns:
        Truncated text.
    """
    tokens = text.split()
    if len(tokens) > max_tokens:
        return " ".join(tokens[:max_tokens])
    return text


def normalize_text(text: str) -> str:
    """Normalize text for deduplication hashing.
    
    Args:
        text: Input text.
        
    Returns:
        Normalized text (lowercase, no extra spaces, no punctuation).
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = collapse_whitespace(text)
    return text


def compute_content_hash(title: str, abstract: str) -> str:
    """Compute hash of normalized title + abstract for deduplication.
    
    Args:
        title: Paper title.
        abstract: Paper abstract.
        
    Returns:
        MD5 hash of normalized content.
    """
    # Combine title and abstract
    content = f"{title} {abstract}"
    # Normalize
    normalized = normalize_text(content)
    # Hash
    return hashlib.md5(normalized.encode()).hexdigest()


def clean_and_preprocess(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Clean, deduplicate, and preprocess text data.
    
    Args:
        df: DataFrame with title and abstract columns.
        config: Preprocessing config dictionary.
        
    Returns:
        Cleaned DataFrame with duplicates removed.
    """
    initial_count = len(df)
    df = df.copy()
    
    logger.info(f"Starting preprocessing with {initial_count} samples")
    
    # Clean HTML if enabled
    if config.get("clean_html", True):
        logger.info("Removing HTML tags...")
        df["title"] = df["title"].fillna("").astype(str).apply(clean_html)
        df["abstract"] = df["abstract"].fillna("").astype(str).apply(clean_html)
    
    # Collapse whitespace if enabled
    if config.get("collapse_whitespace", True):
        logger.info("Collapsing whitespace...")
        df["title"] = df["title"].apply(collapse_whitespace)
        df["abstract"] = df["abstract"].apply(collapse_whitespace)
    
    # Truncate to max tokens if specified
    max_tokens = config.get("max_tokens")
    if max_tokens:
        logger.info(f"Truncating abstracts to {max_tokens} tokens...")
        df["abstract"] = df["abstract"].apply(lambda x: truncate_tokens(x, max_tokens))
    
    # Filter by length constraints
    min_len = config.get("min_abstract_length", 50)
    max_len = config.get("max_abstract_length", 2000)
    
    logger.info(f"Filtering by length ({min_len}-{max_len} chars)...")
    df["abstract_len"] = df["abstract"].str.len()
    df = df[(df["abstract_len"] >= min_len) & (df["abstract_len"] <= max_len)]
    df = df.drop(columns=["abstract_len"])
    
    logger.info(f"After length filtering: {len(df)} samples (removed {initial_count - len(df)})")
    
    # Remove duplicates if enabled
    if config.get("remove_duplicates", True):
        logger.info("Computing content hashes for deduplication...")
        df["content_hash"] = df.apply(
            lambda row: compute_content_hash(row["title"], row["abstract"]),
            axis=1
        )
        
        # Drop duplicates by hash
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["content_hash"], keep="first")
        after_dedup = len(df)
        
        logger.info(
            f"Removed {before_dedup - after_dedup} near-duplicates "
            f"({after_dedup} samples remaining)"
        )
        
        # Drop hash column (no longer needed)
        df = df.drop(columns=["content_hash"])
    
    # Final summary
    total_removed = initial_count - len(df)
    pct_removed = (total_removed / initial_count) * 100 if initial_count > 0 else 0
    
    logger.info(
        f"Preprocessing complete: {len(df)} samples "
        f"(removed {total_removed}, {pct_removed:.1f}%)"
    )
    
    return df
