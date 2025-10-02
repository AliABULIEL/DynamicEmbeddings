"""Positive pair generation for contrastive learning."""

from typing import List, Tuple

import pandas as pd


def create_pairs(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """
    Create positive (title, abstract) pairs from dataset.

    Each paper yields one positive pair: (paper_id, title, abstract).
    Negatives are handled in-batch during training.

    Args:
        df: DataFrame with columns [paper_id, title, abstract, year]

    Returns:
        List of (paper_id, title, abstract) tuples

    Raises:
        ValueError: If required columns are missing
    """
    required_cols = {"paper_id", "title", "abstract"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Filter out rows with null title or abstract
    valid_df = df.dropna(subset=["title", "abstract"])

    if len(valid_df) < len(df):
        n_dropped = len(df) - len(valid_df)
        print(
            f"Warning: Dropped {n_dropped} rows with null title or abstract "
            f"({n_dropped/len(df)*100:.1f}%)"
        )

    pairs = [
        (str(row["paper_id"]), str(row["title"]), str(row["abstract"]))
        for _, row in valid_df.iterrows()
    ]

    return pairs


def pairs_to_dataframe(pairs: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """
    Convert pairs list to DataFrame.

    Args:
        pairs: List of (paper_id, title, abstract) tuples

    Returns:
        DataFrame with columns [paper_id, title, abstract]
    """
    return pd.DataFrame(pairs, columns=["paper_id", "title", "abstract"])


def get_pair_statistics(pairs: List[Tuple[str, str, str]]) -> dict:
    """
    Compute statistics about pairs.

    Args:
        pairs: List of (paper_id, title, abstract) tuples

    Returns:
        Dict with statistics (count, avg lengths, etc.)
    """
    if not pairs:
        return {
            "n_pairs": 0,
            "avg_title_length": 0,
            "avg_abstract_length": 0,
            "max_title_length": 0,
            "max_abstract_length": 0,
        }

    titles = [pair[1] for pair in pairs]
    abstracts = [pair[2] for pair in pairs]

    return {
        "n_pairs": len(pairs),
        "avg_title_length": sum(len(t) for t in titles) / len(titles),
        "avg_abstract_length": sum(len(a) for a in abstracts) / len(abstracts),
        "max_title_length": max(len(t) for t in titles),
        "max_abstract_length": max(len(a) for a in abstracts),
    }
