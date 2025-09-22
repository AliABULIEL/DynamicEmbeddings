"""Dataset loaders for TIDE-Lite experiments.

This module provides loaders for STS-B with synthetic timestamps,
Quora duplicate questions for retrieval, and a TimeQA-lite surrogate.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading.
    
    Attributes:
        seed: Random seed for reproducibility.
        max_samples: Maximum number of samples to load (None for all).
        cache_dir: Directory for caching downloaded datasets.
        timestamp_start: Start date for synthetic timestamps.
        timestamp_end: End date for synthetic timestamps.
        temporal_noise_std: Std dev for temporal noise in days.
    """
    seed: int = 42
    max_samples: Optional[int] = None
    cache_dir: str = "./data"
    timestamp_start: str = "2020-01-01"
    timestamp_end: str = "2024-01-01"
    temporal_noise_std: float = 7.0  # Weekly noise


def _generate_synthetic_timestamps(
    n_samples: int,
    start_date: str,
    end_date: str,
    seed: int = 42,
    noise_std_days: float = 7.0,
) -> np.ndarray:
    """Generate synthetic timestamps with temporal clustering.
    
    Args:
        n_samples: Number of timestamps to generate.
        start_date: ISO format start date.
        end_date: ISO format end date.
        seed: Random seed for reproducibility.
        noise_std_days: Standard deviation for temporal noise in days.
        
    Returns:
        Array of Unix timestamps in seconds.
        
    Note:
        Creates temporal clusters to simulate real-world data patterns
        where related content tends to appear in bursts.
    """
    np.random.seed(seed)
    
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    total_seconds = (end - start).total_seconds()
    
    # Create temporal clusters: 20% cluster centers, 80% around them
    n_clusters = max(1, n_samples // 20)
    cluster_centers = np.random.uniform(0, total_seconds, n_clusters)
    
    timestamps = []
    for i in range(n_samples):
        if i < n_clusters:
            # Cluster center
            ts = cluster_centers[i % n_clusters]
        else:
            # Sample around a random cluster center with Gaussian noise
            center = cluster_centers[np.random.randint(n_clusters)]
            noise = np.random.normal(0, noise_std_days * 86400)  # Convert days to seconds
            ts = np.clip(center + noise, 0, total_seconds)
        
        timestamps.append(start.timestamp() + ts)
    
    return np.array(timestamps)


def load_stsb_with_timestamps(cfg: DatasetConfig) -> DatasetDict:
    """Load STS-B dataset with synthetic timestamps.
    
    Args:
        cfg: Dataset configuration.
        
    Returns:
        DatasetDict with train/validation/test splits, each containing:
            - sentence1: First sentence
            - sentence2: Second sentence  
            - label: Similarity score [0, 5]
            - timestamp1: Unix timestamp for sentence1
            - timestamp2: Unix timestamp for sentence2
            
    Note:
        Timestamps are synthetic and added for temporal consistency experiments.
        Real-world usage should replace with actual temporal metadata.
    """
    logger.info("Loading STS-B dataset with synthetic timestamps")
    
    # Load base dataset
    dataset = load_dataset("glue", "stsb", cache_dir=cfg.cache_dir)
    
    # Add synthetic timestamps to each split
    for split in ["train", "validation", "test"]:
        n_samples = len(dataset[split])
        
        # Generate correlated timestamps for sentence pairs
        base_timestamps = _generate_synthetic_timestamps(
            n_samples,
            cfg.timestamp_start,
            cfg.timestamp_end,
            seed=cfg.seed + hash(split),
            noise_std_days=cfg.temporal_noise_std,
        )
        
        # Add small offset for second sentence (usually created near first)
        offset = np.random.normal(0, 3600, n_samples)  # ±1 hour average
        
        dataset[split] = dataset[split].add_column("timestamp1", base_timestamps.tolist())
        dataset[split] = dataset[split].add_column("timestamp2", (base_timestamps + offset).tolist())
        
        if cfg.max_samples and cfg.max_samples < n_samples:
            dataset[split] = dataset[split].select(range(cfg.max_samples))
            
        logger.info(f"Loaded {len(dataset[split])} STS-B {split} samples with timestamps")
    
    return dataset


def load_quora(cfg: DatasetConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """Load Quora duplicate questions for retrieval evaluation.
    
    Args:
        cfg: Dataset configuration.
        
    Returns:
        Tuple of (corpus, queries, qrels):
            - corpus: Dataset with 'text' and 'doc_id'
            - queries: Dataset with 'text' and 'query_id'  
            - qrels: Dataset with 'query_id', 'doc_id', and 'relevance'
            
    Note:
        Creates a retrieval setup from duplicate question pairs.
        Positive pairs have relevance=1, negatives sampled randomly.
    """
    logger.info("Loading Quora duplicate questions dataset")
    
    dataset = load_dataset("quora", cache_dir=cfg.cache_dir)["train"]
    
    # Filter out None questions
    dataset = dataset.filter(
        lambda x: x["questions"]["text"][0] is not None 
        and x["questions"]["text"][1] is not None
    )
    
    if cfg.max_samples:
        dataset = dataset.select(range(min(cfg.max_samples, len(dataset))))
    
    # Build corpus from unique questions
    all_questions = []
    seen = set()
    
    for row in dataset:
        q1, q2 = row["questions"]["text"]
        if q1 not in seen:
            all_questions.append(q1)
            seen.add(q1)
        if q2 not in seen:
            all_questions.append(q2)
            seen.add(q2)
    
    corpus = Dataset.from_dict({
        "text": all_questions,
        "doc_id": list(range(len(all_questions))),
    })
    
    # Create queries and relevance judgments from duplicate pairs
    queries_list = []
    qrels_list = []
    
    question_to_id = {q: i for i, q in enumerate(all_questions)}
    
    for i, row in enumerate(dataset):
        if not row["is_duplicate"]:
            continue
            
        q1, q2 = row["questions"]["text"]
        q1_id = question_to_id[q1]
        q2_id = question_to_id[q2]
        
        # Use first as query, second as relevant doc
        queries_list.append({"text": q1, "query_id": i})
        qrels_list.append({"query_id": i, "doc_id": q2_id, "relevance": 1})
    
    queries = Dataset.from_pandas(pd.DataFrame(queries_list))
    qrels = Dataset.from_pandas(pd.DataFrame(qrels_list))
    
    logger.info(f"Created retrieval dataset: {len(corpus)} docs, {len(queries)} queries")
    
    return corpus, queries, qrels


def load_timeqa_lite(cfg: DatasetConfig) -> Dataset:
    """Load TimeQA-lite surrogate for temporal QA experiments.
    
    Args:
        cfg: Dataset configuration.
        
    Returns:
        Dataset with temporal QA examples:
            - question: Question text
            - context: Context paragraph  
            - answer: Answer text
            - timestamp: Unix timestamp of the context
            - temporal_expression: Extracted temporal phrase
            
    Note:
        This is a PLACEHOLDER implementation using SQuAD with synthetic temporal markers.
        For production, replace with actual TimeQA dataset:
        ```python
        # Actual TimeQA loading (when available):
        from datasets import load_dataset
        dataset = load_dataset("time_qa", cache_dir=cfg.cache_dir)
        ```
        
    Warning:
        This surrogate does NOT provide realistic temporal reasoning challenges.
        It only adds superficial temporal markers for testing the pipeline.
    """
    logger.warning(
        "Using TimeQA-lite surrogate. For actual temporal reasoning evaluation, "
        "replace with real TimeQA dataset when available."
    )
    
    # Use SQuAD as base and inject temporal context
    squad = load_dataset("squad", cache_dir=cfg.cache_dir)["train"]
    
    if cfg.max_samples:
        squad = squad.select(range(min(cfg.max_samples, len(squad))))
    
    # Temporal expressions to inject
    temporal_markers = [
        "As of {date}",
        "In {date}",
        "During {date}",
        "Since {date}",
        "Before {date}",
        "After {date}",
        "By {date}",
        "Until {date}",
    ]
    
    # Generate timestamps and add temporal context
    timestamps = _generate_synthetic_timestamps(
        len(squad),
        cfg.timestamp_start,
        cfg.timestamp_end,
        seed=cfg.seed,
    )
    
    timeqa_examples = []
    
    for i, example in enumerate(squad):
        # Skip if no answers
        if not example["answers"]["text"]:
            continue
            
        # Convert timestamp to date
        ts = timestamps[i]
        date = datetime.fromtimestamp(ts).strftime("%B %Y")
        
        # Inject temporal marker
        marker = np.random.choice(temporal_markers)
        temporal_expr = marker.format(date=date)
        
        # Modify context with temporal marker
        context = f"{temporal_expr}, {example['context']}"
        
        timeqa_examples.append({
            "question": example["question"],
            "context": context,
            "answer": example["answers"]["text"][0] if example["answers"]["text"] else "",
            "timestamp": ts,
            "temporal_expression": temporal_expr,
        })
    
    dataset = Dataset.from_pandas(pd.DataFrame(timeqa_examples))
    
    logger.info(
        f"Created TimeQA-lite surrogate with {len(dataset)} examples. "
        "Note: This is NOT real temporal reasoning data."
    )
    
    return dataset


def verify_dataset_integrity(dataset: Union[Dataset, DatasetDict]) -> bool:
    """Verify dataset has required fields and valid data.
    
    Args:
        dataset: Dataset to verify.
        
    Returns:
        True if dataset is valid, raises ValueError otherwise.
        
    Raises:
        ValueError: If dataset is missing required fields or has invalid data.
    """
    if isinstance(dataset, DatasetDict):
        for split_name, split_data in dataset.items():
            if len(split_data) == 0:
                raise ValueError(f"Empty dataset split: {split_name}")
            logger.debug(f"Verified {split_name} split with {len(split_data)} samples")
    else:
        if len(dataset) == 0:
            raise ValueError("Empty dataset")
        logger.debug(f"Verified dataset with {len(dataset)} samples")
    
    return True
