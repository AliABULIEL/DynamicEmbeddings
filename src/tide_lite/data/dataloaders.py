"""DataLoader utilities for TIDE-Lite experiments.

This module provides utilities for creating DataLoaders with proper
collation and batching for real datasets.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict

from .collate import create_collator, TextBatcher
from .datasets import load_stsb, load_quora, load_timeqa

logger = logging.getLogger(__name__)


def create_stsb_dataloaders(
    cfg: Dict,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    max_seq_length: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create STS-B dataloaders for similarity learning.
    
    Args:
        cfg: Configuration dictionary.
        batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        max_seq_length: Maximum sequence length.
        num_workers: Number of dataloader workers.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    logger.info("Creating STS-B dataloaders")
    
    # Load dataset
    datasets = load_stsb(cfg)
    
    # Get encoder name from config
    encoder_name = cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Create collator
    collator = create_collator(
        "stsb",
        tokenizer=encoder_name,
        max_length=max_seq_length,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    val_loader = DataLoader(
        datasets["validation"],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    test_loader = DataLoader(
        datasets["test"],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    logger.info(
        f"Created STS-B dataloaders - Train: {len(train_loader)} batches, "
        f"Val: {len(val_loader)} batches, Test: {len(test_loader)} batches"
    )
    
    return train_loader, val_loader, test_loader


def create_temporal_dataloaders(
    cfg: Dict,
    batch_size: int = 32,
    max_seq_length: int = 128,
    num_workers: int = 2,
    skip_if_missing: bool = False,
) -> Optional[Tuple[DataLoader, DataLoader, DataLoader]]:
    """Create temporal (TimeQA/TempLAMA) dataloaders for temporal consistency.
    
    Args:
        cfg: Configuration dictionary.
        batch_size: Training batch size.
        max_seq_length: Maximum sequence length.
        num_workers: Number of dataloader workers.
        skip_if_missing: Return None if dataset is missing instead of raising error.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader) or None if skipped.
    """
    logger.info("Creating temporal dataloaders")
    
    # Check if we should skip temporal
    if cfg.get("skip_temporal", False):
        logger.info("Skipping temporal dataloaders (skip_temporal=True)")
        return None
    
    try:
        # Load temporal dataset
        dataset = load_timeqa(cfg)
        
        # Split into train/val/test (80/10/10)
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        # Use dataset.train_test_split for splitting
        splits = dataset.train_test_split(
            test_size=(val_size + test_size) / total_size,
            seed=cfg.get("seed", 42)
        )
        train_dataset = splits["train"]
        
        # Split remaining into val and test
        remaining = splits["test"]
        val_test_splits = remaining.train_test_split(
            test_size=test_size / (val_size + test_size),
            seed=cfg.get("seed", 42)
        )
        val_dataset = val_test_splits["train"]
        test_dataset = val_test_splits["test"]
        
        # Get encoder name
        encoder_name = cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        
        # Create collator for temporal data
        # This collator should preserve timestamps from the dataset
        collator = create_collator(
            "temporal",
            tokenizer=encoder_name,
            max_length=max_seq_length,
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        
        logger.info(
            f"Created temporal dataloaders - Train: {len(train_loader)} batches, "
            f"Val: {len(val_loader)} batches, Test: {len(test_loader)} batches"
        )
        
        return train_loader, val_loader, test_loader
        
    except FileNotFoundError as e:
        if skip_if_missing:
            logger.warning(f"Temporal dataset not found: {e}")
            logger.warning("Training without temporal consistency loss")
            return None
        else:
            raise


def create_quora_dataloaders(
    cfg: Dict,
    batch_size: int = 128,
    max_seq_length: int = 128,
    num_workers: int = 2,
    max_corpus_size: Optional[int] = None,
    max_queries: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Dataset]:
    """Create Quora retrieval dataloaders.
    
    Args:
        cfg: Configuration dictionary.
        batch_size: Batch size for encoding.
        max_seq_length: Maximum sequence length.
        num_workers: Number of dataloader workers.
        max_corpus_size: Maximum corpus size (None for all).
        max_queries: Maximum number of queries (None for all).
        
    Returns:
        Tuple of (corpus_loader, query_loader, qrels).
    """
    logger.info("Creating Quora retrieval dataloaders")
    
    # Load dataset
    corpus, queries, qrels = load_quora(cfg)
    
    # Limit sizes if specified
    if max_corpus_size and len(corpus) > max_corpus_size:
        corpus = corpus.select(range(max_corpus_size))
        logger.info(f"Limited corpus to {max_corpus_size} documents")
    
    if max_queries and len(queries) > max_queries:
        # Select queries and filter qrels accordingly
        selected_queries = queries.select(range(max_queries))
        selected_query_ids = set(selected_queries["query_id"])
        
        # Filter qrels to only include selected queries
        filtered_qrels = qrels.filter(
            lambda x: x["query_id"] in selected_query_ids
        )
        
        queries = selected_queries
        qrels = filtered_qrels
        logger.info(f"Limited to {max_queries} queries with {len(qrels)} relevance judgments")
    
    # Get encoder name from config
    encoder_name = cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Create collators
    corpus_collator = create_collator(
        "retrieval_corpus",
        tokenizer=encoder_name,
        max_length=max_seq_length,
    )
    query_collator = create_collator(
        "retrieval_query",
        tokenizer=encoder_name,
        max_length=max_seq_length,
    )
    
    # Create dataloaders
    corpus_loader = DataLoader(
        corpus,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=corpus_collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    query_loader = DataLoader(
        queries,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=query_collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    logger.info(
        f"Created retrieval dataloaders - Corpus: {len(corpus_loader)} batches, "
        f"Queries: {len(query_loader)} batches, Qrels: {len(qrels)} judgments"
    )
    
    # Check FAISS availability
    try:
        import faiss
        if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
            logger.info("FAISS GPU support detected")
        else:
            logger.info("Using FAISS CPU")
    except ImportError:
        logger.warning("FAISS not installed - retrieval evaluation may be limited")
    
    return corpus_loader, query_loader, qrels


def create_temporal_qa_dataloader(
    cfg: Dict,
    batch_size: int = 32,
    max_context_length: int = 384,
    max_question_length: int = 64,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    """Create dataloader for temporal QA dataset.
    
    Args:
        cfg: Configuration dictionary.
        batch_size: Batch size.
        max_context_length: Maximum context length.
        max_question_length: Maximum question length.
        num_workers: Number of dataloader workers.
        shuffle: Whether to shuffle data.
        
    Returns:
        DataLoader for temporal QA dataset.
    """
    logger.info("Creating temporal QA dataloader")
    
    # Load dataset (TimeQA or TempLAMA)
    dataset = load_timeqa(cfg)
    
    # Get encoder name from config
    encoder_name = cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Create collator
    collator = create_collator(
        "temporal_qa",
        tokenizer=encoder_name,
        max_context_length=max_context_length,
        max_question_length=max_question_length,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    logger.info(
        f"Created temporal QA dataloader with {len(dataloader)} batches "
        f"({len(dataset)} examples)"
    )
    
    return dataloader


def create_all_dataloaders(
    cfg: Dict,
    tasks: Optional[list] = None,
) -> Dict[str, any]:
    """Create all requested dataloaders.
    
    Args:
        cfg: Configuration dictionary.
        tasks: List of tasks to create dataloaders for.
               If None, creates all available.
        
    Returns:
        Dictionary mapping task names to dataloaders.
    """
    if tasks is None:
        tasks = ["stsb", "quora", "temporal"]
    
    dataloaders = {}
    
    if "stsb" in tasks:
        train_loader, val_loader, test_loader = create_stsb_dataloaders(
            cfg,
            batch_size=cfg.get("batch_size", 32),
            eval_batch_size=cfg.get("eval_batch_size", 64),
            max_seq_length=cfg.get("max_seq_len", 128),
            num_workers=cfg.get("num_workers", 2),
        )
        dataloaders["stsb"] = {
            "train": train_loader,
            "validation": val_loader,
            "test": test_loader,
        }
    
    if "quora" in tasks:
        corpus_loader, query_loader, qrels = create_quora_dataloaders(
            cfg,
            batch_size=cfg.get("eval_batch_size", 128),
            max_seq_length=cfg.get("max_seq_len", 128),
            num_workers=cfg.get("num_workers", 2),
            max_corpus_size=cfg.get("max_corpus_size", None),
            max_queries=cfg.get("max_queries", None),
        )
        dataloaders["quora"] = {
            "corpus": corpus_loader,
            "queries": query_loader,
            "qrels": qrels,
        }
    
    if "temporal" in tasks or "timeqa" in tasks or "templama" in tasks:
        temporal_loader = create_temporal_qa_dataloader(
            cfg,
            batch_size=cfg.get("batch_size", 32),
            max_context_length=cfg.get("max_context_length", 384),
            max_question_length=cfg.get("max_question_length", 64),
            num_workers=cfg.get("num_workers", 2),
            shuffle=True,
        )
        dataloaders["temporal"] = temporal_loader
    
    logger.info(f"Created dataloaders for tasks: {list(dataloaders.keys())}")
    
    return dataloaders
