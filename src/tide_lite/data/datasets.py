"""Dataset loaders for TIDE-Lite experiments.

This module provides loaders for real datasets:
- STS-B for semantic textual similarity
- Quora duplicate questions for retrieval
- TimeQA/TempLAMA for temporal reasoning
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


def load_stsb(cfg: Dict) -> DatasetDict:
    """Load STS-B dataset from GLUE benchmark.
    
    Args:
        cfg: Configuration dictionary with:
            - cache_dir: Directory for caching datasets
            - max_samples: Optional max samples per split
            - seed: Random seed for sampling
        
    Returns:
        DatasetDict with train/validation/test splits containing:
            - sentence1: First sentence
            - sentence2: Second sentence  
            - label: Similarity score [0, 5]
    """
    logger.info("Loading STS-B dataset from GLUE")
    
    cache_dir = cfg.get("cache_dir", "./data")
    max_samples = cfg.get("max_samples", None)
    seed = cfg.get("seed", 42)
    
    # Load from HuggingFace datasets
    dataset = load_dataset(
        "glue",
        "stsb",
        cache_dir=cache_dir
    )
    
    # Apply sample limit if specified
    if max_samples:
        for split in ["train", "validation", "test"]:
            n_samples = len(dataset[split])
            if max_samples < n_samples:
                # Use seed for reproducible sampling
                dataset[split] = dataset[split].shuffle(seed=seed).select(range(max_samples))
                logger.info(f"Limited {split} split to {max_samples} samples")
            
    # Log dataset sizes
    for split in ["train", "validation", "test"]:
        logger.info(f"STS-B {split}: {len(dataset[split])} samples")
    
    return dataset


def load_quora(cfg: Dict) -> Tuple[Dataset, Dataset, Dataset]:
    """Load Quora duplicate questions for retrieval evaluation.
    
    Creates a retrieval setup where:
    - Corpus: All unique questions
    - Queries: Questions from duplicate pairs
    - Relevance: Duplicate questions are relevant (score=1)
    
    Args:
        cfg: Configuration dictionary with:
            - cache_dir: Directory for caching
            - max_samples: Optional max number of question pairs
            - seed: Random seed for sampling
        
    Returns:
        Tuple of (corpus, queries, qrels):
            - corpus: Dataset with 'text' and 'doc_id'
            - queries: Dataset with 'text' and 'query_id'  
            - qrels: Dataset with 'query_id', 'doc_id', and 'relevance'
    """
    logger.info("Loading Quora duplicate questions dataset")
    
    cache_dir = cfg.get("cache_dir", "./data")
    max_samples = cfg.get("max_samples", None)
    seed = cfg.get("seed", 42)
    
    # Load dataset
    dataset = load_dataset(
        "quora",
        cache_dir=cache_dir,
        split="train"
    )
    
    # Filter out None questions
    dataset = dataset.filter(
        lambda x: x["questions"]["text"][0] is not None 
        and x["questions"]["text"][1] is not None
    )
    
    # Apply sample limit
    if max_samples:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))
    
    # Build corpus from all unique questions
    question_to_id = {}
    corpus_texts = []
    
    for row in dataset:
        q1, q2 = row["questions"]["text"]
        
        if q1 not in question_to_id:
            question_to_id[q1] = len(corpus_texts)
            corpus_texts.append(q1)
        
        if q2 not in question_to_id:
            question_to_id[q2] = len(corpus_texts)
            corpus_texts.append(q2)
    
    corpus = Dataset.from_dict({
        "text": corpus_texts,
        "doc_id": list(range(len(corpus_texts))),
    })
    
    # Create queries and relevance judgments
    queries_list = []
    qrels_list = []
    query_id = 0
    
    for row in dataset:
        if not row["is_duplicate"]:
            continue
            
        q1, q2 = row["questions"]["text"]
        q1_id = question_to_id[q1]
        q2_id = question_to_id[q2]
        
        # Use q1 as query, q2 as relevant doc
        queries_list.append({
            "text": q1,
            "query_id": query_id
        })
        qrels_list.append({
            "query_id": query_id,
            "doc_id": q2_id,
            "relevance": 1.0
        })
        query_id += 1
        
        # Also use q2 as query, q1 as relevant doc (bidirectional)
        queries_list.append({
            "text": q2,
            "query_id": query_id
        })
        qrels_list.append({
            "query_id": query_id,
            "doc_id": q1_id,
            "relevance": 1.0
        })
        query_id += 1
    
    queries = Dataset.from_pandas(pd.DataFrame(queries_list))
    qrels = Dataset.from_pandas(pd.DataFrame(qrels_list))
    
    logger.info(f"Created Quora retrieval dataset:")
    logger.info(f"  - Corpus: {len(corpus)} documents")
    logger.info(f"  - Queries: {len(queries)} queries")
    logger.info(f"  - Relevant pairs: {len(qrels)} judgments")
    
    return corpus, queries, qrels


def load_timeqa(cfg: Dict) -> Dataset:
    """Load TimeQA or TempLAMA dataset for temporal reasoning.
    
    First attempts to load TimeQA from specified directory.
    Falls back to TempLAMA if TimeQA is not available.
    
    Args:
        cfg: Configuration dictionary with:
            - timeqa_data_dir: Path to TimeQA dataset
            - templama_path: Fallback path to TempLAMA
            - cache_dir: General cache directory
            - max_samples: Optional sample limit
            - seed: Random seed
        
    Returns:
        Dataset with temporal QA examples:
            - question: Question text
            - context: Context paragraph  
            - answer: Answer text
            - timestamp: Unix timestamp (if available)
    """
    logger.info("Loading temporal QA dataset")
    
    timeqa_dir = cfg.get("timeqa_data_dir", "./data/timeqa")
    templama_dir = cfg.get("templama_path", "./data/templama")
    cache_dir = cfg.get("cache_dir", "./data")
    max_samples = cfg.get("max_samples", None)
    seed = cfg.get("seed", 42)
    
    # Try loading TimeQA first
    timeqa_path = Path(timeqa_dir)
    if timeqa_path.exists() and (timeqa_path / "train.json").exists():
        logger.info(f"Loading TimeQA from {timeqa_path}")
        
        # Load TimeQA format
        train_file = timeqa_path / "train.json"
        with open(train_file, 'r') as f:
            data = json.load(f)
        
        # Convert to our format
        examples = []
        for item in data:
            # TimeQA format typically has:
            # - question, context, answer, timestamps
            examples.append({
                "question": item.get("question", ""),
                "context": item.get("context", ""),
                "answer": item.get("answer", item.get("answers", [""])[0] if isinstance(item.get("answers"), list) else ""),
                "timestamp": item.get("timestamp", 0.0),
            })
        
        dataset = Dataset.from_pandas(pd.DataFrame(examples))
        
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.shuffle(seed=seed).select(range(max_samples))
        
        logger.info(f"Loaded {len(dataset)} TimeQA examples")
        return dataset
    
    # Fallback to TempLAMA
    logger.info("TimeQA not found, falling back to TempLAMA")
    logger.info("Note: TempLAMA provides temporal facts but not full QA pairs.")
    logger.info("For production use, please download TimeQA dataset.")
    
    templama_path = Path(templama_dir)
    
    # Try to load TempLAMA
    if templama_path.exists():
        # TempLAMA typically has temporal facts in JSON format
        train_file = templama_path / "train.jsonl"
        if not train_file.exists():
            train_file = templama_path / "data.jsonl"
        
        if train_file.exists():
            logger.info(f"Loading TempLAMA from {train_file}")
            
            examples = []
            with open(train_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    # Convert TempLAMA format to our QA format
                    # TempLAMA typically has: subject, relation, object, time
                    question = item.get("question", f"What is {item.get('relation', 'related to')} {item.get('subject', '')}?")
                    answer = item.get("object", item.get("answer", ""))
                    context = item.get("context", f"{item.get('subject', '')} {item.get('relation', '')} {answer}")
                    
                    examples.append({
                        "question": question,
                        "context": context,
                        "answer": answer,
                        "timestamp": item.get("time", 0.0) if isinstance(item.get("time"), (int, float)) else 0.0,
                    })
            
            dataset = Dataset.from_pandas(pd.DataFrame(examples))
            
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.shuffle(seed=seed).select(range(max_samples))
            
            logger.info(f"Loaded {len(dataset)} TempLAMA examples")
            return dataset
    
    # If neither dataset is available, raise an error with clear instructions
    error_msg = "\n" + "=" * 60 + "\n"
    error_msg += "Neither TimeQA nor TempLAMA datasets found!\n\n"
    error_msg += "To use TimeQA:\n"
    error_msg += "1. Download TimeQA dataset from official source\n"
    error_msg += f"2. Extract to: {timeqa_dir}\n"
    error_msg += "3. Ensure train.json exists in that directory\n"
    error_msg += f"4. Set timeqa_data_dir in configs/defaults.yaml\n\n"
    error_msg += "To use TempLAMA (simpler alternative):\n"
    error_msg += "1. Download TempLAMA from:\n"
    error_msg += "   https://github.com/google-research/language/tree/master/language/templama\n"
    error_msg += f"2. Place data files in: {templama_dir}\n"
    error_msg += f"3. Set templama_path in configs/defaults.yaml\n\n"
    error_msg += "To skip temporal evaluation:\n"
    error_msg += "  Use --skip-temporal flag when running evaluation\n"
    error_msg += "=" * 60
    
    raise FileNotFoundError(error_msg)


def verify_dataset_integrity(dataset) -> bool:
    """Verify dataset has required fields and valid data.
    
    Args:
        dataset: Dataset to verify (Dataset or DatasetDict)
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If dataset is invalid
    """
    if isinstance(dataset, DatasetDict):
        for split_name, split_data in dataset.items():
            if len(split_data) == 0:
                raise ValueError(f"Empty dataset split: {split_name}")
            logger.debug(f"Verified {split_name} split with {len(split_data)} samples")
    elif isinstance(dataset, tuple):
        # For retrieval datasets (corpus, queries, qrels)
        for i, component in enumerate(dataset):
            if len(component) == 0:
                raise ValueError(f"Empty dataset component {i}")
        logger.debug(f"Verified retrieval dataset with {len(dataset[0])} docs")
    else:
        if len(dataset) == 0:
            raise ValueError("Empty dataset")
        logger.debug(f"Verified dataset with {len(dataset)} samples")
    
    return True
