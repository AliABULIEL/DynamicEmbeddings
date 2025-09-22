"""Collation utilities for batching text data.

This module provides collators for different task types:
- STS-B similarity pairs with timestamps
- Retrieval corpus/query batching  
- Temporal QA examples
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BatchEncoding

logger = logging.getLogger(__name__)


@dataclass 
class TextBatcher:
    """Tokenizer and batch collator for text inputs.
    
    Attributes:
        model_name: HuggingFace model name for tokenizer.
        max_length: Maximum sequence length for truncation.
        padding: Padding strategy ('max_length', 'longest', or False).
        truncation: Whether to truncate sequences.
    """
    
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 128
    padding: Union[bool, str] = "max_length"
    truncation: bool = True
    
    def __post_init__(self) -> None:
        """Initialize tokenizer after dataclass init."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info(f"Initialized tokenizer for {self.model_name}")
    
    def __call__(
        self, 
        texts: Union[str, List[str]],
        return_tensors: str = "pt",
    ) -> BatchEncoding:
        """Tokenize and batch text inputs.
        
        Args:
            texts: Single text or list of texts to tokenize.
            return_tensors: Format for returned tensors ('pt' for PyTorch).
            
        Returns:
            BatchEncoding with input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )
    
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs back to text.
        
        Args:
            token_ids: Tensor of token IDs.
            
        Returns:
            List of decoded text strings.
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


class STSBCollator:
    """Collator for STS-B sentence pairs with temporal metadata.
    
    This collator handles sentence pairs with similarity scores and timestamps,
    preparing them for temporal consistency training.
    """
    
    def __init__(
        self,
        tokenizer: TextBatcher,
        include_timestamps: bool = True,
    ) -> None:
        """Initialize STS-B collator.
        
        Args:
            tokenizer: TextBatcher instance for tokenization.
            include_timestamps: Whether to include temporal information.
        """
        self.tokenizer = tokenizer
        self.include_timestamps = include_timestamps
        logger.info(f"Initialized STS-B collator (timestamps={include_timestamps})")
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of STS-B examples.
        
        Args:
            batch: List of examples with sentence1, sentence2, label, timestamps.
            
        Returns:
            Dictionary with:
                - sentence1_inputs: Tokenized first sentences
                - sentence2_inputs: Tokenized second sentences  
                - labels: Similarity scores as tensor
                - timestamps1: Unix timestamps for sentence1 (if enabled)
                - timestamps2: Unix timestamps for sentence2 (if enabled)
        """
        sentences1 = [item["sentence1"] for item in batch]
        sentences2 = [item["sentence2"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        
        # Tokenize sentences
        sent1_encoding = self.tokenizer(sentences1, return_tensors="pt")
        sent2_encoding = self.tokenizer(sentences2, return_tensors="pt")
        
        collated = {
            "sentence1_inputs": sent1_encoding,
            "sentence2_inputs": sent2_encoding,
            "labels": labels,
        }
        
        if self.include_timestamps:
            collated["timestamps1"] = torch.tensor(
                [item["timestamp1"] for item in batch], 
                dtype=torch.float64
            )
            collated["timestamps2"] = torch.tensor(
                [item["timestamp2"] for item in batch],
                dtype=torch.float64
            )
        
        return collated


class RetrievalCollator:
    """Collator for retrieval tasks with corpus and queries.
    
    Handles separate batching for document corpus and queries,
    optimized for efficient similarity search.
    """
    
    def __init__(
        self,
        tokenizer: TextBatcher,
        is_corpus: bool = True,
    ) -> None:
        """Initialize retrieval collator.
        
        Args:
            tokenizer: TextBatcher instance for tokenization.
            is_corpus: True for corpus docs, False for queries.
        """
        self.tokenizer = tokenizer
        self.is_corpus = is_corpus
        self.mode = "corpus" if is_corpus else "query"
        logger.info(f"Initialized retrieval collator for {self.mode}")
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of retrieval documents or queries.
        
        Args:
            batch: List of documents or queries with 'text' field.
            
        Returns:
            Dictionary with:
                - input_ids: Tokenized text inputs
                - attention_mask: Attention masks
                - doc_ids or query_ids: Original IDs
        """
        texts = [item["text"] for item in batch]
        
        # Tokenize texts
        encoding = self.tokenizer(texts, return_tensors="pt")
        
        if self.is_corpus:
            ids = torch.tensor([item["doc_id"] for item in batch], dtype=torch.long)
            id_key = "doc_ids"
        else:
            ids = torch.tensor([item["query_id"] for item in batch], dtype=torch.long)
            id_key = "query_ids"
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            id_key: ids,
        }


class TemporalQACollator:
    """Collator for temporal QA examples.
    
    Handles question-context pairs with temporal metadata,
    preparing them for temporal reasoning tasks.
    """
    
    def __init__(
        self,
        tokenizer: TextBatcher,
        max_context_length: int = 384,
        max_question_length: int = 64,
    ) -> None:
        """Initialize temporal QA collator.
        
        Args:
            tokenizer: TextBatcher instance for tokenization.
            max_context_length: Maximum length for context.
            max_question_length: Maximum length for question.
        """
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_question_length = max_question_length
        logger.info("Initialized temporal QA collator")
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of temporal QA examples.
        
        Args:
            batch: List of QA examples with question, context, answer, timestamp.
            
        Returns:
            Dictionary with:
                - question_inputs: Tokenized questions
                - context_inputs: Tokenized contexts  
                - answers: Answer texts (not tokenized for flexibility)
                - timestamps: Unix timestamps as tensor
                - temporal_expressions: Temporal phrases (if available)
        """
        questions = [item["question"] for item in batch]
        contexts = [item["context"] for item in batch]
        answers = [item.get("answer", "") for item in batch]
        
        # Tokenize questions and contexts separately for better control
        question_encoding = self.tokenizer(
            questions,
            return_tensors="pt",
            max_length=self.max_question_length,
            truncation=True,
            padding="max_length",
        )
        
        context_encoding = self.tokenizer(
            contexts,
            return_tensors="pt", 
            max_length=self.max_context_length,
            truncation=True,
            padding="max_length",
        )
        
        # Extract timestamps
        timestamps = torch.tensor(
            [item["timestamp"] for item in batch],
            dtype=torch.float64
        )
        
        collated = {
            "question_inputs": question_encoding,
            "context_inputs": context_encoding,
            "answers": answers,  # Keep as strings for flexibility
            "timestamps": timestamps,
        }
        
        # Include temporal expressions if available
        if "temporal_expression" in batch[0]:
            collated["temporal_expressions"] = [
                item.get("temporal_expression", "") for item in batch
            ]
        
        return collated


def create_collator(
    task: str,
    tokenizer: TextBatcher,
    **kwargs: Any,
) -> Union[STSBCollator, RetrievalCollator, TemporalQACollator]:
    """Factory function to create appropriate collator for task.
    
    Args:
        task: Task name ('stsb', 'retrieval_corpus', 'retrieval_query', 'temporal_qa').
        tokenizer: TextBatcher instance.
        **kwargs: Additional arguments for specific collator.
        
    Returns:
        Appropriate collator instance for the task.
        
    Raises:
        ValueError: If task is not recognized.
    """
    task_lower = task.lower()
    
    if task_lower == "stsb":
        return STSBCollator(tokenizer, **kwargs)
    elif task_lower == "retrieval_corpus":
        return RetrievalCollator(tokenizer, is_corpus=True, **kwargs)
    elif task_lower == "retrieval_query":
        return RetrievalCollator(tokenizer, is_corpus=False, **kwargs)
    elif task_lower in ["temporal_qa", "timeqa"]:
        return TemporalQACollator(tokenizer, **kwargs)
    else:
        raise ValueError(
            f"Unknown task: {task}. "
            f"Choose from: stsb, retrieval_corpus, retrieval_query, temporal_qa"
        )


def pad_timestamps(
    timestamps: List[torch.Tensor],
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad variable-length timestamp sequences.
    
    Args:
        timestamps: List of timestamp tensors.
        pad_value: Value to use for padding.
        
    Returns:
        Padded timestamp tensor.
    """
    if not timestamps:
        return torch.tensor([])
    
    if all(ts.dim() == 0 for ts in timestamps):
        # Scalar timestamps, just stack
        return torch.stack(timestamps)
    else:
        # Variable length sequences, pad
        return pad_sequence(timestamps, batch_first=True, padding_value=pad_value)
