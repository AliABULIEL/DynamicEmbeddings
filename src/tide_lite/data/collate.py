"""Collation utilities for batching text data.

This module provides collators for different task types:
- STS-B similarity pairs
- Retrieval corpus/query batching  
- Temporal QA examples
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Union

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
    """Collator for STS-B sentence pairs.
    
    Handles sentence pairs with similarity scores for training
    semantic textual similarity models.
    """
    
    def __init__(
        self,
        tokenizer: Union[TextBatcher, AutoTokenizer, str],
        max_length: int = 128,
    ) -> None:
        """Initialize STS-B collator.
        
        Args:
            tokenizer: TextBatcher, tokenizer instance, or model name string.
            max_length: Maximum sequence length.
        """
        if isinstance(tokenizer, str):
            self.tokenizer = TextBatcher(model_name=tokenizer, max_length=max_length)
        elif isinstance(tokenizer, TextBatcher):
            self.tokenizer = tokenizer
        else:
            # Assume it's a HF tokenizer
            self.tokenizer = TextBatcher(max_length=max_length)
            self.tokenizer.tokenizer = tokenizer
        
        logger.info("Initialized STS-B collator")
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of STS-B examples.
        
        Args:
            batch: List of examples with sentence1, sentence2, label.
            
        Returns:
            Dictionary with:
                - sentence1_inputs: Tokenized first sentences (input_ids, attention_mask)
                - sentence2_inputs: Tokenized second sentences (input_ids, attention_mask)
                - labels: Similarity scores as tensor [0, 5]
        """
        sentences1 = [item["sentence1"] for item in batch]
        sentences2 = [item["sentence2"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        
        # Tokenize sentences
        sent1_encoding = self.tokenizer(sentences1, return_tensors="pt")
        sent2_encoding = self.tokenizer(sentences2, return_tensors="pt")
        
        return {
            "sentence1_inputs": sent1_encoding,
            "sentence2_inputs": sent2_encoding,
            "labels": labels,
        }


class RetrievalCollator:
    """Collator for retrieval tasks with corpus and queries.
    
    Handles batching for document corpus and queries,
    optimized for efficient similarity search.
    """
    
    def __init__(
        self,
        tokenizer: Union[TextBatcher, AutoTokenizer, str],
        max_length: int = 128,
        is_corpus: bool = True,
    ) -> None:
        """Initialize retrieval collator.
        
        Args:
            tokenizer: TextBatcher, tokenizer instance, or model name string.
            max_length: Maximum sequence length.
            is_corpus: True for corpus docs, False for queries.
        """
        if isinstance(tokenizer, str):
            self.tokenizer = TextBatcher(model_name=tokenizer, max_length=max_length)
        elif isinstance(tokenizer, TextBatcher):
            self.tokenizer = tokenizer
        else:
            self.tokenizer = TextBatcher(max_length=max_length)
            self.tokenizer.tokenizer = tokenizer
        
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
    
    Handles question-context pairs for temporal reasoning tasks,
    including TimeQA and TempLAMA datasets.
    """
    
    def __init__(
        self,
        tokenizer: Union[TextBatcher, AutoTokenizer, str],
        max_context_length: int = 384,
        max_question_length: int = 64,
    ) -> None:
        """Initialize temporal QA collator.
        
        Args:
            tokenizer: TextBatcher, tokenizer instance, or model name string.
            max_context_length: Maximum length for context.
            max_question_length: Maximum length for question.
        """
        if isinstance(tokenizer, str):
            # Create two batcher instances with different max lengths
            self.question_tokenizer = TextBatcher(
                model_name=tokenizer, 
                max_length=max_question_length
            )
            self.context_tokenizer = TextBatcher(
                model_name=tokenizer,
                max_length=max_context_length
            )
        else:
            # Use same tokenizer with different configs
            base_tokenizer = tokenizer.tokenizer if isinstance(tokenizer, TextBatcher) else tokenizer
            self.question_tokenizer = TextBatcher(max_length=max_question_length)
            self.question_tokenizer.tokenizer = base_tokenizer
            self.context_tokenizer = TextBatcher(max_length=max_context_length)
            self.context_tokenizer.tokenizer = base_tokenizer
        
        logger.info("Initialized temporal QA collator")
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of temporal QA examples.
        
        Args:
            batch: List of QA examples with question, context, answer, optionally timestamp.
            
        Returns:
            Dictionary with:
                - question_inputs: Tokenized questions
                - context_inputs: Tokenized contexts  
                - answers: Answer texts (strings for flexibility)
                - timestamps: Unix timestamps if available
        """
        questions = [item["question"] for item in batch]
        contexts = [item["context"] for item in batch]
        answers = [item.get("answer", "") for item in batch]
        
        # Tokenize questions and contexts
        question_encoding = self.question_tokenizer(questions, return_tensors="pt")
        context_encoding = self.context_tokenizer(contexts, return_tensors="pt")
        
        collated = {
            "question_inputs": question_encoding,
            "context_inputs": context_encoding,
            "answers": answers,  # Keep as strings for flexibility
        }
        
        # Include timestamps if available in the data
        if "timestamp" in batch[0]:
            timestamps = torch.tensor(
                [item.get("timestamp", 0.0) for item in batch],
                dtype=torch.float32
            )
            collated["timestamps"] = timestamps
        
        return collated


def create_collator(
    task: str,
    tokenizer: Union[str, TextBatcher, AutoTokenizer] = None,
    max_length: int = 128,
    **kwargs: Any,
) -> Union[STSBCollator, RetrievalCollator, TemporalQACollator]:
    """Factory function to create appropriate collator for task.
    
    Args:
        task: Task name ('stsb', 'retrieval_corpus', 'retrieval_query', 'temporal_qa').
        tokenizer: Model name, TextBatcher, or tokenizer instance.
        max_length: Maximum sequence length.
        **kwargs: Additional arguments for specific collator.
        
    Returns:
        Appropriate collator instance for the task.
        
    Raises:
        ValueError: If task is not recognized.
    """
    task_lower = task.lower()
    
    # Default tokenizer if not provided
    if tokenizer is None:
        tokenizer = "sentence-transformers/all-MiniLM-L6-v2"
    
    if task_lower == "stsb":
        return STSBCollator(tokenizer, max_length=max_length, **kwargs)
    elif task_lower == "retrieval_corpus":
        return RetrievalCollator(tokenizer, max_length=max_length, is_corpus=True, **kwargs)
    elif task_lower == "retrieval_query":
        return RetrievalCollator(tokenizer, max_length=max_length, is_corpus=False, **kwargs)
    elif task_lower in ["temporal_qa", "timeqa", "templama"]:
        return TemporalQACollator(tokenizer, **kwargs)
    else:
        raise ValueError(
            f"Unknown task: {task}. "
            f"Choose from: stsb, retrieval_corpus, retrieval_query, temporal_qa"
        )
