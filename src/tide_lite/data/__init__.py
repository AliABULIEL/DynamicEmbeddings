"""Data loading and processing utilities for TIDE-Lite."""

from .datasets import (
    load_stsb,
    load_quora,
    load_timeqa,
    verify_dataset_integrity,
)
from .collate import (
    TextBatcher,
    STSBCollator,
    RetrievalCollator,
    TemporalQACollator,
    create_collator,
)
from .dataloaders import (
    create_stsb_dataloaders,
    create_quora_dataloaders,
    create_temporal_qa_dataloader,
    create_all_dataloaders,
)

__all__ = [
    # Dataset loaders
    "load_stsb",
    "load_quora", 
    "load_timeqa",
    "verify_dataset_integrity",
    # Collators
    "TextBatcher",
    "STSBCollator",
    "RetrievalCollator",
    "TemporalQACollator",
    "create_collator",
    # DataLoader creators
    "create_stsb_dataloaders",
    "create_quora_dataloaders",
    "create_temporal_qa_dataloader",
    "create_all_dataloaders",
]
