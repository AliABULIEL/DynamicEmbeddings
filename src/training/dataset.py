"""
Dataset classes for training Dynamic Embeddings
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Optional, Tuple, Dict
import random
import logging

logger = logging.getLogger(__name__)


class DynamicEmbeddingDataset(Dataset):
    """
    Dataset for training dynamic embeddings with contrastive learning
    """

    def __init__(self,
                 texts: List[str],
                 labels: Optional[List[int]] = None,
                 domains: Optional[List[str]] = None,
                 augment: bool = True,
                 augmentation_prob: float = 0.5):
        """
        Initialize dataset

        Args:
            texts: List of text samples
            labels: Optional labels for supervised contrastive learning
            domains: Optional domain labels for analysis
            augment: Whether to apply data augmentation
            augmentation_prob: Probability of applying augmentation
        """
        self.texts = texts
        self.labels = labels if labels is not None else list(range(len(texts)))
        self.domains = domains if domains is not None else ['unknown'] * len(texts)
        self.augment = augment
        self.augmentation_prob = augmentation_prob

        # Create label to indices mapping for positive pair mining
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        domain = self.domains[idx]

        # Apply augmentation
        if self.augment and random.random() < self.augmentation_prob:
            text = self._augment_text(text)

        return text, label

    def _augment_text(self, text: str) -> str:
        """
        Apply text augmentation techniques

        Args:
            text: Input text

        Returns:
            Augmented text
        """
        augmentation_type = random.choice(['truncate', 'mask', 'paraphrase', 'noise'])

        if augmentation_type == 'truncate':
            # Randomly truncate text
            words = text.split()
            if len(words) > 10:
                start = random.randint(0, max(1, len(words) - 10))
                end = min(start + random.randint(5, 15), len(words))
                text = ' '.join(words[start:end])

        elif augmentation_type == 'mask':
            # Randomly mask words
            words = text.split()
            if len(words) > 5:
                num_mask = random.randint(1, min(3, len(words) // 5))
                mask_indices = random.sample(range(len(words)), num_mask)
                for idx in mask_indices:
                    words[idx] = '[MASK]'
                text = ' '.join(words)

        elif augmentation_type == 'paraphrase':
            # Simple paraphrase by reordering clauses
            if ',' in text:
                parts = text.split(',')
                random.shuffle(parts)
                text = ','.join(parts)
            elif '.' in text:
                sentences = text.split('.')
                random.shuffle(sentences)
                text = '.'.join(sentences)

        elif augmentation_type == 'noise':
            # Add noise words
            noise_words = ['basically', 'actually', 'indeed', 'moreover', 'furthermore']
            words = text.split()
            if len(words) > 5:
                insert_pos = random.randint(1, len(words) - 1)
                words.insert(insert_pos, random.choice(noise_words))
                text = ' '.join(words)

        return text

    def get_positive_pair(self, idx: int) -> Tuple[str, str]:
        """
        Get a positive pair for contrastive learning

        Args:
            idx: Index of anchor sample

        Returns:
            Tuple of (anchor_text, positive_text)
        """
        anchor_text = self.texts[idx]
        label = self.labels[idx]

        # Find other samples with same label
        positive_indices = [i for i in self.label_to_indices[label] if i != idx]

        if positive_indices:
            positive_idx = random.choice(positive_indices)
            positive_text = self.texts[positive_idx]
        else:
            # If no other samples with same label, use augmentation
            positive_text = self._augment_text(anchor_text)

        return anchor_text, positive_text


class ContrastiveLearningDataset(Dataset):
    """
    Dataset specifically for contrastive learning with positive/negative pairs
    """

    def __init__(self,
                 texts: List[str],
                 labels: Optional[List[int]] = None,
                 domains: Optional[List[str]] = None,
                 num_negatives: int = 5):
        """
        Initialize contrastive learning dataset

        Args:
            texts: List of text samples
            labels: Optional labels for supervised contrastive
            domains: Optional domain labels
            num_negatives: Number of negative samples per positive
        """
        self.texts = texts
        self.labels = labels if labels is not None else list(range(len(texts)))
        self.domains = domains if domains is not None else ['unknown'] * len(texts)
        self.num_negatives = num_negatives

        # Create label to indices mapping
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        # Pre-compute negative indices for efficiency
        self._precompute_negatives()

    def _precompute_negatives(self):
        """Pre-compute negative samples for each index"""
        self.negative_indices = {}

        for idx in range(len(self.texts)):
            label = self.labels[idx]
            # Get all indices with different labels
            negative_pool = []
            for other_label, indices in self.label_to_indices.items():
                if other_label != label:
                    negative_pool.extend(indices)

            # Sample negatives
            if len(negative_pool) >= self.num_negatives:
                self.negative_indices[idx] = random.sample(
                    negative_pool, self.num_negatives
                )
            else:
                # If not enough negatives, use all available
                self.negative_indices[idx] = negative_pool

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        anchor_text = self.texts[idx]
        label = self.labels[idx]

        # Get positive sample
        positive_indices = [i for i in self.label_to_indices[label] if i != idx]
        if positive_indices:
            positive_idx = random.choice(positive_indices)
            positive_text = self.texts[positive_idx]
        else:
            # Use augmentation if no other positives
            positive_text = anchor_text  # Will be augmented in collate_fn

        # Get negative samples
        negative_texts = [self.texts[i] for i in self.negative_indices[idx]]

        return {
            'anchor': anchor_text,
            'positive': positive_text,
            'negatives': negative_texts,
            'label': label
        }


def create_training_dataset(config, texts: List[str],
                            labels: Optional[List[int]] = None,
                            domains: Optional[List[str]] = None) -> Dataset:
    """
    Create appropriate dataset based on configuration

    Args:
        config: Training configuration
        texts: List of texts
        labels: Optional labels
        domains: Optional domain labels

    Returns:
        Dataset instance
    """
    if config.training.use_contrastive_pairs:
        return ContrastiveLearningDataset(
            texts, labels, domains,
            num_negatives=config.training.num_negatives
        )
    else:
        return DynamicEmbeddingDataset(
            texts, labels, domains,
            augment=config.data.use_augmentation,
            augmentation_prob=config.data.augmentation_prob
        )


def collate_fn(batch):
    """
    Custom collate function for batching

    Args:
        batch: List of samples from dataset

    Returns:
        Batched tensors
    """
    if isinstance(batch[0], tuple):
        # Simple dataset
        texts, labels = zip(*batch)
        return list(texts), torch.tensor(labels)
    elif isinstance(batch[0], dict):
        # Contrastive learning dataset
        anchors = [item['anchor'] for item in batch]
        positives = [item['positive'] for item in batch]
        negatives = [neg for item in batch for neg in item['negatives']]
        labels = [item['label'] for item in batch]

        return {
            'anchors': anchors,
            'positives': positives,
            'negatives': negatives,
            'labels': torch.tensor(labels)
        }
    else:
        raise ValueError(f"Unknown batch type: {type(batch[0])}")