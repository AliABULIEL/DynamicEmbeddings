"""
Dataset loading and management
src/data/dataset_loader.py
"""
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import logging

# Note: In production, these would come from config
# from config.settings import EVALUATION_DATASETS, TEST_SAMPLE_SIZE

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Handles loading and preprocessing of evaluation datasets
    """

    def __init__(self, sample_size: Optional[int] = 1000):
        """
        Initialize dataset loader

        Args:
            sample_size: Number of samples to use (None for full dataset)
        """
        self.sample_size = sample_size
        self.datasets = {}

    def load_ag_news(self) -> Tuple[List[str], List[int]]:
        """
        Load AG News dataset for classification
        """
        logger.info("Loading AG News dataset...")

        try:
            if self.sample_size:
                dataset = load_dataset('ag_news', split=f'test[:{self.sample_size}]')
            else:
                dataset = load_dataset('ag_news', split='test')

            texts = dataset['text']
            labels = dataset['label']

            # IMPORTANT: Convert to Python list to avoid type issues
            texts = list(texts)
            labels = [int(label) for label in labels]  # Convert to Python int

            logger.info(f"Loaded {len(texts)} samples from AG News")
            logger.info(f"Label distribution: {np.bincount(labels)}")

            return texts, labels

        except Exception as e:
            logger.error(f"Failed to load AG News: {e}")
            raise

    def load_dbpedia(self) -> Tuple[List[str], List[int]]:
        """
        Load DBPedia dataset for classification
        """
        logger.info("Loading DBPedia dataset...")

        try:
            if self.sample_size:
                dataset = load_dataset('dbpedia_14', split=f'test[:{self.sample_size}]')
            else:
                dataset = load_dataset('dbpedia_14', split='test')

            # Combine title and content
            texts = [f"{title}. {content}" for title, content in
                     zip(dataset['title'], dataset['content'])]

            # Convert labels to Python int list
            labels = [int(label) for label in dataset['label']]

            logger.info(f"Loaded {len(texts)} samples from DBPedia")

            return texts, labels

        except Exception as e:
            logger.error(f"Failed to load DBPedia: {e}")
            raise

    def load_stsb(self) -> Tuple[List[Tuple[str, str]], List[float]]:
        """
        Load STS Benchmark dataset for semantic similarity

        STS-B contains sentence pairs with similarity scores from 0-5

        Returns:
            Tuple of (text_pairs, similarity_scores)
        """
        logger.info("Loading STS Benchmark dataset...")

        try:
            if self.sample_size:
                dataset = load_dataset('glue', 'stsb',
                                       split=f'validation[:{self.sample_size}]')
            else:
                dataset = load_dataset('glue', 'stsb', split='validation')

            text_pairs = []
            scores = []

            for item in dataset:
                # Create pairs
                pair = (item['sentence1'], item['sentence2'])
                text_pairs.append(pair)

                # Normalize scores from 0-5 to 0-1
                normalized_score = item['label'] / 5.0
                scores.append(normalized_score)

            logger.info(f"Loaded {len(text_pairs)} sentence pairs from STS-B")
            logger.info(f"Score range: {min(scores):.2f} - {max(scores):.2f}")

            return text_pairs, scores

        except Exception as e:
            logger.error(f"Failed to load STS-B: {e}")
            raise

    def load_twenty_newsgroups(self) -> Tuple[List[str], List[int]]:
        """
        Load 20 Newsgroups dataset for classification

        20 Newsgroups has 20 classes of newsgroup topics

        Returns:
            Tuple of (texts, labels)
        """
        logger.info("Loading 20 Newsgroups dataset...")

        try:
            # Using sklearn's fetch_20newsgroups
            newsgroups = fetch_20newsgroups(
                subset='test',
                remove=('headers', 'footers', 'quotes'),  # Clean the text
                shuffle=True,
                random_state=42
            )

            texts = newsgroups.data
            labels = newsgroups.target

            # Apply sampling if needed
            if self.sample_size and len(texts) > self.sample_size:
                # Random sample
                indices = np.random.choice(len(texts), self.sample_size, replace=False)
                texts = [texts[i] for i in indices]
                labels = [labels[i] for i in indices]

            logger.info(f"Loaded {len(texts)} samples from 20 Newsgroups")
            logger.info(f"Number of classes: {len(newsgroups.target_names)}")

            return texts, labels

        except Exception as e:
            logger.error(f"Failed to load 20 Newsgroups: {e}")
            raise

    def load_custom_dataset(self, filepath: str) -> Tuple[List[str], List[int]]:
        """
        Load a custom dataset from file

        Expected format: CSV with 'text' and 'label' columns

        Args:
            filepath: Path to the dataset file

        Returns:
            Tuple of (texts, labels)
        """
        import pandas as pd

        logger.info(f"Loading custom dataset from {filepath}...")

        try:
            df = pd.read_csv(filepath)

            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must have 'text' and 'label' columns")

            texts = df['text'].tolist()
            labels = df['label'].tolist()

            # Apply sampling if needed
            if self.sample_size and len(texts) > self.sample_size:
                df_sampled = df.sample(n=self.sample_size, random_state=42)
                texts = df_sampled['text'].tolist()
                labels = df_sampled['label'].tolist()

            logger.info(f"Loaded {len(texts)} samples from custom dataset")

            return texts, labels

        except Exception as e:
            logger.error(f"Failed to load custom dataset: {e}")
            raise

    def load_all_datasets(self) -> Dict:
        """
        Load all evaluation datasets

        Returns:
            Dictionary with all datasets
        """
        logger.info("Loading all evaluation datasets...")

        datasets = {}

        # Classification datasets
        try:
            datasets['ag_news'] = self.load_ag_news()
        except Exception as e:
            logger.warning(f"Skipping AG News: {e}")

        try:
            datasets['dbpedia'] = self.load_dbpedia()
        except Exception as e:
            logger.warning(f"Skipping DBPedia: {e}")

        try:
            datasets['twenty_newsgroups'] = self.load_twenty_newsgroups()
        except Exception as e:
            logger.warning(f"Skipping 20 Newsgroups: {e}")

        # Similarity dataset
        try:
            datasets['stsb'] = self.load_stsb()
        except Exception as e:
            logger.warning(f"Skipping STS-B: {e}")

        logger.info(f"Successfully loaded {len(datasets)} datasets")

        return datasets

    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a dataset without loading it fully

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset information
        """
        info = {}

        if dataset_name == 'ag_news':
            info = {
                'name': 'AG News',
                'task': 'classification',
                'num_classes': 4,
                'classes': ['World', 'Sports', 'Business', 'Sci/Tech'],
                'size': '7,600 test samples',
                'description': 'News articles categorized into 4 topics'
            }
        elif dataset_name == 'dbpedia':
            info = {
                'name': 'DBPedia',
                'task': 'classification',
                'num_classes': 14,
                'size': '70,000 test samples',
                'description': 'Wikipedia articles categorized into 14 ontology classes'
            }
        elif dataset_name == 'stsb':
            info = {
                'name': 'STS Benchmark',
                'task': 'similarity',
                'score_range': '0-5 (normalized to 0-1)',
                'size': '1,379 validation pairs',
                'description': 'Sentence pairs with human-annotated similarity scores'
            }
        elif dataset_name == 'twenty_newsgroups':
            info = {
                'name': '20 Newsgroups',
                'task': 'classification',
                'num_classes': 20,
                'size': '7,532 test samples',
                'description': 'Newsgroup posts across 20 different topics'
            }

        return info


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize loader with small sample for testing
    loader = DatasetLoader(sample_size=100)

    # Test loading each dataset
    print("\n" + "=" * 50)
    print("Testing Dataset Loader")
    print("=" * 50)

    # Test AG News
    print("\n1. Testing AG News:")
    texts, labels = loader.load_ag_news()
    print(f"   Loaded {len(texts)} texts")
    print(f"   First text: {texts[0][:100]}...")
    print(f"   Label distribution: {np.bincount(labels)}")

    # Test STS-B
    print("\n2. Testing STS-B:")
    pairs, scores = loader.load_stsb()
    print(f"   Loaded {len(pairs)} pairs")
    print(f"   First pair: '{pairs[0][0][:50]}...' <-> '{pairs[0][1][:50]}...'")
    print(f"   Similarity: {scores[0]:.3f}")

    # Get dataset info
    print("\n3. Dataset Information:")
    for dataset in ['ag_news', 'dbpedia', 'stsb']:
        info = loader.get_dataset_info(dataset)
        print(f"\n   {info['name']}:")
        print(f"   - Task: {info['task']}")
        print(f"   - Description: {info['description']}")