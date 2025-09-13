"""
Simplified evaluator - only proven methods
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, pearsonr
from config.settings import EMBEDDING_DIM
from src.models.embedding_composer import EmbeddingComposer
from src.evaluation.baselines import BaselineModels
from src.data.dataset_loader import DatasetLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Simplified evaluator - removed complex failing methods
    """

    def __init__(self):
        logger.info("Initializing Evaluator...")
        self.composer = EmbeddingComposer()
        self.baselines = BaselineModels()
        self.data_loader = DatasetLoader()
        self.results = {}

    def evaluate_classification(self, dataset_name: str = 'ag_news') -> Dict:
        """
        Simple classification evaluation
        """
        logger.info(f"Evaluating classification on {dataset_name}")

        # Load data
        if dataset_name == 'ag_news':
            texts, labels = self.data_loader.load_ag_news()
        elif dataset_name == 'dbpedia':
            texts, labels = self.data_loader.load_dbpedia()
        else:
            texts, labels = self.data_loader.load_twenty_newsgroups()

        results = {}

        # 1. Test INSTRUCTOR if available
        if self.composer.has_instructor:
            logger.info("Testing INSTRUCTOR embeddings...")
            embeddings = self.composer.compose_batch(texts, method='instructor', task='classification')

            clf = LogisticRegression(max_iter=1000, random_state=42)
            scores = cross_val_score(clf, embeddings, labels, cv=5, scoring='accuracy')

            results['instructor'] = {
                'mean': scores.mean(),
                'std': scores.std()
            }
            logger.info(f"INSTRUCTOR: {scores.mean():.4f} ± {scores.std():.4f}")

        # 2. Test MPNet baseline (best single model)
        logger.info("Testing MPNet baseline...")
        baseline_embeddings = self.baselines.get_batch_embeddings(texts, 'mpnet')

        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, baseline_embeddings, labels, cv=5, scoring='accuracy')

        results['mpnet'] = {
            'mean': scores.mean(),
            'std': scores.std()
        }
        logger.info(f"MPNet: {scores.mean():.4f} ± {scores.std():.4f}")

        return results

    def evaluate_similarity(self) -> Dict:
        """
        Simple similarity evaluation
        """
        logger.info("Evaluating similarity on STS-B")

        text_pairs, true_scores = self.data_loader.load_stsb()
        results = {}

        # Test with INSTRUCTOR
        if self.composer.has_instructor:
            similarities = []
            for text1, text2 in text_pairs:
                emb1 = self.composer.compose(text1, method='instructor', task='similarity')
                emb2 = self.composer.compose(text2, method='instructor', task='similarity')

                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities.append(similarity)

            spearman_corr, _ = spearmanr(similarities, true_scores)
            pearson_corr, _ = pearsonr(similarities, true_scores)

            results['instructor'] = {
                'spearman': spearman_corr,
                'pearson': pearson_corr
            }
            logger.info(f"INSTRUCTOR: Spearman={spearman_corr:.4f}, Pearson={pearson_corr:.4f}")

        # MPNet baseline
        baseline_similarities = []
        for text1, text2 in text_pairs:
            emb1 = self.baselines.get_embedding(text1, 'mpnet')
            emb2 = self.baselines.get_embedding(text2, 'mpnet')

            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            baseline_similarities.append(similarity)

        spearman_corr, _ = spearmanr(baseline_similarities, true_scores)
        pearson_corr, _ = pearsonr(baseline_similarities, true_scores)

        results['mpnet'] = {
            'spearman': spearman_corr,
            'pearson': pearson_corr
        }
        logger.info(f"MPNet: Spearman={spearman_corr:.4f}, Pearson={pearson_corr:.4f}")

        return results

    # Remove all these methods that don't work:
    # - evaluate_classification_topk
    # - evaluate_moe_classification
    # - evaluate_moe_similarity
    # - learn_optimal_weights
    # - select_domains_by_performance
    # - run_ablation_study (keep simplified version)

    def run_ablation_study(self) -> Dict:
        """
        Simplified ablation - only test what matters
        """
        logger.info("Running simplified ablation...")
        texts, labels = self.data_loader.load_ag_news()

        results = {}

        # Only test INSTRUCTOR vs best baseline
        methods = ['instructor', 'mpnet']

        for method in methods:
            if method == 'instructor' and self.composer.has_instructor:
                embeddings = self.composer.compose_batch(texts, method='instructor')
            elif method == 'mpnet':
                embeddings = self.baselines.get_batch_embeddings(texts, 'mpnet')
            else:
                continue

            clf = LogisticRegression(max_iter=1000, random_state=42)
            scores = cross_val_score(clf, embeddings, labels, cv=3, scoring='accuracy')

            results[method] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std()
            }
            logger.info(f"{method}: {scores.mean():.4f} ± {scores.std():.4f}")

        return results