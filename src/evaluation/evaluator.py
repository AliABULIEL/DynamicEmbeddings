"""
Main evaluation logic for experiments
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import spearmanr, pearsonr
from config.settings import COMPOSITION_METHODS
from src.models.embedding_composer import EmbeddingComposer
from src.evaluation.baselines import BaselineModels
from src.data.dataset_loader import DatasetLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Evaluates embedding composition methods against baselines
    """

    def __init__(self):
        """Initialize evaluator with models and data loader"""
        logger.info("Initializing Evaluator...")

        self.composer = EmbeddingComposer()
        self.baselines = BaselineModels()
        self.data_loader = DatasetLoader()

        self.results = {}

    def evaluate_classification(self,
                                dataset_name: str = 'ag_news',
                                composition_method: str = 'weighted_sum') -> Dict:
        """
        Evaluate on classification task

        Args:
            dataset_name: Name of dataset to use
            composition_method: Composition method to evaluate

        Returns:
            Dictionary with results
        """
        logger.info(f"Evaluating classification on {dataset_name}")

        # Load data
        if dataset_name == 'stsb':
            logger.error("STS-B is not a classification dataset")
            return {}

        texts, labels = self.data_loader.load_ag_news() if dataset_name == 'ag_news' \
            else self.data_loader.load_dbpedia() if dataset_name == 'dbpedia' \
            else self.data_loader.load_twenty_newsgroups()

        results = {}

        # 1. Evaluate composed embeddings
        logger.info(f"Getting composed embeddings with {composition_method}...")
        composed_embeddings = self.composer.compose_batch(
            texts,
            method=composition_method
        )

        # Train classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, composed_embeddings, labels, cv=5, scoring='accuracy')

        results[f'composed_{composition_method}'] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }

        logger.info(f"Composed ({composition_method}): {scores.mean():.4f} ± {scores.std():.4f}")

        # 2. Evaluate baselines
        for baseline_name in ['bert-base', 'mpnet', 'minilm']:
            logger.info(f"Evaluating baseline: {baseline_name}")

            baseline_embeddings = self.baselines.get_batch_embeddings(
                texts,
                baseline_name
            )

            scores = cross_val_score(clf, baseline_embeddings, labels, cv=5, scoring='accuracy')

            results[baseline_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }

            logger.info(f"{baseline_name}: {scores.mean():.4f} ± {scores.std():.4f}")

        # 3. Evaluate individual domain models
        logger.info("Evaluating individual domain models...")
        for domain in self.composer.domains:
            domain_embeddings = np.array([
                self.composer.embedder_manager.get_embedding(text, domain)
                for text in texts
            ])

            scores = cross_val_score(clf, domain_embeddings, labels, cv=5, scoring='accuracy')

            results[f'domain_{domain}'] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }

            logger.info(f"Domain {domain}: {scores.mean():.4f} ± {scores.std():.4f}")

        return results

    def evaluate_similarity(self,
                            composition_method: str = 'weighted_sum') -> Dict:
        """
        Evaluate on semantic similarity task (STS-B)

        Args:
            composition_method: Composition method to evaluate

        Returns:
            Dictionary with correlation results
        """
        logger.info("Evaluating semantic similarity on STS-B")

        # Load data
        text_pairs, true_scores = self.data_loader.load_stsb()

        results = {}

        # 1. Evaluate composed embeddings
        logger.info(f"Computing similarities with composed embeddings ({composition_method})...")

        composed_similarities = []
        for text1, text2 in text_pairs:
            emb1 = self.composer.compose(text1, method=composition_method)
            emb2 = self.composer.compose(text2, method=composition_method)

            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            composed_similarities.append(similarity)

        # Calculate correlations
        spearman_corr, _ = spearmanr(composed_similarities, true_scores)
        pearson_corr, _ = pearsonr(composed_similarities, true_scores)

        results[f'composed_{composition_method}'] = {
            'spearman': spearman_corr,
            'pearson': pearson_corr
        }

        logger.info(f"Composed: Spearman={spearman_corr:.4f}, Pearson={pearson_corr:.4f}")

        # 2. Evaluate baselines
        for baseline_name in ['bert-base', 'mpnet']:
            logger.info(f"Evaluating baseline: {baseline_name}")

            baseline_similarities = []
            for text1, text2 in text_pairs:
                emb1 = self.baselines.get_embedding(text1, baseline_name)
                emb2 = self.baselines.get_embedding(text2, baseline_name)

                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                baseline_similarities.append(similarity)

            spearman_corr, _ = spearmanr(baseline_similarities, true_scores)
            pearson_corr, _ = pearsonr(baseline_similarities, true_scores)

            results[baseline_name] = {
                'spearman': spearman_corr,
                'pearson': pearson_corr
            }

            logger.info(f"{baseline_name}: Spearman={spearman_corr:.4f}, Pearson={pearson_corr:.4f}")

        return results

    def run_ablation_study(self) -> Dict:
        """
        Run ablation studies on composition methods

        Returns:
            Dictionary with ablation results
        """
        logger.info("Running ablation studies...")

        # Test different composition methods on AG News
        texts, labels = self.data_loader.load_ag_news()

        ablation_results = {}

        for method in COMPOSITION_METHODS:
            logger.info(f"Testing composition method: {method}")

            try:
                embeddings = self.composer.compose_batch(texts, method=method)

                clf = LogisticRegression(max_iter=1000, random_state=42)
                scores = cross_val_score(clf, embeddings, labels, cv=3, scoring='accuracy')

                ablation_results[method] = {
                    'mean_accuracy': scores.mean(),
                    'std_accuracy': scores.std()
                }

                logger.info(f"{method}: {scores.mean():.4f} ± {scores.std():.4f}")

            except Exception as e:
                logger.error(f"Failed to evaluate {method}: {e}")
                ablation_results[method] = {'error': str(e)}

        return ablation_results

    def analyze_domain_influence(self, sample_texts: Optional[List[str]] = None) -> Dict:
        """
        Analyze how domain probabilities affect final embeddings

        Args:
            sample_texts: Optional list of texts to analyze

        Returns:
            Analysis results
        """
        if sample_texts is None:
            sample_texts = [
                "The COVID-19 vaccine showed 95% efficacy in clinical trials",
                "Apple stock surged after announcing quarterly earnings",
                "The Supreme Court ruled on the constitutional challenge",
                "Breaking: Major earthquake hits Japan, tsunami warning issued",
                "LOL this tweet is going viral! 😂 #trending"
            ]

        analysis = []

        for text in sample_texts:
            # Get detailed composition info
            embedding, domain_probs, domain_embeddings = self.composer.compose(
                text,
                method='weighted_sum',
                return_details=True
            )

            # Calculate contribution of each domain
            contributions = {}
            for i, domain in enumerate(self.composer.domains):
                domain_contribution = domain_probs[i] * np.linalg.norm(domain_embeddings[domain])
                contributions[domain] = domain_contribution

            analysis.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'domain_probs': {d: float(p) for d, p in zip(self.composer.domains, domain_probs)},
                'dominant_domain': self.composer.domains[np.argmax(domain_probs)],
                'entropy': float(-np.sum(domain_probs * np.log(domain_probs + 1e-10))),
                'contributions': contributions
            })

        return analysis