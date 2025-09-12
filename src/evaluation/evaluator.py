"""
Enhanced evaluation logic with task-specific strategies
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import spearmanr, pearsonr
from config.settings import COMPOSITION_METHODS, EMBEDDING_DIM
from src.models.embedding_composer import EmbeddingComposer
from src.evaluation.baselines import BaselineModels, MultiEmbeddingBaseline
from src.data.dataset_loader import DatasetLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Enhanced evaluator with improved strategies
    """

    def __init__(self):
        """Initialize evaluator with models and data loader"""
        logger.info("Initializing Evaluator...")

        self.composer = EmbeddingComposer()
        self.baselines = BaselineModels()
        self.multi_baseline = MultiEmbeddingBaseline()
        self.data_loader = DatasetLoader()

        self.results = {}
        self.optimal_domains = None
        self.optimal_weights = None

    def evaluate_classification(self,
                                dataset_name: str = 'ag_news',
                                composition_method: str = 'weighted_sum',
                                use_task_specific: bool = True) -> Dict:
        """
        Evaluate on classification task with improvements
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

        # 1. Evaluate composed embeddings with task-specific strategy
        if use_task_specific:
            logger.info("Using task-specific composition (Top-4)")
            composed_embeddings = [
                self.composer.compose_for_task(text, task='classification')
                for text in texts
            ]
        else:
            logger.info(f"Getting composed embeddings with {composition_method}...")
            composed_embeddings = self.composer.compose_batch(
                texts,
                method=composition_method,
                task='classification'
            )

        composed_embeddings = np.array(composed_embeddings)

        # Train classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, composed_embeddings, labels, cv=5, scoring='accuracy')

        method_name = 'composed_task_specific' if use_task_specific else f'composed_{composition_method}'
        results[method_name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }

        logger.info(f"Composed: {scores.mean():.4f} ± {scores.std():.4f}")

        # 2. Evaluate with learned optimal weights
        logger.info("Learning optimal domain weights...")
        optimal_weights = self.learn_optimal_weights(texts[:500], labels[:500])

        if optimal_weights is not None:
            embeddings = []
            for text in texts:
                emb = self.compose_with_learned_weights(text, optimal_weights)
                embeddings.append(emb)
            embeddings = np.array(embeddings)

            scores = cross_val_score(clf, embeddings, labels, cv=5, scoring='accuracy')
            results['composed_optimal_weights'] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist(),
                'weights': optimal_weights.tolist()
            }
            logger.info(f"Optimal weights: {scores.mean():.4f} ± {scores.std():.4f}")

        # 3. Evaluate attention-based composition
        logger.info("Testing attention-based composition...")
        attention_embeddings = []
        for text in texts[:500]:  # Test on subset
            emb = self.composer.attention_compose(text)
            attention_embeddings.append(emb)
        attention_embeddings = np.array(attention_embeddings)

        scores = cross_val_score(clf, attention_embeddings, labels[:500], cv=3, scoring='accuracy')
        results['composed_attention'] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
        logger.info(f"Attention-based: {scores.mean():.4f} ± {scores.std():.4f}")

        # 4. Evaluate baselines
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

        # 5. Evaluate performance-selected domains
        logger.info("Selecting domains by performance...")
        best_domains = self.select_domains_by_performance(texts[:200], labels[:200])

        if best_domains:
            embeddings = []
            for text in texts:
                emb = self.compose_with_selected_domains(text, best_domains)
                embeddings.append(emb)
            embeddings = np.array(embeddings)

            scores = cross_val_score(clf, embeddings, labels, cv=5, scoring='accuracy')
            results['composed_selected_domains'] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist(),
                'domains': best_domains
            }
            logger.info(f"Selected domains: {scores.mean():.4f} ± {scores.std():.4f}")
            logger.info(f"Best domains: {best_domains}")

        # 6. Evaluate individual domain models
        logger.info("Evaluating individual domain models...")
        for domain in self.composer.domains:
            try:
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
            except Exception as e:
                logger.warning(f"Failed to evaluate domain {domain}: {e}")

        return results

    def evaluate_similarity(self,
                            composition_method: str = 'weighted_sum',
                            use_task_specific: bool = True) -> Dict:
        """
        Evaluate on semantic similarity task with improvements
        """
        logger.info("Evaluating semantic similarity on STS-B")

        # Load data
        text_pairs, true_scores = self.data_loader.load_stsb()

        results = {}

        # 1. Task-specific strategy (single best domain)
        if use_task_specific:
            logger.info("Using task-specific strategy (best single domain)")

            composed_similarities = []
            for text1, text2 in text_pairs:
                emb1 = self.composer.compose_for_task(text1, task='similarity')
                emb2 = self.composer.compose_for_task(text2, task='similarity')

                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                composed_similarities.append(similarity)

            spearman_corr, _ = spearmanr(composed_similarities, true_scores)
            pearson_corr, _ = pearsonr(composed_similarities, true_scores)

            results['task_specific_similarity'] = {
                'spearman': spearman_corr,
                'pearson': pearson_corr
            }

            logger.info(f"Task-specific: Spearman={spearman_corr:.4f}, Pearson={pearson_corr:.4f}")

        # 2. Standard composed embeddings
        logger.info(f"Computing similarities with composed embeddings ({composition_method})...")

        composed_similarities = []
        for text1, text2 in text_pairs:
            emb1 = self.composer.compose(text1, method=composition_method)
            emb2 = self.composer.compose(text2, method=composition_method)

            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            composed_similarities.append(similarity)

        spearman_corr, _ = spearmanr(composed_similarities, true_scores)
        pearson_corr, _ = pearsonr(composed_similarities, true_scores)

        results[f'composed_{composition_method}'] = {
            'spearman': spearman_corr,
            'pearson': pearson_corr
        }

        logger.info(f"Composed: Spearman={spearman_corr:.4f}, Pearson={pearson_corr:.4f}")

        # 3. Evaluate baselines
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

    def evaluate_classification_topk(self, dataset_name: str, k: int = 2,
                                    base_method: str = 'weighted_sum') -> Dict:
        """
        Evaluate classification using top-k domains
        """
        logger.info(f"Evaluating classification with top-{k} domains on {dataset_name}")

        # Load data
        if dataset_name == 'ag_news':
            texts, labels = self.data_loader.load_ag_news()
        elif dataset_name == 'dbpedia':
            texts, labels = self.data_loader.load_dbpedia()
        else:
            texts, labels = self.data_loader.load_twenty_newsgroups()

        results = {}

        # Evaluate top-k composition
        embeddings = []
        for text in texts:
            emb = self.composer.compose_topk(text, k=k, method=base_method)
            embeddings.append(emb)
        embeddings = np.array(embeddings)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, embeddings, labels, cv=5, scoring='accuracy')

        results[f'composed_top{k}_{base_method}'] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }

        logger.info(f"Top-{k} {base_method}: {scores.mean():.4f} ± {scores.std():.4f}")

        return results

    def evaluate_similarity_best_domain(self) -> Dict:
        """
        Evaluate similarity using best single domain for each text pair
        """
        logger.info("Evaluating similarity with best domain selection")

        text_pairs, true_scores = self.data_loader.load_stsb()
        results = {}

        similarities = []
        for text1, text2 in text_pairs:
            # Get best domain for each text
            probs1 = self.composer.classifier.classify(text1)
            probs2 = self.composer.classifier.classify(text2)

            # Use the domain that's best for both (highest average prob)
            avg_probs = (probs1 + probs2) / 2
            best_domain = self.composer.domains[np.argmax(avg_probs)]

            emb1 = self.composer.embedder_manager.get_embedding(text1, best_domain)
            emb2 = self.composer.embedder_manager.get_embedding(text2, best_domain)

            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append(similarity)

        spearman_corr, _ = spearmanr(similarities, true_scores)
        pearson_corr, _ = pearsonr(similarities, true_scores)

        results['best_domain_selection'] = {
            'spearman': spearman_corr,
            'pearson': pearson_corr
        }

        logger.info(f"Best domain: Spearman={spearman_corr:.4f}, Pearson={pearson_corr:.4f}")

        return results

    def learn_optimal_weights(self, texts: List[str], labels: List[int],
                             k: int = 3) -> Optional[np.ndarray]:
        """
        Learn optimal domain weights using validation data
        """
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                texts, labels, test_size=0.3, random_state=42, stratify=labels)

            best_weights = None
            best_score = 0

            # Grid search over weight combinations (simplified)
            for w1 in np.arange(0, 1.1, 0.25):
                for w2 in np.arange(0, 1.1 - w1, 0.25):
                    for w3 in np.arange(0, 1.1 - w1 - w2, 0.25):
                        w4 = max(0, 1.0 - w1 - w2 - w3)
                        weights = np.array([w1, w2, w3, w4, 0])  # Top-4 only

                        # Test these weights
                        embeddings = []
                        for text in X_val[:100]:  # Use subset for speed
                            emb = self.compose_with_learned_weights(text, weights)
                            embeddings.append(emb)

                        embeddings = np.array(embeddings)

                        # Evaluate
                        clf = LogisticRegression(max_iter=500, random_state=42)
                        clf.fit(embeddings, y_val[:100])
                        score = clf.score(embeddings, y_val[:100])

                        if score > best_score:
                            best_score = score
                            best_weights = weights

            logger.info(f"Best learned weights: {best_weights}, Score: {best_score:.4f}")
            return best_weights

        except Exception as e:
            logger.error(f"Failed to learn optimal weights: {e}")
            return None

    def compose_with_learned_weights(self, text: str, weights: np.ndarray) -> np.ndarray:
        """
        Compose using learned fixed weights instead of probabilities
        """
        domain_embeddings = self.composer.embedder_manager.get_all_embeddings(text)

        final_embedding = np.zeros(EMBEDDING_DIM)
        for i, domain in enumerate(self.composer.domains):
            if i < len(weights) and domain in domain_embeddings:
                final_embedding += weights[i] * domain_embeddings[domain]

        return final_embedding

    def select_domains_by_performance(self, texts: List[str], labels: List[int]) -> List[str]:
        """
        Select domains based on actual performance
        """
        domain_scores = {}

        for domain in self.composer.domains:
            try:
                embeddings = [
                    self.composer.embedder_manager.get_embedding(text, domain)
                    for text in texts[:100]
                ]
                clf = LogisticRegression(max_iter=500, random_state=42)
                scores = cross_val_score(clf, embeddings, labels[:100], cv=3, scoring='accuracy')
                domain_scores[domain] = scores.mean()
                logger.info(f"Domain {domain} performance: {scores.mean():.4f}")
            except Exception as e:
                logger.warning(f"Failed to evaluate domain {domain}: {e}")
                domain_scores[domain] = 0

        # Return top-3 performing domains
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        best_domains = [d[0] for d in sorted_domains[:3]]

        return best_domains

    def compose_with_selected_domains(self, text: str, domains: List[str]) -> np.ndarray:
        """
        Compose using only selected domains
        """
        embeddings = []
        for domain in domains:
            try:
                emb = self.composer.embedder_manager.get_embedding(text, domain)
                embeddings.append(emb)
            except:
                embeddings.append(np.zeros(EMBEDDING_DIM))

        # Simple average of selected domains
        return np.mean(embeddings, axis=0)

    def run_ablation_study(self) -> Dict:
        """
        Enhanced ablation studies
        """
        logger.info("Running ablation studies...")

        texts, labels = self.data_loader.load_ag_news()
        ablation_results = {}

        # Test different composition methods
        for method in COMPOSITION_METHODS + ['attention_based']:
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
                if domain in domain_embeddings:
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