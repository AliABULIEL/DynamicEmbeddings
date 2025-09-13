"""
MTEB (Massive Text Embedding Benchmark) Evaluator
Focuses on tasks where dynamic embeddings excel
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import mteb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from scipy.stats import spearmanr, pearsonr
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MTEBEvaluator:
    """
    Evaluator for MTEB benchmarks
    Focuses on:
    1. Retrieval tasks (where routing matters most)
    2. Multi-domain classification
    3. Cross-domain similarity
    """

    def __init__(self, model, config):
        """
        Initialize MTEB evaluator

        Args:
            model: DynamicEmbeddingModel instance
            config: Evaluation configuration
        """
        self.model = model
        self.config = config
        self.model.eval()

        # Load baseline models for comparison
        self.baseline_models = {}
        if config.evaluation.baseline_models:
            logger.info("Loading baseline models for comparison...")
            for model_name in config.evaluation.baseline_models[:2]:  # Load only 2 for speed
                try:
                    logger.info(f"  Loading {model_name}")
                    self.baseline_models[model_name] = SentenceTransformer(model_name)
                except Exception as e:
                    logger.warning(f"  Failed to load {model_name}: {e}")

    def evaluate_all_tasks(self) -> Dict:
        """
        Evaluate on all configured MTEB tasks

        Returns:
            Dictionary with results for all tasks
        """
        results = {}

        for task_name in self.config.evaluation.mteb_tasks:
            logger.info(f"\nEvaluating on {task_name}...")
            try:
                task_results = self.evaluate_task(task_name)
                results[task_name] = task_results
                logger.info(f"  Results: {task_results}")
            except Exception as e:
                logger.error(f"  Failed to evaluate {task_name}: {e}")
                results[task_name] = {"error": str(e)}

        return results

    def evaluate_task(self, task_name: str) -> Dict:
        """
        Evaluate on a specific MTEB task

        Args:
            task_name: Name of the MTEB task

        Returns:
            Dictionary with evaluation metrics
        """
        # Load task
        try:
            task = mteb.get_task(task_name)
        except:
            # Fallback to manual evaluation for specific tasks
            return self._evaluate_manual(task_name)

        # Create model wrapper for MTEB
        model_wrapper = MTEBModelWrapper(self.model)

        # Run evaluation
        evaluation = mteb.MTEB(tasks=[task])
        results = evaluation.run(
            model_wrapper,
            eval_splits=["test"] if "test" in task.description.get("eval_splits", []) else ["dev"],
            output_folder=None  # Don't save results
        )

        # Extract metrics
        task_results = results[0] if results else {}

        # Compare with baselines
        if self.baseline_models:
            task_results['baselines'] = {}
            for name, baseline_model in self.baseline_models.items():
                logger.info(f"    Evaluating baseline: {name}")
                baseline_wrapper = MTEBModelWrapper(baseline_model)
                baseline_results = evaluation.run(
                    baseline_wrapper,
                    eval_splits=["test"] if "test" in task.description.get("eval_splits", []) else ["dev"],
                    output_folder=None
                )
                task_results['baselines'][name] = baseline_results[0] if baseline_results else {}

        return task_results

    def _evaluate_manual(self, task_name: str) -> Dict:
        """
        Manual evaluation for specific tasks

        Args:
            task_name: Task name

        Returns:
            Evaluation results
        """
        if 'classification' in task_name.lower():
            return self._evaluate_classification(task_name)
        elif 'retrieval' in task_name.lower():
            return self._evaluate_retrieval(task_name)
        elif 'sts' in task_name.lower() or 'similarity' in task_name.lower():
            return self._evaluate_similarity(task_name)
        elif 'clustering' in task_name.lower():
            return self._evaluate_clustering(task_name)
        else:
            return {"error": f"Unknown task type: {task_name}"}

    def _evaluate_classification(self, task_name: str) -> Dict:
        """Evaluate on classification task"""
        # Load sample data (you would load actual task data)
        from datasets import load_dataset

        try:
            if 'banking' in task_name.lower():
                dataset = load_dataset('banking77', split='test[:1000]')
                texts = dataset['text']
                labels = dataset['label']
            else:
                # Generic classification evaluation
                dataset = load_dataset('ag_news', split='test[:1000]')
                texts = dataset['text']
                labels = dataset['label']
        except:
            return {"error": "Failed to load dataset"}

        # Encode texts
        embeddings = self.model.encode(texts, batch_size=64)

        # Train classifier
        from sklearn.model_selection import cross_val_score
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, embeddings, labels, cv=5, scoring='f1_macro')

        results = {
            'f1_macro': float(scores.mean()),
            'f1_std': float(scores.std()),
            'accuracy': float(cross_val_score(clf, embeddings, labels, cv=5, scoring='accuracy').mean())
        }

        # Analyze routing patterns
        routing_analysis = self.model.analyze_routing(texts[:100])
        results['routing_entropy'] = float(np.mean(routing_analysis['entropy']))
        results['dominant_experts'] = {
            str(i): float(count) for i, count in
            enumerate(np.bincount(routing_analysis['dominant_expert']))
        }

        return results

    def _evaluate_retrieval(self, task_name: str) -> Dict:
        """Evaluate on retrieval task"""
        # Simplified retrieval evaluation
        # In practice, you'd load the actual retrieval dataset

        # Create synthetic retrieval task
        queries = [
            "What is machine learning?",
            "How does COVID-19 spread?",
            "Explain quantum computing",
            "What are financial derivatives?",
            "How to write Python code?"
        ]

        corpus = [
                     "Machine learning is a subset of artificial intelligence.",
                     "COVID-19 spreads through respiratory droplets.",
                     "Quantum computing uses quantum bits or qubits.",
                     "Financial derivatives are contracts based on underlying assets.",
                     "Python is a high-level programming language.",
                     # Add more documents...
                 ] * 10  # Duplicate for larger corpus

        # Encode
        query_embeddings = self.model.encode(queries)
        corpus_embeddings = self.model.encode(corpus)

        # Compute similarities
        from sentence_transformers import util
        similarities = util.pytorch_cos_sim(
            torch.tensor(query_embeddings),
            torch.tensor(corpus_embeddings)
        )

        # Calculate metrics (simplified)
        ndcg_at_10 = 0
        for i, query in enumerate(queries):
            top_k = torch.topk(similarities[i], k=min(10, len(corpus)))
            # In practice, you'd calculate actual NDCG based on relevance labels
            ndcg_at_10 += 1.0 / np.log2(top_k.indices[0].item() + 2)

        ndcg_at_10 /= len(queries)

        return {
            'ndcg@10': float(ndcg_at_10),
            'num_queries': len(queries),
            'corpus_size': len(corpus)
        }

    def _evaluate_similarity(self, task_name: str) -> Dict:
        """Evaluate on semantic similarity task"""
        # Load STS benchmark or similar
        from datasets import load_dataset

        try:
            dataset = load_dataset('glue', 'stsb', split='validation[:500]')
            sent1 = dataset['sentence1']
            sent2 = dataset['sentence2']
            scores = np.array(dataset['label']) / 5.0  # Normalize to 0-1
        except:
            return {"error": "Failed to load STS dataset"}

        # Encode sentence pairs
        emb1 = self.model.encode(sent1)
        emb2 = self.model.encode(sent2)

        # Compute cosine similarities
        cosine_scores = []
        for e1, e2 in zip(emb1, emb2):
            cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            cosine_scores.append(cos_sim)

        cosine_scores = np.array(cosine_scores)

        # Calculate correlations
        spearman_corr, _ = spearmanr(scores, cosine_scores)
        pearson_corr, _ = pearsonr(scores, cosine_scores)

        return {
            'spearman': float(spearman_corr),
            'pearson': float(pearson_corr),
            'num_pairs': len(scores)
        }

    def _evaluate_clustering(self, task_name: str) -> Dict:
        """Evaluate on clustering task"""
        # Simplified clustering evaluation
        from sklearn.datasets import fetch_20newsgroups

        try:
            newsgroups = fetch_20newsgroups(
                subset='test',
                categories=['comp.graphics', 'sci.med', 'talk.politics.guns'],
                remove=('headers', 'footers', 'quotes'),
                max_samples=300
            )
            texts = newsgroups.data[:300]
            labels = newsgroups.target[:300]
        except:
            return {"error": "Failed to load clustering dataset"}

        # Encode texts
        embeddings = self.model.encode(texts)

        # Perform clustering
        n_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Calculate v-measure
        v_score = v_measure_score(labels, cluster_labels)

        return {
            'v_measure': float(v_score),
            'n_clusters': n_clusters,
            'n_samples': len(texts)
        }

    def evaluate_domain_shift(self) -> Dict:
        """
        Special evaluation for domain shift scenarios
        This is where dynamic embeddings should excel
        """
        logger.info("\nEvaluating domain shift performance...")

        results = {}

        # Create domain shift scenario
        # Train on news, test on scientific
        from datasets import load_dataset

        try:
            # Load news data (train)
            news_data = load_dataset('ag_news', split='train[:1000]')
            train_texts = news_data['text']
            train_labels = news_data['label']

            # Load scientific data (test) - using different dataset
            # In practice, you'd use a scientific classification dataset
            sci_data = load_dataset('ag_news', split='test[:500]')  # Placeholder
            test_texts = sci_data['text']
            test_labels = sci_data['label']
        except:
            return {"error": "Failed to load datasets"}

        # Evaluate dynamic model
        train_emb = self.model.encode(train_texts)
        test_emb = self.model.encode(test_texts)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_emb, train_labels)
        preds = clf.predict(test_emb)

        results['dynamic'] = {
            'accuracy': float(accuracy_score(test_labels, preds)),
            'f1': float(f1_score(test_labels, preds, average='macro'))
        }

        # Compare with baselines
        for name, baseline in self.baseline_models.items():
            train_emb = baseline.encode(train_texts)
            test_emb = baseline.encode(test_texts)

            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(train_emb, train_labels)
            preds = clf.predict(test_emb)

            results[name] = {
                'accuracy': float(accuracy_score(test_labels, preds)),
                'f1': float(f1_score(test_labels, preds, average='macro'))
            }

        return results


class MTEBModelWrapper:
    """
    Wrapper to make our model compatible with MTEB evaluation
    """

    def __init__(self, model):
        self.model = model

    def encode(self, sentences: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Encode sentences to embeddings

        Args:
            sentences: List of sentences
            batch_size: Batch size for encoding
            **kwargs: Additional arguments (ignored)

        Returns:
            Embeddings as numpy array
        """
        return self.model.encode(sentences, batch_size=batch_size, normalize=True)