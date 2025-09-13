# scripts/test_enhancements.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.embedding_composer import EmbeddingComposer
from src.evaluation.baselines import MultiEmbeddingBaseline
from src.data.dataset_loader import DatasetLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np


def test_all_approaches():
    """Test all enhancement approaches"""

    print("=" * 60)
    print("TESTING ENHANCED APPROACHES")
    print("=" * 60)

    # Load data
    loader = DatasetLoader(sample_size=1000)
    texts, labels = loader.load_ag_news()

    # Initialize models
    composer = EmbeddingComposer()
    multi_baseline = MultiEmbeddingBaseline()

    results = {}

    # 1. Original approach
    print("\n1. Original Weighted Sum (all 5 domains):")
    embeddings = composer.compose_batch(texts, method='weighted_sum')
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, embeddings, labels, cv=5)
    results['original'] = scores.mean()
    print(f"   Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # 2. Top-2 domains only
    print("\n2. Top-2 Domains Only:")
    embeddings = [composer.compose_topk(text, k=2) for text in texts]
    scores = cross_val_score(clf, embeddings, labels, cv=5)
    results['top2'] = scores.mean()
    print(f"   Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # 3. Top-3 domains
    print("\n3. Top-3 Domains:")
    embeddings = [composer.compose_topk(text, k=3) for text in texts]
    scores = cross_val_score(clf, embeddings, labels, cv=5)
    results['top3'] = scores.mean()
    print(f"   Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # 4. Aligned embeddings
    print("\n4. Aligned Embeddings:")
    embeddings = [composer.compose_aligned(text) for text in texts[:100]]  # Test on subset
    scores = cross_val_score(clf, embeddings, labels[:100], cv=3)
    results['aligned'] = scores.mean()
    print(f"   Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # 5. Multi-embedding baseline (concatenated)
    print("\n5. Multi-Embedding Baseline (Concatenated):")
    embeddings = [multi_baseline.get_multi_embedding(text, 'concat') for text in texts[:100]]
    scores = cross_val_score(clf, embeddings, labels[:100], cv=3)
    results['multi_concat'] = scores.mean()
    print(f"   Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # 6. Multi-embedding baseline (averaged)
    print("\n6. Multi-Embedding Baseline (Averaged):")
    embeddings = [multi_baseline.get_multi_embedding(text, 'average') for text in texts[:100]]
    scores = cross_val_score(clf, embeddings, labels[:100], cv=3)
    results['multi_avg'] = scores.mean()
    print(f"   Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # 7. Single best domain
    print("\n7. Single Best Domain (News):")
    embeddings = [composer.embedder_manager.get_embedding(text, 'news') for text in texts]
    scores = cross_val_score(clf, embeddings, labels, cv=5)
    results['best_single'] = scores.mean()
    print(f"   Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Sorted by Performance:")
    print("=" * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for method, score in sorted_results:
        print(f"{method:20s}: {score:.4f}")

    # Identify winner
    best_method = sorted_results[0]
    print(f"\nBest approach: {best_method[0]} ({best_method[1]:.4f})")

    improvement = best_method[1] - results['original']
    if improvement > 0:
        print(f"Improvement over original: +{improvement:.4f}")
    else:
        print(f"Original is still best")


if __name__ == "__main__":
    test_all_approaches()