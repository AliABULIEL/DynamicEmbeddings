"""
Quick test script to verify everything is working
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.domain_classifier import DomainClassifier
from src.models.domain_embedders import DomainEmbedderManager
from src.models.embedding_composer import EmbeddingComposer
from src.utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)


def test_domain_classifier():
    """Test domain classification"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Domain Classifier")
    logger.info("=" * 50)

    classifier = DomainClassifier()

    test_texts = [
        "The COVID-19 vaccine showed 95% efficacy in clinical trials",
        "Breaking: Stock market hits all-time high",
        "The court ruled in favor of the defendant",
        "LOL this is so funny 😂 #viral",
        "Quantum computing breakthrough announced by researchers"
    ]

    for text in test_texts:
        probs = classifier.classify(text, return_dict=True)
        dominant = classifier.get_dominant_domain(text)

        logger.info(f"\nText: {text[:60]}...")
        logger.info(f"Dominant domain: {dominant[0]} ({dominant[1]:.3f})")
        logger.info("All probabilities:")
        for domain, prob in probs.items():
            logger.info(f"  {domain:10s}: {prob:.3f}")


def test_embedders():
    """Test domain embedders"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Domain Embedders")
    logger.info("=" * 50)

    manager = DomainEmbedderManager(load_all=False)

    test_text = "Machine learning revolutionizes healthcare"

    # Test loading individual models
    for domain in ['scientific', 'medical']:
        logger.info(f"\nTesting {domain} embedder...")
        embedding = manager.get_embedding(test_text, domain)
        logger.info(f"  Embedding shape: {embedding.shape}")
        logger.info(f"  Embedding norm: {np.linalg.norm(embedding):.3f}")


def test_composer():
    """Test embedding composition"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Embedding Composer")
    logger.info("=" * 50)

    composer = EmbeddingComposer()

    test_texts = [
        "The patient's blood pressure medication was adjusted",
        "Apple announces new AI features in iPhone",
        "Supreme Court hearing on constitutional rights"
    ]

    for text in test_texts:
        logger.info(f"\nText: {text}")

        # Test different composition methods
        for method in ['weighted_sum', 'max_pooling']:
            embedding, probs, domain_embs = composer.compose(
                text,
                method=method,
                return_details=True
            )

            logger.info(f"\n  Method: {method}")
            logger.info(f"  Final embedding shape: {embedding.shape}")
            logger.info(f"  Final embedding norm: {np.linalg.norm(embedding):.3f}")
            logger.info(f"  Domain probabilities:")
            for i, domain in enumerate(composer.domains):
                logger.info(f"    {domain:10s}: {probs[i]:.3f}")


def test_end_to_end():
    """Test complete pipeline"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing End-to-End Pipeline")
    logger.info("=" * 50)

    from src.evaluation.baselines import BaselineModels

    composer = EmbeddingComposer()
    baselines = BaselineModels(load_all=False)

    test_text = "COVID-19 vaccine research published in medical journal"

    # Get composed embedding
    composed_emb = composer.compose(test_text, method='weighted_sum')

    # Get baseline embedding
    baseline_emb = baselines.get_embedding(test_text, 'bert-base')

    # Compare
    logger.info(f"\nTest text: {test_text}")
    logger.info(f"Composed embedding shape: {composed_emb.shape}")
    logger.info(f"Baseline embedding shape: {baseline_emb.shape}")

    # Calculate cosine similarity between methods
    cos_sim = np.dot(composed_emb, baseline_emb) / (
            np.linalg.norm(composed_emb) * np.linalg.norm(baseline_emb)
    )
    logger.info(f"Cosine similarity between composed and baseline: {cos_sim:.3f}")


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("RUNNING QUICK TESTS")
    logger.info("=" * 60)

    try:
        test_domain_classifier()
        test_embedders()
        test_composer()
        test_end_to_end()

        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n✗ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()