import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings first
from src.utils.logger import suppress_warnings
suppress_warnings()

from src.models.domain_classifier import DomainClassifier
from src.models.domain_embedders import DomainEmbedderManager
from src.models.embedding_composer import EmbeddingComposer
from src.evaluation.baselines import BaselineModels
import numpy as np
import time

# Simple print functions for clean output
def print_header(text):
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def print_section(text):
    print(f"\n▶ {text}")
    print("-"*40)

def print_result(label, value):
    print(f"  {label:20s}: {value}")

def format_probs(probs, domains):
    """Format domain probabilities nicely"""
    result = []
    for domain, prob in zip(domains, probs):
        if prob > 0.1:  # Only show significant probabilities
            result.append(f"{domain}({prob:.1%})")
    return " | ".join(result)


def test_domain_classifier():
    """Test domain classification"""
    print_header("TESTING DOMAIN CLASSIFIER")

    print("Loading classifier...")
    start = time.time()
    classifier = DomainClassifier()
    print(f"✓ Loaded in {time.time()-start:.1f}s")

    test_cases = [
        ("The COVID-19 vaccine showed 95% efficacy", "medical"),
        ("Breaking: Stock market hits record high", "news"),
        ("Court rules in favor of defendant", "legal"),
        ("LOL this is hilarious 😂 #viral", "social"),
        ("Quantum computing breakthrough", "scientific/news")
    ]

    print_section("Classification Results")
    for text, expected in test_cases:
        probs = classifier.classify(text)
        dominant = classifier.get_dominant_domain(text)

        print(f"\n'{text[:40]}...'")
        print(f"  Expected: {expected}")
        print(f"  Result: {dominant[0]} ({dominant[1]:.1%})")
        print(f"  All: {format_probs(probs, classifier.domains)}")

    return True


def test_embedders():
    """Test domain embedders"""
    print_header("TESTING DOMAIN EMBEDDERS")

    manager = DomainEmbedderManager(load_all=False)
    test_text = "Machine learning revolutionizes healthcare"

    print_section("Testing Individual Embedders")

    for domain in ['scientific', 'medical']:
        print(f"\n{domain.upper()} embedder:")
        start = time.time()
        embedding = manager.get_embedding(test_text, domain)
        load_time = time.time() - start

        print_result("Load + embed time", f"{load_time:.2f}s")
        print_result("Embedding shape", str(embedding.shape))
        print_result("Embedding norm", f"{np.linalg.norm(embedding):.2f}")
        print_result("Non-zero values", f"{np.count_nonzero(embedding)}/{len(embedding)}")

    return True


def test_composer():
    """Test embedding composition"""
    print_header("TESTING EMBEDDING COMPOSER")

    print("Initializing composer (this loads all 5 models)...")
    start = time.time()
    composer = EmbeddingComposer()
    print(f"✓ Initialized in {time.time()-start:.1f}s")

    test_cases = [
        "FDA approves new cancer drug after trials",  # Medical + Legal
        "Apple stock soars on AI announcement",        # News + Tech
        "Climate change research published in Nature"  # Scientific + News
    ]

    print_section("Composition Results")

    for text in test_cases:
        print(f"\n'{text}'")

        # Get detailed results
        embedding, probs, domain_embs = composer.compose(
            text,
            method='weighted_sum',
            return_details=True
        )

        # Show domain distribution
        print(f"  Domains: {format_probs(probs, composer.domains)}")

        # Show embedding info
        print_result("Final embedding norm", f"{np.linalg.norm(embedding):.2f}")

        # Compare methods
        emb_weighted = embedding
        emb_max = composer.compose(text, method='max_pooling')

        similarity = np.dot(emb_weighted, emb_max) / (
            np.linalg.norm(emb_weighted) * np.linalg.norm(emb_max)
        )
        print_result("Weighted vs Max sim", f"{similarity:.3f}")

    return True


def test_comparison():
    """Compare composed vs baseline embeddings"""
    print_header("COMPARING COMPOSED VS BASELINE")

    composer = EmbeddingComposer()
    baselines = BaselineModels(load_all=False)

    test_text = "COVID vaccine research published in medical journal"

    print_section(f"Text: '{test_text}'")

    # Get embeddings
    composed = composer.compose(test_text)
    baseline = baselines.get_embedding(test_text, 'bert-base')

    # Analysis
    print_result("Composed norm", f"{np.linalg.norm(composed):.2f}")
    print_result("Baseline norm", f"{np.linalg.norm(baseline):.2f}")

    cos_sim = np.dot(composed, baseline) / (
        np.linalg.norm(composed) * np.linalg.norm(baseline)
    )
    print_result("Cosine similarity", f"{cos_sim:.3f}")

    # Component analysis
    print("\n  Component statistics:")
    print_result("Composed mean", f"{composed.mean():.4f}")
    print_result("Composed std", f"{composed.std():.4f}")
    print_result("Baseline mean", f"{baseline.mean():.4f}")
    print_result("Baseline std", f"{baseline.std():.4f}")

    return True


def main():
    """Run all tests with clean output"""
    print("\n" + "🚀 "*20)
    print("DOMAIN EMBEDDING COMPOSITION - SYSTEM TEST")
    print("🚀 "*20)

    # Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("\n⚠ No GPU - using CPU (will be slower)")

    try:
        # Run tests
        test_domain_classifier()
        test_embedders()
        test_composer()
        test_comparison()

        print("\n" + "="*60)
        print(" ✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        print("\nYour system is ready for experiments!")
        print("Next step: python main.py --experiments classification --datasets ag_news")

    except Exception as e:
        print("\n" + "="*60)
        print(f" ❌ TEST FAILED: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()