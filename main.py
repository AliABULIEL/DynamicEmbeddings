"""
Enhanced main entry point with all new methods integrated
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.evaluation.evaluator import Evaluator
from src.evaluation.baselines import MultiEmbeddingBaseline
from src.utils.logger import get_logger
from config.settings import RESULTS_DIR

logger = get_logger(__name__)


def run_enhanced_methods(evaluator, dataset_name, texts, labels):
    """
    Run all enhanced composition methods
    """
    results = {}
    clf = LogisticRegression(max_iter=1000, random_state=42)

    # 1. Top-K compositions
    for k in [2, 3, 4]:
        logger.info(f"Testing Top-{k} composition...")
        embeddings = []
        for text in texts:
            emb = evaluator.composer.compose_topk(text, k=k, method='weighted_sum')
            embeddings.append(emb)
        embeddings = np.array(embeddings)

        scores = cross_val_score(clf, embeddings, labels, cv=5, scoring='accuracy')
        results[f'composed_top{k}'] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
        logger.info(f"Top-{k}: {scores.mean():.4f} ± {scores.std():.4f}")

    # 2. Aligned composition
    logger.info("Testing aligned composition...")
    embeddings = []
    for text in texts:
        emb = evaluator.composer.compose_aligned(text)
        embeddings.append(emb)
    embeddings = np.array(embeddings)

    scores = cross_val_score(clf, embeddings, labels, cv=5, scoring='accuracy')
    results['composed_aligned'] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores.tolist()
    }
    logger.info(f"Aligned: {scores.mean():.4f} ± {scores.std():.4f}")

    # 3. Multi-embedding baseline
    logger.info("Testing multi-embedding baselines...")
    multi_baseline = MultiEmbeddingBaseline()

    for method in ['concat', 'average', 'max']:
        logger.info(f"Testing multi-embedding {method}...")
        embeddings = []

        # Use subset for concat due to 3x size
        test_texts = texts[:500] if method == 'concat' else texts
        test_labels = labels[:500] if method == 'concat' else labels

        for text in test_texts:
            emb = multi_baseline.get_multi_embedding(text, method=method)
            embeddings.append(emb)
        embeddings = np.array(embeddings)

        scores = cross_val_score(clf, embeddings, test_labels, cv=3, scoring='accuracy')
        results[f'multi_baseline_{method}'] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist(),
            'embedding_dim': embeddings.shape[1]
        }
        logger.info(f"Multi-{method}: {scores.mean():.4f} ± {scores.std():.4f} (dim={embeddings.shape[1]})")

    return results


def main(args):
    """
    Enhanced main function with all new methods
    """
    logger.info("=" * 80)
    logger.info("Starting Enhanced Domain-Based Embedding Composition Experiments")
    logger.info("=" * 80)

    # Initialize evaluator
    evaluator = Evaluator()

    # Results dictionary
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'composition_method': args.composition_method,
            'datasets': args.datasets,
            'run_ablation': args.ablation,
            'run_enhanced': args.enhanced,
            'use_topk': args.topk,
            'k_value': args.k_value
        },
        'results': {}
    }

    # Run classification experiments
    if 'classification' in args.experiments or 'all' in args.experiments:
        logger.info("\n" + "=" * 50)
        logger.info("CLASSIFICATION EXPERIMENTS")
        logger.info("=" * 50)

        for dataset in args.datasets:
            if dataset != 'stsb':  # Skip STS-B for classification
                logger.info(f"\nDataset: {dataset}")

                # Standard evaluation
                if args.use_topk:
                    # Use top-k composition instead of all domains
                    logger.info(f"Using Top-{args.k_value} composition")
                    results = evaluator.evaluate_classification_topk(
                        dataset_name=dataset,
                        k=args.k_value,
                        base_method=args.composition_method
                    )
                else:
                    # Original method
                    results = evaluator.evaluate_classification(
                        dataset_name=dataset,
                        composition_method=args.composition_method
                    )

                all_results['results'][f'classification_{dataset}'] = results

                # Run enhanced methods if requested
                if args.enhanced:
                    logger.info("\n" + "-" * 40)
                    logger.info("ENHANCED METHODS")
                    logger.info("-" * 40)

                    # Load data for enhanced methods
                    if dataset == 'ag_news':
                        texts, labels = evaluator.data_loader.load_ag_news()
                    elif dataset == 'dbpedia':
                        texts, labels = evaluator.data_loader.load_dbpedia()
                    else:
                        texts, labels = evaluator.data_loader.load_twenty_newsgroups()

                    enhanced_results = run_enhanced_methods(evaluator, dataset, texts, labels)
                    all_results['results'][f'enhanced_{dataset}'] = enhanced_results

    # Run similarity experiments
    if 'similarity' in args.experiments or 'all' in args.experiments:
        logger.info("\n" + "=" * 50)
        logger.info("SIMILARITY EXPERIMENTS")
        logger.info("=" * 50)

        if args.use_topk:
            # For similarity, use best single domain instead
            logger.info("Using best single domain for similarity")
            results = evaluator.evaluate_similarity_best_domain()
        else:
            results = evaluator.evaluate_similarity(
                composition_method=args.composition_method
            )
        all_results['results']['similarity'] = results

    # Run ablation studies
    if args.ablation:
        logger.info("\n" + "=" * 50)
        logger.info("ABLATION STUDIES")
        logger.info("=" * 50)

        ablation_results = evaluator.run_ablation_study()

        # Add top-k ablation
        if args.enhanced:
            logger.info("\nTop-K Ablation...")
            for k in [1, 2, 3, 4, 5]:
                logger.info(f"Testing k={k}")
                texts, labels = evaluator.data_loader.load_ag_news()
                embeddings = []
                for text in texts:
                    emb = evaluator.composer.compose_topk(text, k=k)
                    embeddings.append(emb)

                clf = LogisticRegression(max_iter=1000, random_state=42)
                scores = cross_val_score(clf, embeddings, labels, cv=3, scoring='accuracy')
                ablation_results[f'topk_{k}'] = {
                    'mean_accuracy': scores.mean(),
                    'std_accuracy': scores.std()
                }

        all_results['results']['ablation'] = ablation_results

    # Analyze domain influence
    if args.analyze:
        logger.info("\n" + "=" * 50)
        logger.info("DOMAIN INFLUENCE ANALYSIS")
        logger.info("=" * 50)

        analysis = evaluator.analyze_domain_influence()
        all_results['results']['domain_analysis'] = analysis

        # Print analysis
        for item in analysis:
            logger.info(f"\nText: {item['text']}")
            logger.info(f"Dominant domain: {item['dominant_domain']}")
            logger.info(f"Entropy: {item['entropy']:.3f}")
            logger.info("Domain probabilities:")
            for domain, prob in item['domain_probs'].items():
                logger.info(f"  {domain}: {prob:.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"results_{'enhanced_' if args.enhanced else ''}{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    # Print comprehensive summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # Find best performing method
    best_score = 0
    best_method = ""

    for exp_name, exp_results in all_results['results'].items():
        logger.info(f"\n{exp_name}:")
        if isinstance(exp_results, dict):
            for method, scores in exp_results.items():
                if isinstance(scores, dict) and 'mean' in scores:
                    score = scores['mean']
                    logger.info(f"  {method}: {score:.4f} ± {scores.get('std', 0):.4f}")

                    # Track best
                    if score > best_score and 'classification' in exp_name:
                        best_score = score
                        best_method = f"{exp_name}/{method}"

                elif isinstance(scores, dict) and 'spearman' in scores:
                    logger.info(f"  {method}: Spearman={scores['spearman']:.4f}, Pearson={scores['pearson']:.4f}")

    if best_method:
        logger.info("\n" + "=" * 80)
        logger.info(f"BEST PERFORMING METHOD: {best_method} ({best_score:.4f})")
        logger.info("=" * 80)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Domain Embedding Composition Experiments")

    parser.add_argument(
        '--composition-method',
        type=str,
        default='weighted_sum',
        choices=['weighted_sum', 'attention', 'max_pooling', 'learned_gate'],
        help='Base composition method to use'
    )

    parser.add_argument(
        '--experiments',
        nargs='+',
        default=['all'],
        choices=['all', 'classification', 'similarity'],
        help='Which experiments to run'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['ag_news', 'dbpedia'],
        choices=['ag_news', 'dbpedia', 'twenty_newsgroups', 'stsb'],
        help='Which datasets to use'
    )

    parser.add_argument(
        '--ablation',
        action='store_true',
        help='Run ablation studies'
    )

    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run domain influence analysis'
    )

    # New arguments for enhanced methods
    parser.add_argument(
        '--enhanced',
        action='store_true',
        help='Run enhanced methods (top-k, aligned, multi-baseline)'
    )

    parser.add_argument(
        '--use-topk',
        action='store_true',
        help='Use top-k composition instead of all domains'
    )

    parser.add_argument(
        '--k-value',
        type=int,
        default=2,
        help='Number of top domains to use (default: 2)'
    )

    args = parser.parse_args()

    # Run experiments
    results = main(args)