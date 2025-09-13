"""
Enhanced main entry point with all improvements integrated
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
from config.settings import RESULTS_DIR, OPTIMAL_K_FOR_CLASSIFICATION, EMBEDDING_DIM

logger = get_logger(__name__)


def run_enhanced_methods(evaluator, dataset_name, texts, labels):
    """
    Run all enhanced composition methods with fixes
    """
    results = {}
    clf = LogisticRegression(max_iter=1000, random_state=42)

    # 1. Test optimal Top-K (use k=4 based on your results)
    logger.info(f"Testing optimal Top-{OPTIMAL_K_FOR_CLASSIFICATION} composition...")
    embeddings = []
    for text in texts:
        emb = evaluator.composer.compose_topk(text, k=OPTIMAL_K_FOR_CLASSIFICATION, method='weighted_sum')
        embeddings.append(emb)
    embeddings = np.array(embeddings)

    scores = cross_val_score(clf, embeddings, labels, cv=5, scoring='accuracy')
    results[f'composed_top{OPTIMAL_K_FOR_CLASSIFICATION}_optimal'] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores.tolist()
    }
    logger.info(f"Top-{OPTIMAL_K_FOR_CLASSIFICATION} (optimal): {scores.mean():.4f} ± {scores.std():.4f}")

    # 2. Test task-specific composition
    logger.info("Testing task-specific composition...")
    embeddings = []
    for text in texts:
        emb = evaluator.composer.compose_for_task(text, task='classification')
        embeddings.append(emb)
    embeddings = np.array(embeddings)

    scores = cross_val_score(clf, embeddings, labels, cv=5, scoring='accuracy')
    results['task_specific_classification'] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores.tolist()
    }
    logger.info(f"Task-specific: {scores.mean():.4f} ± {scores.std():.4f}")

    # 3. Test attention-based composition
    logger.info("Testing attention-based composition...")
    embeddings = []
    for text in texts[:500]:  # Subset for speed
        emb = evaluator.composer.attention_compose(text)
        embeddings.append(emb)
    embeddings = np.array(embeddings)

    scores = cross_val_score(clf, embeddings, labels[:500], cv=3, scoring='accuracy')
    results['composed_attention_new'] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores.tolist()
    }
    logger.info(f"Attention-based: {scores.mean():.4f} ± {scores.std():.4f}")

    # 4. Multi-baseline comparisons (fixed for dimension issues)
    logger.info("Testing multi-embedding baselines...")
    multi_baseline = MultiEmbeddingBaseline()

    for method in ['average', 'max', 'weighted']:  # Skip concat due to memory
        logger.info(f"Testing multi-embedding {method}...")
        embeddings = []

        test_texts = texts[:500]  # Use subset
        test_labels = labels[:500]

        for text in test_texts:
            emb = multi_baseline.get_multi_embedding(text, method=method)
            embeddings.append(emb)
        embeddings = np.array(embeddings)

        scores = cross_val_score(clf, embeddings, test_labels, cv=3, scoring='accuracy')
        results[f'multi_baseline_{method}'] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
        logger.info(f"Multi-{method}: {scores.mean():.4f} ± {scores.std():.4f}")

    return results


def main(args):
    """
    Main function with all improvements
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
            'use_topk': args.use_topk,
            'k_value': args.k_value,
            'use_task_specific': args.task_specific,
            'run_moe': args.moe
        },
        'results': {}
    }

    # Run classification experiments
    if 'classification' in args.experiments or 'all' in args.experiments:
        logger.info("\n" + "=" * 50)
        logger.info("CLASSIFICATION EXPERIMENTS")
        logger.info("=" * 50)

        for dataset in args.datasets:
            if dataset != 'stsb':
                logger.info(f"\nDataset: {dataset}")

                # Use improved evaluation with task-specific strategies
                if args.task_specific:
                    logger.info("Using task-specific strategies")
                    results = evaluator.evaluate_classification(
                        dataset_name=dataset,
                        composition_method=args.composition_method,
                        use_task_specific=True
                    )
                elif args.use_topk:
                    results = evaluator.evaluate_classification_topk(
                        dataset_name=dataset,
                        k=args.k_value,
                        base_method=args.composition_method
                    )
                else:
                    results = evaluator.evaluate_classification(
                        dataset_name=dataset,
                        composition_method=args.composition_method,
                        use_task_specific=False
                    )

                all_results['results'][f'classification_{dataset}'] = results

                # Run enhanced methods if requested
                if args.enhanced:
                    logger.info("\n" + "-" * 40)
                    logger.info("ENHANCED METHODS")
                    logger.info("-" * 40)

                    # Load data
                    if dataset == 'ag_news':
                        texts, labels = evaluator.data_loader.load_ag_news()
                    elif dataset == 'dbpedia':
                        texts, labels = evaluator.data_loader.load_dbpedia()
                    else:
                        texts, labels = evaluator.data_loader.load_twenty_newsgroups()

                    enhanced_results = run_enhanced_methods(evaluator, dataset, texts, labels)
                    all_results['results'][f'enhanced_{dataset}'] = enhanced_results

    # Run similarity experiments with fixes
    if 'similarity' in args.experiments or 'all' in args.experiments:
        logger.info("\n" + "=" * 50)
        logger.info("SIMILARITY EXPERIMENTS")
        logger.info("=" * 50)

        # Always use task-specific for similarity (single domain works better)
        results = evaluator.evaluate_similarity(
            composition_method=args.composition_method,
            use_task_specific=True  # Force task-specific for similarity
        )
        all_results['results']['similarity'] = results

    # Run MoE experiments if requested
    if args.moe:
        logger.info("\n" + "=" * 50)
        logger.info("MIXTURE OF EXPERTS EXPERIMENTS")
        logger.info("=" * 50)

        for dataset in args.datasets:
            if dataset != 'stsb':
                logger.info(f"\nMoE evaluation on {dataset}")
                moe_results = evaluator.evaluate_moe_classification(dataset)
                all_results['results'][f'moe_{dataset}'] = moe_results

        # MoE similarity
        logger.info("\nMoE similarity evaluation")
        moe_sim_results = evaluator.evaluate_moe_similarity()
        all_results['results']['moe_similarity'] = moe_sim_results

    # Run ablation studies
    if args.ablation:
        logger.info("\n" + "=" * 50)
        logger.info("ABLATION STUDIES")
        logger.info("=" * 50)

        ablation_results = evaluator.run_ablation_study()
        all_results['results']['ablation'] = ablation_results

    # Analyze domain influence
    if args.analyze:
        logger.info("\n" + "=" * 50)
        logger.info("DOMAIN INFLUENCE ANALYSIS")
        logger.info("=" * 50)

        analysis = evaluator.analyze_domain_influence()
        all_results['results']['domain_analysis'] = analysis

        for item in analysis:
            logger.info(f"\nText: {item['text']}")
            logger.info(f"Dominant domain: {item['dominant_domain']}")
            logger.info(f"Entropy: {item['entropy']:.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"results_{'enhanced_' if args.enhanced else ''}{'moe_' if args.moe else ''}{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    # Print comprehensive summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    best_score = 0
    best_method = ""

    for exp_name, exp_results in all_results['results'].items():
        logger.info(f"\n{exp_name}:")
        if isinstance(exp_results, dict):
            for method, scores in exp_results.items():
                if isinstance(scores, dict) and 'mean' in scores:
                    score = scores['mean']
                    logger.info(f"  {method}: {score:.4f} ± {scores.get('std', 0):.4f}")

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

    parser.add_argument(
        '--enhanced',
        action='store_true',
        help='Run enhanced methods'
    )

    parser.add_argument(
        '--use-topk',
        action='store_true',
        help='Use top-k composition'
    )

    parser.add_argument(
        '--k-value',
        type=int,
        default=4,  # Changed default to 4 based on your results
        help='Number of top domains to use (default: 4)'
    )

    parser.add_argument(
        '--task-specific',
        action='store_true',
        help='Use task-specific composition strategies'
    )

    parser.add_argument(
        '--moe',
        action='store_true',
        help='Run MoE experiments'
    )

    args = parser.parse_args()

    # Run experiments
    results = main(args)