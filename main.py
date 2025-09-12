"""
Main entry point for running experiments
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
from src.evaluation.evaluator import Evaluator
from src.utils.logger import get_logger
from config.settings import RESULTS_DIR

logger = get_logger(__name__)


def main(args):
    """
    Main function to run experiments
    """
    logger.info("=" * 80)
    logger.info("Starting Domain-Based Embedding Composition Experiments")
    logger.info("=" * 80)

    # Initialize evaluator
    evaluator = Evaluator()

    # Results dictionary
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'composition_method': args.composition_method,
            'datasets': args.datasets,
            'run_ablation': args.ablation
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
                results = evaluator.evaluate_classification(
                    dataset_name=dataset,
                    composition_method=args.composition_method
                )
                all_results['results'][f'classification_{dataset}'] = results

    # Run similarity experiments
    if 'similarity' in args.experiments or 'all' in args.experiments:
        logger.info("\n" + "=" * 50)
        logger.info("SIMILARITY EXPERIMENTS")
        logger.info("=" * 50)

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
    results_file = RESULTS_DIR / f"results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    for exp_name, exp_results in all_results['results'].items():
        logger.info(f"\n{exp_name}:")
        if isinstance(exp_results, dict):
            for method, scores in exp_results.items():
                if isinstance(scores, dict) and 'mean' in scores:
                    logger.info(f"  {method}: {scores['mean']:.4f} ± {scores.get('std', 0):.4f}")
                elif isinstance(scores, dict) and 'spearman' in scores:
                    logger.info(f"  {method}: Spearman={scores['spearman']:.4f}, Pearson={scores['pearson']:.4f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Embedding Composition Experiments")

    parser.add_argument(
        '--composition-method',
        type=str,
        default='weighted_sum',
        choices=['weighted_sum', 'attention', 'max_pooling', 'learned_gate'],
        help='Composition method to use'
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

    args = parser.parse_args()

    # Run experiments
    results = main(args)