"""
Simplified main - only run what works
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
    Simplified main - only proven methods
    """
    logger.info("=" * 80)
    logger.info("Starting Simplified Embedding Experiments")
    logger.info("=" * 80)

    evaluator = Evaluator()

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'datasets': args.datasets,
            'experiments': args.experiments
        },
        'results': {}
    }

    # Classification experiments
    if 'classification' in args.experiments or 'all' in args.experiments:
        logger.info("\nCLASSIFICATION EXPERIMENTS")
        logger.info("=" * 50)

        for dataset in args.datasets:
            if dataset != 'stsb':
                logger.info(f"\nDataset: {dataset}")
                results = evaluator.evaluate_classification(dataset)
                all_results['results'][f'classification_{dataset}'] = results

    # Similarity experiments
    if 'similarity' in args.experiments or 'all' in args.experiments:
        logger.info("\nSIMILARITY EXPERIMENTS")
        logger.info("=" * 50)
        results = evaluator.evaluate_similarity()
        all_results['results']['similarity'] = results

    # Simple ablation
    if args.ablation:
        logger.info("\nABLATION STUDY")
        logger.info("=" * 50)
        ablation_results = evaluator.run_ablation_study()
        all_results['results']['ablation'] = ablation_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    for exp_name, exp_results in all_results['results'].items():
        logger.info(f"\n{exp_name}:")
        if isinstance(exp_results, dict):
            for method, scores in exp_results.items():
                if isinstance(scores, dict):
                    if 'mean' in scores:
                        logger.info(f"  {method}: {scores['mean']:.4f}")
                    elif 'spearman' in scores:
                        logger.info(f"  {method}: Spearman={scores['spearman']:.4f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified Embedding Experiments")

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
        default=['ag_news'],  # Default to just ag_news for speed
        choices=['ag_news', 'dbpedia', 'stsb'],
        help='Which datasets to use'
    )

    parser.add_argument(
        '--ablation',
        action='store_true',
        help='Run simple ablation'
    )

    args = parser.parse_args()
    results = main(args)