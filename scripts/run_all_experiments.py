"""
Run all experiments with different configurations
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import json
from pathlib import Path
from datetime import datetime
from config.settings import RESULTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_experiment(composition_method, dataset, experiment_type):
    """Run a single experiment configuration"""
    cmd = [
        "python", "main.py",
        "--composition-method", composition_method,
        "--datasets", dataset,
        "--experiments", experiment_type
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Experiment failed: {result.stderr}")
        return None

    return result.stdout


def main():
    """Run all experiment configurations"""
    logger.info("=" * 60)
    logger.info("RUNNING ALL EXPERIMENTS")
    logger.info("=" * 60)

    # Define all configurations to test
    # Define all configurations to test
    configurations = [
        # Test all composition methods on AG News
        ("weighted_sum", "ag_news", "classification"),
        ("weighted_sum", "ag_news", "similarity"),
        ("weighted_sum", "dbpedia", "similarity"),
        ("attention", "ag_news", "classification"),
        ("max_pooling", "ag_news", "classification"),
        ("learned_gate", "ag_news", "classification"),

        # Test all composition methods on DBPedia (ADD THESE)
        ("weighted_sum", "dbpedia", "classification"),
        ("attention", "dbpedia", "classification"),
        ("max_pooling", "dbpedia", "classification"),
        ("learned_gate", "dbpedia", "classification"),

        # Test on 20 newsgroups (ADD THIS)
        ("weighted_sum", "twenty_newsgroups", "classification"),

        # Similarity tests (already there but wrong dataset)
        ("weighted_sum", "stsb", "similarity"),  # Fix: should be stsb not ag_news

        # Run with ablation
        ("weighted_sum", "ag_news", "all"),
    ]

    all_results = {
        "experiment_suite": "complete_evaluation",
        "timestamp": datetime.now().isoformat(),
        "configurations": []
    }

    for comp_method, dataset, exp_type in configurations:
        logger.info(f"\n" + "-" * 40)
        logger.info(f"Configuration: {comp_method} / {dataset} / {exp_type}")
        logger.info("-" * 40)

        output = run_experiment(comp_method, dataset, exp_type)

        if output:
            all_results["configurations"].append({
                "composition_method": comp_method,
                "dataset": dataset,
                "experiment_type": exp_type,
                "status": "success"
            })
        else:
            all_results["configurations"].append({
                "composition_method": comp_method,
                "dataset": dataset,
                "experiment_type": exp_type,
                "status": "failed"
            })

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = RESULTS_DIR / f"experiment_summary_{timestamp}.json"

    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"✓ All experiments completed!")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()