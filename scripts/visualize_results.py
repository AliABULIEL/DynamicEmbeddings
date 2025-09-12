"""
Visualize experiment results
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from config.settings import RESULTS_DIR
import numpy as np

sns.set_style("whitegrid")


def load_latest_results():
    """Load the most recent results file"""
    results_files = list(RESULTS_DIR.glob("results_*.json"))

    if not results_files:
        raise FileNotFoundError("No results files found")

    latest_file = max(results_files, key=lambda x: x.stat().st_mtime)

    with open(latest_file, 'r') as f:
        return json.load(f), latest_file


def plot_classification_comparison(results):
    """Create bar plot comparing classification accuracies"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract classification results
    classification_data = {}

    for key, value in results.items():
        if 'classification' in key:
            dataset = key.replace('classification_', '')
            classification_data[dataset] = value

    if not classification_data:
        print("No classification results found")
        return

    # Prepare data for plotting
    methods = []
    scores = []
    datasets = []

    for dataset, dataset_results in classification_data.items():
        for method, method_scores in dataset_results.items():
            if isinstance(method_scores, dict) and 'mean' in method_scores:
                methods.append(method)
                scores.append(method_scores['mean'])
                datasets.append(dataset)

    # Create DataFrame
    df = pd.DataFrame({
        'Method': methods,
        'Accuracy': scores,
        'Dataset': datasets
    })

    # Plot
    sns.barplot(data=df, x='Method', y='Accuracy', hue='Dataset', ax=ax)
    ax.set_title('Classification Performance Comparison')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Method')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save
    output_file = RESULTS_DIR / 'classification_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.show()


def plot_similarity_comparison(results):
    """Create bar plot for similarity task results"""
    if 'similarity' not in results:
        print("No similarity results found")
        return

    similarity_data = results['similarity']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Prepare data
    methods = []
    spearman_scores = []
    pearson_scores = []

    for method, scores in similarity_data.items():
        if isinstance(scores, dict) and 'spearman' in scores:
            methods.append(method)
            spearman_scores.append(scores['spearman'])
            pearson_scores.append(scores['pearson'])

    # Plot Spearman
    ax1.bar(methods, spearman_scores)
    ax1.set_title('Spearman Correlation (STS-B)')
    ax1.set_ylabel('Correlation')
    ax1.set_xlabel('Method')
    ax1.set_ylim([0, 1])
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot Pearson
    ax2.bar(methods, pearson_scores)
    ax2.set_title('Pearson Correlation (STS-B)')
    ax2.set_ylabel('Correlation')
    ax2.set_xlabel('Method')
    ax2.set_ylim([0, 1])
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save
    output_file = RESULTS_DIR / 'similarity_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.show()


def plot_ablation_results(results):
    """Plot ablation study results"""
    if 'ablation' not in results:
        print("No ablation results found")
        return

    ablation_data = results['ablation']

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = []
    accuracies = []
    errors = []

    for method, scores in ablation_data.items():
        if 'mean_accuracy' in scores:
            methods.append(method)
            accuracies.append(scores['mean_accuracy'])
            errors.append(scores.get('std_accuracy', 0))

    # Plot with error bars
    ax.bar(methods, accuracies, yerr=errors, capsize=5)
    ax.set_title('Ablation Study: Composition Methods')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Composition Method')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Save
    output_file = RESULTS_DIR / 'ablation_study.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.show()


def plot_domain_analysis(results):
    """Visualize domain probability distributions"""
    if 'domain_analysis' not in results:
        print("No domain analysis found")
        return

    analysis = results['domain_analysis']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, item in enumerate(analysis[:6]):
        ax = axes[idx]

        # Get domain probabilities
        domains = list(item['domain_probs'].keys())
        probs = list(item['domain_probs'].values())

        # Plot
        ax.bar(domains, probs)
        ax.set_title(f"Text {idx + 1}\nEntropy: {item['entropy']:.2f}")
        ax.set_ylabel('Probability')
        ax.set_ylim([0, 1])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle('Domain Probability Distributions')
    plt.tight_layout()

    # Save
    output_file = RESULTS_DIR / 'domain_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.show()


def create_summary_table(results):
    """Create a summary table of all results"""
    summary = []

    # Classification results
    for key, value in results.items():
        if 'classification' in key:
            dataset = key.replace('classification_', '')
            for method, scores in value.items():
                if isinstance(scores, dict) and 'mean' in scores:
                    summary.append({
                        'Task': 'Classification',
                        'Dataset': dataset,
                        'Method': method,
                        'Score': f"{scores['mean']:.4f} ± {scores.get('std', 0):.4f}"
                    })

    # Similarity results
    if 'similarity' in results:
        for method, scores in results['similarity'].items():
            if isinstance(scores, dict) and 'spearman' in scores:
                summary.append({
                    'Task': 'Similarity',
                    'Dataset': 'STS-B',
                    'Method': method,
                    'Score': f"S:{scores['spearman']:.4f} P:{scores['pearson']:.4f}"
                })

    df = pd.DataFrame(summary)

    # Save as CSV
    output_file = RESULTS_DIR / 'results_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved summary table to {output_file}")

    # Display
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))

    return df


def main():
    """Main visualization function"""
    print("Loading results...")

    try:
        data, filepath = load_latest_results()
        print(f"Loaded results from: {filepath}")

        results = data.get('results', {})

        # Create all visualizations
        plot_classification_comparison(results)
        plot_similarity_comparison(results)
        plot_ablation_results(results)
        plot_domain_analysis(results)

        # Create summary table
        create_summary_table(results)

        print("\n✓ All visualizations created successfully!")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()