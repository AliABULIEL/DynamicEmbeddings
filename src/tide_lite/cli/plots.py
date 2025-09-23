"""Generate plots from aggregated metrics.

This module creates visualization plots for metrics comparison,
ablation studies, and performance analysis.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


class MetricsPlotter:
    """Generate plots from metrics summary."""
    
    def __init__(self, summary_path: Path, output_dir: Path = Path("results/figures")) -> None:
        """Initialize plotter.
        
        Args:
            summary_path: Path to summary.json file.
            output_dir: Directory to save plots.
        """
        self.summary_path = Path(summary_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path) as f:
            self.summary = json.load(f)
    
    def plot_model_comparison(self) -> None:
        """Plot comparison of all models across tasks."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = list(self.summary["models"].keys())
        
        # STS-B Spearman
        ax = axes[0]
        spearman_scores = [
            self.summary["models"][m].get("stsb", {}).get("spearman", 0)
            for m in models
        ]
        ax.bar(models, spearman_scores)
        ax.set_title("STS-B Spearman Correlation")
        ax.set_ylabel("Spearman ρ")
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', rotation=45)
        
        # Quora nDCG@10
        ax = axes[1]
        ndcg_scores = [
            self.summary["models"][m].get("quora", {}).get("ndcg_at_10", 0)
            for m in models
        ]
        ax.bar(models, ndcg_scores)
        ax.set_title("Quora Retrieval nDCG@10")
        ax.set_ylabel("nDCG@10")
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', rotation=45)
        
        # Temporal Consistency
        ax = axes[2]
        consistency_scores = [
            self.summary["models"][m].get("temporal", {}).get("consistency_score", 0)
            for m in models
        ]
        ax.bar(models, consistency_scores)
        ax.set_title("Temporal Consistency Score")
        ax.set_ylabel("Consistency")
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / "model_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved model comparison plot to {output_path}")
    
    def plot_ablation_heatmap(self) -> None:
        """Plot ablation study results as heatmap."""
        if not self.summary.get("ablations"):
            logger.warning("No ablation results to plot")
            return
        
        # Extract ablation data
        ablation_data = []
        for name, data in self.summary["ablations"].items():
            config = data.get("config", {})
            metrics = data.get("metrics", {})
            
            ablation_data.append({
                "mlp_hidden": config.get("time_mlp_hidden", 0),
                "consistency_weight": config.get("consistency_weight", 0),
                "spearman": metrics.get("spearman_rho", 0),
            })
        
        if not ablation_data:
            return
        
        # Create pivot table
        df = pd.DataFrame(ablation_data)
        pivot = df.pivot_table(
            index="consistency_weight",
            columns="mlp_hidden",
            values="spearman",
        )
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={'label': 'Spearman ρ'})
        plt.title("Ablation Study: Hyperparameter Impact on STS-B Performance")
        plt.xlabel("MLP Hidden Dimension")
        plt.ylabel("Consistency Weight (λ)")
        
        output_path = self.output_dir / "ablation_heatmap.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved ablation heatmap to {output_path}")
    
    def plot_latency_vs_quality(self) -> None:
        """Plot latency vs quality trade-off."""
        models = list(self.summary["models"].keys())
        
        latencies = []
        qualities = []
        labels = []
        
        for model in models:
            metrics = self.summary["models"][model]
            
            # Get latency (from Quora retrieval)
            latency = metrics.get("quora", {}).get("latency_median_ms", 0)
            
            # Get quality (average of normalized metrics)
            spearman = metrics.get("stsb", {}).get("spearman", 0)
            ndcg = metrics.get("quora", {}).get("ndcg_at_10", 0)
            consistency = metrics.get("temporal", {}).get("consistency_score", 0)
            
            if latency > 0 and (spearman > 0 or ndcg > 0):
                quality = np.mean([spearman, ndcg, consistency])
                latencies.append(latency)
                qualities.append(quality)
                labels.append(model)
        
        if not latencies:
            logger.warning("No latency data to plot")
            return
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(latencies, qualities, s=100)
        
        # Add labels
        for i, label in enumerate(labels):
            plt.annotate(label, (latencies[i], qualities[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel("Median Latency (ms)")
        plt.ylabel("Average Quality Score")
        plt.title("Latency vs Quality Trade-off")
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / "latency_vs_quality.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved latency vs quality plot to {output_path}")
    
    def plot_all(self) -> None:
        """Generate all plots."""
        self.plot_model_comparison()
        self.plot_ablation_heatmap()
        self.plot_latency_vs_quality()
        logger.info(f"All plots saved to {self.output_dir}")


def main() -> None:
    """Command-line interface for plotting."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate plots from metrics")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/summary.json"),
        help="Input summary JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show plan without executing (default)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually generate plots",
    )
    
    args = parser.parse_args()
    
    if not args.run:
        print("[DRY RUN] Would generate plots from:", args.input)
        print("[DRY RUN] Would save plots to:", args.output_dir)
        print("[DRY RUN] Plots to generate:")
        print("  - model_comparison.png")
        print("  - ablation_heatmap.png")
        print("  - latency_vs_quality.png")
        return
    
    plotter = MetricsPlotter(args.input, args.output_dir)
    plotter.plot_all()


if __name__ == "__main__":
    main()
