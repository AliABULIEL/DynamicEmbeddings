"""Plotting module for TIDE-Lite results visualization.

This module generates plots for ablation studies, model comparisons,
and performance visualizations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style for consistent plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ResultPlotter:
    """Generate plots from aggregated results."""
    
    def __init__(
        self,
        summary_file: Union[str, Path],
        output_dir: Union[str, Path] = "results/figures",
    ) -> None:
        """Initialize plotter.
        
        Args:
            summary_file: Path to summary JSON file.
            output_dir: Directory to save plots.
        """
        self.summary_file = Path(summary_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load summary data
        with open(self.summary_file) as f:
            self.data = json.load(f)
        
        self.models = self.data.get("models", {})
        self.summary = self.data.get("summary", {})
    
    def plot_model_comparison(self) -> Path:
        """Plot comparison of all models across metrics.
        
        Returns:
            Path to saved figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # STS-B comparison
        ax = axes[0]
        model_names = []
        spearman_scores = []
        
        for name, model_data in self.models.items():
            if model_data.get("stsb"):
                model_names.append(name)
                spearman_scores.append(model_data["stsb"]["spearman"])
        
        if model_names:
            ax.bar(model_names, spearman_scores)
            ax.set_title("STS-B Spearman Correlation")
            ax.set_ylabel("Spearman ρ")
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
        
        # Quora comparison
        ax = axes[1]
        model_names = []
        ndcg_scores = []
        
        for name, model_data in self.models.items():
            if model_data.get("quora"):
                model_names.append(name)
                ndcg_scores.append(model_data["quora"]["ndcg_at_10"])
        
        if model_names:
            ax.bar(model_names, ndcg_scores)
            ax.set_title("Quora nDCG@10")
            ax.set_ylabel("nDCG@10")
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
        
        # Temporal comparison
        ax = axes[2]
        model_names = []
        consistency_scores = []
        
        for name, model_data in self.models.items():
            if model_data.get("temporal"):
                model_names.append(name)
                consistency_scores.append(model_data["temporal"]["consistency_score"])
        
        if model_names:
            ax.bar(model_names, consistency_scores)
            ax.set_title("Temporal Consistency Score")
            ax.set_ylabel("Consistency Score")
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_file = self.output_dir / "model_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved model comparison plot to {output_file}")
        return output_file
    
    def plot_latency_vs_quality(self) -> Path:
        """Plot latency vs quality trade-off.
        
        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        latencies = []
        qualities = []
        labels = []
        
        for name, model_data in self.models.items():
            if model_data.get("quora") and model_data.get("stsb"):
                latency = model_data["quora"].get("latency_median_ms", 0)
                quality = model_data["stsb"]["spearman"]
                
                if latency > 0:
                    latencies.append(latency)
                    qualities.append(quality)
                    labels.append(name)
        
        if latencies:
            ax.scatter(latencies, qualities, s=100)
            
            for i, label in enumerate(labels):
                ax.annotate(label, (latencies[i], qualities[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel("Median Query Latency (ms)")
            ax.set_ylabel("STS-B Spearman Correlation")
            ax.set_title("Latency vs Quality Trade-off")
            ax.grid(True, alpha=0.3)
        
        output_file = self.output_dir / "latency_vs_quality.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved latency vs quality plot to {output_file}")
        return output_file
    
    def plot_ablation_heatmap(self, ablation_dir: Optional[Path] = None) -> Optional[Path]:
        """Plot ablation study results as heatmap.
        
        Args:
            ablation_dir: Directory containing ablation results.
            
        Returns:
            Path to saved figure or None if no ablation data.
        """
        if ablation_dir is None:
            ablation_dir = Path("results/ablation")
        
        if not ablation_dir.exists():
            logger.warning(f"Ablation directory not found: {ablation_dir}")
            return None
        
        # Collect ablation results
        ablation_data = []
        
        for subdir in ablation_dir.glob("ablation_*"):
            if subdir.is_dir():
                # Parse configuration from directory name
                name_parts = subdir.name.split("_")
                config = {}
                
                for part in name_parts:
                    if part.startswith("mlp"):
                        config["mlp_hidden"] = int(part[3:])
                    elif part.startswith("w"):
                        config["weight"] = float(part[1:])
                    elif part.startswith("enc"):
                        config["encoding"] = part[3:]
                
                # Load metrics
                metrics_file = subdir / "metrics_stsb_*.json"
                metrics_files = list(subdir.glob("metrics_stsb_*.json"))
                
                if metrics_files:
                    with open(metrics_files[0]) as f:
                        metrics = json.load(f)
                    
                    config["spearman"] = metrics.get("spearman_correlation", 0)
                    ablation_data.append(config)
        
        if not ablation_data:
            logger.warning("No ablation results found")
            return None
        
        # Create DataFrame and pivot for heatmap
        df = pd.DataFrame(ablation_data)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create pivot table for heatmap
        pivot = df.pivot_table(
            values="spearman",
            index="mlp_hidden",
            columns="weight",
            aggfunc="mean"
        )
        
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
        ax.set_title("Ablation Study: MLP Hidden Size vs Consistency Weight")
        ax.set_xlabel("Consistency Weight (λ)")
        ax.set_ylabel("MLP Hidden Size")
        
        output_file = self.output_dir / "ablation_heatmap.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ablation heatmap to {output_file}")
        return output_file
    
    def plot_training_curves(self, training_logs: Optional[Path] = None) -> Optional[Path]:
        """Plot training curves if available.
        
        Args:
            training_logs: Path to training logs.
            
        Returns:
            Path to saved figure or None if no training data.
        """
        if training_logs is None:
            # Try to find training logs
            log_files = list(Path("results").glob("**/training_log.json"))
            if not log_files:
                logger.warning("No training logs found")
                return None
            training_logs = log_files[0]
        
        if not training_logs.exists():
            return None
        
        # Load training history
        with open(training_logs) as f:
            history = [json.loads(line) for line in f]
        
        if not history:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract metrics
        epochs = [h.get("epoch", i) for i, h in enumerate(history)]
        train_loss = [h.get("train_loss", 0) for h in history]
        val_loss = [h.get("val_loss", 0) for h in history]
        val_spearman = [h.get("val_spearman", 0) for h in history]
        learning_rate = [h.get("learning_rate", 0) for h in history]
        
        # Plot training loss
        ax = axes[0, 0]
        ax.plot(epochs, train_loss, label="Train Loss")
        if any(val_loss):
            ax.plot(epochs, val_loss, label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot validation metric
        ax = axes[0, 1]
        ax.plot(epochs, val_spearman)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Spearman Correlation")
        ax.set_title("Validation Performance")
        ax.grid(True, alpha=0.3)
        
        # Plot learning rate
        ax = axes[1, 0]
        ax.plot(epochs, learning_rate)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        
        # Plot loss components if available
        ax = axes[1, 1]
        if "cosine_loss" in history[0]:
            cosine_loss = [h.get("cosine_loss", 0) for h in history]
            temporal_loss = [h.get("temporal_loss", 0) for h in history]
            ax.plot(epochs, cosine_loss, label="Cosine Loss")
            ax.plot(epochs, temporal_loss, label="Temporal Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Loss Components")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "training_curves.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves to {output_file}")
        return output_file
    
    def plot_all(self) -> Dict[str, Path]:
        """Generate all plots.
        
        Returns:
            Dictionary mapping plot names to file paths.
        """
        plots = {}
        
        # Model comparison
        plots["model_comparison"] = self.plot_model_comparison()
        
        # Latency vs quality
        plots["latency_vs_quality"] = self.plot_latency_vs_quality()
        
        # Ablation heatmap (if available)
        ablation_plot = self.plot_ablation_heatmap()
        if ablation_plot:
            plots["ablation_heatmap"] = ablation_plot
        
        # Training curves (if available)
        training_plot = self.plot_training_curves()
        if training_plot:
            plots["training_curves"] = training_plot
        
        logger.info(f"Generated {len(plots)} plots")
        return plots


def generate_plots(
    summary_file: Union[str, Path],
    output_dir: Union[str, Path] = "results/figures",
    dry_run: bool = False,
) -> Dict[str, Path]:
    """Generate plots from summary data.
    
    Args:
        summary_file: Path to summary JSON file.
        output_dir: Directory to save plots.
        dry_run: If True, only show plan without generating.
        
    Returns:
        Dictionary mapping plot names to file paths.
    """
    if dry_run:
        logger.info("[DRY RUN] Would generate plots:")
        logger.info(f"  Input: {summary_file}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info("  Plots:")
        logger.info("    - model_comparison.png")
        logger.info("    - latency_vs_quality.png")
        logger.info("    - ablation_heatmap.png")
        logger.info("    - training_curves.png")
        return {}
    
    plotter = ResultPlotter(summary_file, output_dir)
    return plotter.plot_all()
