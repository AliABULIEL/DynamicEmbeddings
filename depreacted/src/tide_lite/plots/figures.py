"""Figure generation utilities for TIDE-Lite experiments.

This module creates static plots for analyzing experimental results
including ablation studies, model comparisons, and performance metrics.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


@dataclass
class PlotConfig:
    """Configuration for plot generation.
    
    Attributes:
        figure_size: Figure size as (width, height) in inches.
        dpi: Dots per inch for saved figures.
        style: Matplotlib/seaborn style.
        color_palette: Color palette name or list.
        font_size: Base font size.
        save_format: Output format (png, pdf, svg).
    """
    figure_size: Tuple[float, float] = (8, 6)
    dpi: int = 100
    style: str = "seaborn-v0_8-darkgrid"
    color_palette: Union[str, List[str]] = "husl"
    font_size: int = 10
    save_format: str = "png"


class PlotGenerator:
    """Generates plots for TIDE-Lite experimental results."""
    
    def __init__(
        self,
        config: Optional[PlotConfig] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Initialize plot generator.
        
        Args:
            config: Plot configuration (uses defaults if None).
            output_dir: Directory for saving plots.
        """
        self.config = config or PlotConfig()
        self.output_dir = Path(output_dir) if output_dir else Path("results/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        try:
            plt.style.use(self.config.style)
        except:
            plt.style.use("seaborn-v0_8")
        
        # Set default font sizes
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.font_size + 2,
            'axes.labelsize': self.config.font_size,
            'xtick.labelsize': self.config.font_size - 1,
            'ytick.labelsize': self.config.font_size - 1,
            'legend.fontsize': self.config.font_size - 1,
        })
        
        # Set color palette
        sns.set_palette(self.config.color_palette)
        
        logger.info(f"Initialized plot generator with output dir: {self.output_dir}")
    
    def plot_lambda_vs_metrics(
        self,
        data: Dict[float, Dict[str, float]],
        metrics: List[str],
        title: str = "Temporal Weight (λ) vs Performance Metrics",
        xlabel: str = "Temporal Weight (λ)",
        save_name: str = "lambda_vs_metrics",
    ) -> Figure:
        """Plot temporal weight vs various metrics.
        
        Args:
            data: Dict mapping lambda values to metric dictionaries.
            metrics: List of metric names to plot.
            title: Plot title.
            xlabel: X-axis label.
            save_name: Filename for saved plot.
            
        Returns:
            Matplotlib Figure object.
        """
        fig, axes = plt.subplots(
            1, len(metrics),
            figsize=(self.config.figure_size[0] * len(metrics) / 2, self.config.figure_size[1]),
            squeeze=False,
        )
        axes = axes.flatten()
        
        lambdas = sorted(data.keys())
        
        for idx, metric in enumerate(metrics):
            values = [data[lam].get(metric, 0) for lam in lambdas]
            
            ax = axes[idx]
            ax.plot(lambdas, values, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            
            # Find and mark best value
            best_idx = np.argmax(values)
            ax.plot(lambdas[best_idx], values[best_idx], 'r*', markersize=15)
            ax.annotate(
                f"Best: {values[best_idx]:.3f}",
                xy=(lambdas[best_idx], values[best_idx]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                color='red',
            )
        
        fig.suptitle(title, fontsize=self.config.font_size + 3, y=1.02)
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.{self.config.save_format}"
        fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_mlp_size_vs_metrics(
        self,
        data: Dict[int, Dict[str, float]],
        metrics: List[str],
        title: str = "MLP Hidden Size vs Performance",
        xlabel: str = "MLP Hidden Dimension",
        save_name: str = "mlp_size_vs_metrics",
    ) -> Figure:
        """Plot MLP hidden size vs various metrics.
        
        Args:
            data: Dict mapping MLP sizes to metric dictionaries.
            metrics: List of metric names to plot.
            title: Plot title.
            xlabel: X-axis label.
            save_name: Filename for saved plot.
            
        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        mlp_sizes = sorted(data.keys())
        x_pos = np.arange(len(mlp_sizes))
        
        # Plot each metric as a line
        for metric in metrics:
            values = [data[size].get(metric, 0) for size in mlp_sizes]
            ax.plot(
                x_pos,
                values,
                marker='o',
                label=metric.replace('_', ' ').title(),
                linewidth=2,
                markersize=8,
            )
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Metric Value")
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(mlp_sizes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.{self.config.save_format}"
        fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_time_encoding_comparison(
        self,
        data: Dict[int, Dict[str, float]],
        metric: str = "temporal_consistency_score",
        title: str = "Time Encoding Dimension Impact",
        xlabel: str = "Time Encoding Dimension",
        save_name: str = "time_encoding_comparison",
    ) -> Figure:
        """Plot time encoding dimension comparison.
        
        Args:
            data: Dict mapping encoding dims to metric dictionaries.
            metric: Metric to focus on.
            title: Plot title.
            xlabel: X-axis label.
            save_name: Filename for saved plot.
            
        Returns:
            Matplotlib Figure object.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.config.figure_size[0] * 1.5, self.config.figure_size[1]))
        
        dims = sorted(data.keys())
        values = [data[dim].get(metric, 0) for dim in dims]
        
        # Bar plot
        x_pos = np.arange(len(dims))
        bars = ax1.bar(x_pos, values)
        
        # Color bars based on value
        colors = plt.cm.viridis(np.array(values) / max(values))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.set_title("Performance by Encoding Dimension")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(dims)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (dim, val) in enumerate(zip(dims, values)):
            ax1.text(i, val + 0.01, f"{val:.3f}", ha='center', va='bottom')
        
        # Scatter plot with trend line
        ax2.scatter(dims, values, s=100, c=values, cmap='viridis', edgecolors='black', linewidth=1)
        
        # Fit polynomial trend
        z = np.polyfit(dims, values, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(dims), max(dims), 100)
        ax2.plot(x_smooth, p(x_smooth), 'r--', alpha=0.7, label='Trend')
        
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(metric.replace('_', ' ').title())
        ax2.set_title("Trend Analysis")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=self.config.font_size + 3, y=1.02)
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.{self.config.save_format}"
        fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_model_comparison(
        self,
        models_data: Dict[str, Dict[str, float]],
        metrics: List[str],
        title: str = "Model Performance Comparison",
        save_name: str = "model_comparison",
    ) -> Figure:
        """Create grouped bar chart comparing models across metrics.
        
        Args:
            models_data: Dict mapping model names to metric dictionaries.
            metrics: List of metrics to compare.
            title: Plot title.
            save_name: Filename for saved plot.
            
        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(self.config.figure_size[0] * 1.2, self.config.figure_size[1]))
        
        # Prepare data
        model_names = list(models_data.keys())
        n_models = len(model_names)
        n_metrics = len(metrics)
        
        x = np.arange(n_metrics)
        width = 0.8 / n_models
        
        # Plot bars for each model
        for i, model in enumerate(model_names):
            values = [models_data[model].get(metric, 0) for metric in metrics]
            offset = width * (i - n_models / 2 + 0.5)
            bars = ax.bar(x + offset, values, width, label=model)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{val:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )
        
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.{self.config.save_format}"
        fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_ablation_heatmap(
        self,
        ablation_data: Dict[Tuple[Any, Any], float],
        x_param: str = "mlp_hidden_dim",
        y_param: str = "temporal_weight",
        metric: str = "spearman",
        title: str = "Ablation Study Heatmap",
        save_name: str = "ablation_heatmap",
    ) -> Figure:
        """Create heatmap for ablation study results.
        
        Args:
            ablation_data: Dict mapping (x_val, y_val) tuples to metric values.
            x_param: Parameter for x-axis.
            y_param: Parameter for y-axis.
            metric: Metric being visualized.
            title: Plot title.
            save_name: Filename for saved plot.
            
        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Extract unique x and y values
        x_values = sorted(set(k[0] for k in ablation_data.keys()))
        y_values = sorted(set(k[1] for k in ablation_data.keys()))
        
        # Create matrix
        matrix = np.zeros((len(y_values), len(x_values)))
        for i, y in enumerate(y_values):
            for j, x in enumerate(x_values):
                matrix[i, j] = ablation_data.get((x, y), np.nan)
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_xticklabels(x_values)
        ax.set_yticklabels(y_values)
        
        # Add labels
        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel(y_param.replace('_', ' ').title())
        ax.set_title(f"{title}\n{metric.replace('_', ' ').title()}")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric.replace('_', ' ').title(), rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                text = ax.text(
                    j, i, f'{matrix[i, j]:.3f}',
                    ha="center", va="center", color="black", fontsize=8,
                )
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.{self.config.save_format}"
        fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_training_curves(
        self,
        metrics_data: Dict[str, List[float]],
        title: str = "Training Progress",
        xlabel: str = "Epoch",
        save_name: str = "training_curves",
    ) -> Figure:
        """Plot training and validation curves.
        
        Args:
            metrics_data: Dict mapping metric names to lists of values per epoch.
            title: Plot title.
            xlabel: X-axis label.
            save_name: Filename for saved plot.
            
        Returns:
            Matplotlib Figure object.
        """
        fig, axes = plt.subplots(2, 2, figsize=(self.config.figure_size[0] * 1.5, self.config.figure_size[1] * 1.5))
        axes = axes.flatten()
        
        # Define which metrics go in which subplot
        plot_config = [
            (["train_loss", "val_loss"], "Loss", axes[0]),
            (["val_spearman"], "Spearman Correlation", axes[1]),
            (["learning_rate"], "Learning Rate", axes[2]),
            (["train_loss", "val_loss"], "Loss (Log Scale)", axes[3]),
        ]
        
        for metrics_subset, ylabel, ax in plot_config:
            for metric in metrics_subset:
                if metric in metrics_data:
                    values = metrics_data[metric]
                    epochs = list(range(1, len(values) + 1))
                    ax.plot(epochs, values, marker='o', label=metric.replace('_', ' ').title(), linewidth=2)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Log scale for last subplot
            if ylabel == "Loss (Log Scale)":
                ax.set_yscale('log')
        
        fig.suptitle(title, fontsize=self.config.font_size + 3)
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.{self.config.save_format}"
        fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def generate_all_plots(
        self,
        summary_json: Union[str, Path],
        dry_run: bool = False,
    ) -> Dict[str, Figure]:
        """Generate all standard plots from summary JSON.
        
        Args:
            summary_json: Path to summary JSON file.
            dry_run: If True, only describe what would be plotted.
            
        Returns:
            Dictionary mapping plot names to Figure objects.
        """
        summary_path = Path(summary_json)
        
        if dry_run:
            print("\n" + "=" * 70)
            print("PLOT GENERATION PLAN")
            print("=" * 70)
            print(f"\nInput file: {summary_path}")
            print(f"Output directory: {self.output_dir}")
            print("\nPlots to generate:")
            print("  1. Lambda vs Metrics (temporal weight analysis)")
            print("  2. MLP Size vs Performance")
            print("  3. Time Encoding Comparison")
            print("  4. Model Comparison Bar Chart")
            print("  5. Ablation Study Heatmap")
            print("  6. Training Curves")
            print("\n[DRY RUN] No plots will be generated")
            print("=" * 70)
            return {}
        
        # Load summary data
        with open(summary_path, "r") as f:
            data = json.load(f)
        
        figures = {}
        
        # Generate each plot type based on available data
        logger.info("Generating plots from summary data")
        
        # Model comparison
        if "models" in data:
            models_data = {}
            for model_name, tasks in data["models"].items():
                if "STS-B" in tasks:
                    models_data[model_name] = tasks["STS-B"]
            
            if models_data:
                fig = self.plot_model_comparison(
                    models_data,
                    ["spearman", "pearson", "mse"],
                    title="STS-B Performance Comparison",
                )
                figures["model_comparison"] = fig
        
        # Training curves (if available)
        if "models" in data:
            for model_name, tasks in data["models"].items():
                if "training" in tasks and "train_loss" in tasks["training"]:
                    fig = self.plot_training_curves(
                        tasks["training"],
                        title=f"Training Progress - {model_name}",
                        save_name=f"training_curves_{model_name}",
                    )
                    figures[f"training_curves_{model_name}"] = fig
                    break  # Just do first one
        
        logger.info(f"Generated {len(figures)} plots")
        
        return figures


def create_publication_figures(
    summary_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[PlotConfig] = None,
) -> None:
    """Create publication-ready figures from results.
    
    Args:
        summary_json: Path to summary JSON file.
        output_dir: Output directory for figures.
        config: Plot configuration.
    """
    # Use high-quality settings for publication
    if config is None:
        config = PlotConfig(
            figure_size=(6, 4),
            dpi=300,
            style="seaborn-v0_8-whitegrid",
            save_format="pdf",
        )
    
    generator = PlotGenerator(config, output_dir)
    generator.generate_all_plots(summary_json)
    
    logger.info(f"Created publication figures in {output_dir}")
