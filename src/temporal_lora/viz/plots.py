"""Visualization utilities for results and embeddings."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP

from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_performance_heatmap(
    results_df: pd.DataFrame,
    metric: str = "ndcg@10",
    title: str = "Performance Heatmap",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Create a heatmap of retrieval performance.
    
    Args:
        results_df: DataFrame with scenarios as rows and metrics as columns.
        metric: Metric to visualize.
        title: Plot title.
        output_path: Path to save figure.
        figsize: Figure size.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
    """
    plt.figure(figsize=figsize)
    
    # Extract metric column
    if metric not in results_df.columns:
        logger.warning(f"Metric {metric} not found in results")
        return
    
    # Create matrix for heatmap
    # Rows: scenarios, Cols: buckets
    # Parse scenario names
    data_matrix = []
    row_labels = []
    col_labels = []
    
    for idx, row in results_df.iterrows():
        row_labels.append(str(idx))
        data_matrix.append([row[metric]])
    
    if len(data_matrix) == 0:
        logger.warning("No data to plot")
        return
    
    # Create heatmap
    sns.heatmap(
        data_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=vmin if vmin is not None else 0.0,
        vmax=vmax if vmax is not None else 1.0,
        cbar_kws={"label": metric},
        yticklabels=row_labels,
        xticklabels=[metric],
    )
    
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Metric")
    plt.ylabel("Scenario")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved heatmap to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_comparison_heatmaps(
    baseline_results: pd.DataFrame,
    lora_results: pd.DataFrame,
    metric: str = "ndcg@10",
    output_dir: Path = None,
    figsize: Tuple[int, int] = (18, 6),
) -> None:
    """Create three heatmaps: baseline, LoRA, and delta.
    
    Args:
        baseline_results: Baseline results DataFrame.
        lora_results: LoRA results DataFrame.
        metric: Metric to visualize.
        output_dir: Directory to save figures.
        figsize: Figure size.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Get common scenarios
    common_scenarios = list(set(baseline_results.index) & set(lora_results.index))
    common_scenarios.sort()
    
    if len(common_scenarios) == 0:
        logger.warning("No common scenarios between baseline and LoRA")
        return
    
    # Extract values
    baseline_values = [baseline_results.loc[s, metric] for s in common_scenarios]
    lora_values = [lora_results.loc[s, metric] for s in common_scenarios]
    delta_values = [l - b for l, b in zip(lora_values, baseline_values)]
    
    # Determine consistent value range
    all_values = baseline_values + lora_values
    vmin = min(all_values) * 0.95
    vmax = max(all_values) * 1.05
    
    # Delta range (symmetric around 0)
    delta_max = max(abs(min(delta_values)), abs(max(delta_values)))
    delta_range = (-delta_max * 1.1, delta_max * 1.1)
    
    # Baseline heatmap
    sns.heatmap(
        np.array(baseline_values).reshape(-1, 1),
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": metric},
        yticklabels=common_scenarios,
        xticklabels=["Baseline"],
        ax=axes[0],
    )
    axes[0].set_title("Baseline", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Scenario")
    
    # LoRA heatmap
    sns.heatmap(
        np.array(lora_values).reshape(-1, 1),
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": metric},
        yticklabels=common_scenarios,
        xticklabels=["LoRA"],
        ax=axes[1],
    )
    axes[1].set_title("LoRA", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("")
    
    # Delta heatmap
    sns.heatmap(
        np.array(delta_values).reshape(-1, 1),
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=delta_range[0],
        vmax=delta_range[1],
        cbar_kws={"label": "Δ (LoRA - Baseline)"},
        yticklabels=common_scenarios,
        xticklabels=["Δ"],
        ax=axes[2],
    )
    axes[2].set_title("Improvement (Δ)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("")
    
    plt.suptitle(f"Performance Comparison: {metric}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"comparison_heatmaps_{metric}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved comparison heatmaps to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_umap_visualization(
    embeddings_dict: Dict[str, np.ndarray],
    output_path: Optional[Path] = None,
    max_points: int = 10000,
    seed: int = 42,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Create UMAP visualization of embeddings colored by bucket.
    
    Args:
        embeddings_dict: Dictionary mapping bucket name -> embeddings array.
        output_path: Path to save figure.
        max_points: Maximum points to visualize (sampled if exceeded).
        seed: Random seed for sampling.
        figsize: Figure size.
    """
    logger.info("Creating UMAP visualization...")
    
    # Collect all embeddings
    all_embeddings = []
    all_labels = []
    bucket_names = []
    
    rng = np.random.RandomState(seed)
    
    for bucket_name, embeddings in embeddings_dict.items():
        n_samples = len(embeddings)
        
        # Sample if too many points
        if n_samples > max_points // len(embeddings_dict):
            sample_size = max_points // len(embeddings_dict)
            indices = rng.choice(n_samples, size=sample_size, replace=False)
            sampled_embeddings = embeddings[indices]
        else:
            sampled_embeddings = embeddings
        
        all_embeddings.append(sampled_embeddings)
        all_labels.extend([bucket_name] * len(sampled_embeddings))
        bucket_names.append(bucket_name)
        
        logger.info(f"Bucket {bucket_name}: {len(sampled_embeddings)} points")
    
    # Concatenate
    all_embeddings = np.vstack(all_embeddings)
    logger.info(f"Total points for UMAP: {len(all_embeddings)}")
    
    # Run UMAP
    logger.info("Running UMAP projection...")
    umap_model = UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    embedding_2d = umap_model.fit_transform(all_embeddings)
    
    # Plot
    plt.figure(figsize=figsize)
    
    # Color palette
    colors = sns.color_palette("husl", len(bucket_names))
    
    for i, bucket_name in enumerate(bucket_names):
        mask = np.array(all_labels) == bucket_name
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[colors[i]],
            label=bucket_name,
            alpha=0.6,
            s=20,
            edgecolors="none",
        )
    
    plt.title("UMAP Projection of Embeddings", fontsize=14, fontweight="bold")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Time Bucket", loc="best", framealpha=0.9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved UMAP to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_results(
    baseline_results_path: Optional[Path],
    lora_results_path: Optional[Path],
    embeddings_dir: Optional[Path],
    output_dir: Path,
) -> None:
    """Generate all visualization plots.
    
    Args:
        baseline_results_path: Path to baseline results CSV.
        lora_results_path: Path to LoRA results CSV.
        embeddings_dir: Directory with cached embeddings for UMAP.
        output_dir: Output directory for figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    baseline_df = None
    lora_df = None
    
    if baseline_results_path and baseline_results_path.exists():
        baseline_df = pd.read_csv(baseline_results_path, index_col=0)
        logger.info(f"Loaded baseline results: {baseline_results_path}")
    
    if lora_results_path and lora_results_path.exists():
        lora_df = pd.read_csv(lora_results_path, index_col=0)
        logger.info(f"Loaded LoRA results: {lora_results_path}")
    
    # Create comparison heatmaps if both available
    if baseline_df is not None and lora_df is not None:
        for metric in ["ndcg@10", "recall@10", "recall@100", "mrr"]:
            if metric in baseline_df.columns and metric in lora_df.columns:
                create_comparison_heatmaps(
                    baseline_df,
                    lora_df,
                    metric=metric,
                    output_dir=output_dir,
                )
    
    # Create UMAP visualization if embeddings available
    if embeddings_dir and embeddings_dir.exists():
        from ..eval.encoder import load_embeddings
        
        # Load embeddings for each bucket
        embeddings_dict = {}
        
        for bucket_dir in embeddings_dir.iterdir():
            if not bucket_dir.is_dir():
                continue
            
            bucket_name = bucket_dir.name
            
            # Load test split embeddings
            test_dir = bucket_dir / "test"
            if test_dir.exists():
                embeddings, _ = load_embeddings(test_dir)
                embeddings_dict[bucket_name] = embeddings
        
        if embeddings_dict:
            umap_path = output_dir / "umap_embeddings.png"
            create_umap_visualization(embeddings_dict, output_path=umap_path)
