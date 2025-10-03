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


def plot_heatmaps_panel(
    baseline_matrix: pd.DataFrame,
    lora_matrix: pd.DataFrame,
    delta_matrix: Optional[pd.DataFrame] = None,
    metric_name: str = "NDCG@10",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (18, 6),
    annot: bool = True,
    fmt: str = ".3f",
) -> None:
    """Create three-panel heatmap: baseline, LoRA, and delta.
    
    Args:
        baseline_matrix: Baseline performance matrix (query_bucket × doc_bucket).
        lora_matrix: LoRA performance matrix (query_bucket × doc_bucket).
        delta_matrix: Delta matrix (LoRA - baseline). Computed if not provided.
        metric_name: Name of metric being visualized.
        output_path: Path to save figure.
        figsize: Figure size.
        annot: Whether to annotate cells with values.
        fmt: Format string for annotations.
    """
    logger.info(f"Creating heatmap panel for {metric_name}")
    
    # Compute delta if not provided
    if delta_matrix is None:
        delta_matrix = lora_matrix - baseline_matrix
    
    # Determine consistent value range for baseline and LoRA
    vmin = min(baseline_matrix.values.min(), lora_matrix.values.min())
    vmax = max(baseline_matrix.values.max(), lora_matrix.values.max())
    
    # Add some padding
    value_range = vmax - vmin
    vmin = vmin - 0.05 * value_range
    vmax = vmax + 0.05 * value_range
    
    # Delta range (symmetric around 0)
    delta_abs_max = max(abs(delta_matrix.values.min()), abs(delta_matrix.values.max()))
    delta_vmin = -delta_abs_max * 1.1
    delta_vmax = delta_abs_max * 1.1
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Baseline heatmap
    sns.heatmap(
        baseline_matrix,
        annot=annot,
        fmt=fmt,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": metric_name},
        ax=axes[0],
        square=True,
    )
    axes[0].set_title("Baseline (Frozen)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Doc Bucket")
    axes[0].set_ylabel("Query Bucket")
    
    # LoRA heatmap
    sns.heatmap(
        lora_matrix,
        annot=annot,
        fmt=fmt,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": metric_name},
        ax=axes[1],
        square=True,
    )
    axes[1].set_title("LoRA", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Doc Bucket")
    axes[1].set_ylabel("Query Bucket")
    
    # Delta heatmap
    sns.heatmap(
        delta_matrix,
        annot=annot,
        fmt=fmt,
        cmap="RdYlGn",
        center=0,
        vmin=delta_vmin,
        vmax=delta_vmax,
        cbar_kws={"label": f"Δ ({metric_name})"},
        ax=axes[2],
        square=True,
    )
    axes[2].set_title("Improvement (LoRA - Baseline)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Doc Bucket")
    axes[2].set_ylabel("Query Bucket")
    
    # Overall title
    plt.suptitle(
        f"Cross-Period Retrieval Performance: {metric_name}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved heatmap panel to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_umap_sample(
    embeddings_dict: Dict[str, np.ndarray],
    output_path: Optional[Path] = None,
    max_points: int = 10000,
    seed: int = 42,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Create UMAP visualization with sampled embeddings.
    
    Args:
        embeddings_dict: Dictionary mapping bucket name -> embeddings array.
        output_path: Path to save figure.
        max_points: Maximum total points to visualize.
        seed: Random seed for reproducible sampling.
        figsize: Figure size.
    """
    logger.info(f"Creating UMAP visualization (max {max_points} points)...")
    
    # Collect all embeddings with sampling
    all_embeddings = []
    all_labels = []
    bucket_names = sorted(embeddings_dict.keys())
    
    rng = np.random.RandomState(seed)
    
    # Calculate points per bucket
    total_points = sum(len(emb) for emb in embeddings_dict.values())
    points_per_bucket = min(max_points // len(bucket_names), total_points // len(bucket_names))
    
    for bucket_name in bucket_names:
        embeddings = embeddings_dict[bucket_name]
        n_samples = len(embeddings)
        
        # Sample if necessary
        if n_samples > points_per_bucket:
            indices = rng.choice(n_samples, size=points_per_bucket, replace=False)
            sampled_embeddings = embeddings[indices]
        else:
            sampled_embeddings = embeddings
        
        all_embeddings.append(sampled_embeddings)
        all_labels.extend([bucket_name] * len(sampled_embeddings))
        
        logger.info(f"  {bucket_name}: {len(sampled_embeddings)} points")
    
    # Concatenate
    all_embeddings = np.vstack(all_embeddings)
    logger.info(f"Total points: {len(all_embeddings)}")
    
    # Run UMAP
    logger.info("Running UMAP projection...")
    umap_model = UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
    )
    embedding_2d = umap_model.fit_transform(all_embeddings)
    
    # Plot
    plt.figure(figsize=figsize)
    
    # Color palette
    colors = sns.color_palette("husl", len(bucket_names))
    
    # Plot each bucket
    for i, bucket_name in enumerate(bucket_names):
        mask = np.array(all_labels) == bucket_name
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=[colors[i]],
            label=bucket_name,
            alpha=0.6,
            s=30,
            edgecolors="white",
            linewidth=0.5,
        )
    
    plt.title(
        f"UMAP Projection of Embeddings (n={len(all_embeddings):,}, seed={seed})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("UMAP Dimension 1", fontsize=11)
    plt.ylabel("UMAP Dimension 2", fontsize=11)
    plt.legend(title="Time Bucket", loc="best", framealpha=0.95, edgecolor="black")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved UMAP to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_all_heatmaps(
    results_dir: Path,
    output_dir: Path,
    baseline_mode: str = "baseline_frozen",
    lora_mode: str = "lora",
) -> None:
    """Create heatmap panels for all metrics.
    
    Args:
        results_dir: Directory containing evaluation result CSVs.
        output_dir: Output directory for figures.
        baseline_mode: Name of baseline mode.
        lora_mode: Name of LoRA mode.
    """
    logger.info("Creating heatmap panels for all metrics...")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "ndcg_at_10": "NDCG@10",
        "recall_at_10": "Recall@10",
        "recall_at_100": "Recall@100",
        "mrr": "MRR",
    }
    
    for metric_file, metric_display in metrics.items():
        # Load baseline and LoRA matrices
        baseline_path = results_dir / f"{baseline_mode}_{metric_file}.csv"
        lora_path = results_dir / f"{lora_mode}_{metric_file}.csv"
        
        if not baseline_path.exists() or not lora_path.exists():
            logger.warning(f"Missing results for {metric_file}, skipping...")
            continue
        
        baseline_df = pd.read_csv(baseline_path, index_col=0)
        lora_df = pd.read_csv(lora_path, index_col=0)
        
        # Create heatmap panel
        output_path = output_dir / f"heatmap_panel_{metric_file}.png"
        plot_heatmaps_panel(
            baseline_matrix=baseline_df,
            lora_matrix=lora_df,
            metric_name=metric_display,
            output_path=output_path,
        )


def visualize_results(
    results_dir: Path,
    embeddings_dir: Optional[Path],
    output_dir: Path,
    baseline_mode: str = "baseline_frozen",
    lora_mode: str = "lora",
) -> None:
    """Generate all visualization plots.
    
    Args:
        results_dir: Directory with evaluation result CSVs.
        embeddings_dir: Directory with cached embeddings for UMAP.
        output_dir: Output directory for figures.
        baseline_mode: Name of baseline mode.
        lora_mode: Name of LoRA mode.
    """
    logger.info("\n" + "="*60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create heatmap panels
    create_all_heatmaps(
        results_dir=results_dir,
        output_dir=output_dir,
        baseline_mode=baseline_mode,
        lora_mode=lora_mode,
    )
    
    # Create UMAP visualization
    if embeddings_dir and embeddings_dir.exists():
        from ..eval.encoder import load_embeddings
        
        logger.info("\nLoading embeddings for UMAP...")
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
                logger.info(f"  Loaded {bucket_name}: {len(embeddings)} embeddings")
        
        if embeddings_dict:
            umap_path = output_dir / "umap_embeddings.png"
            plot_umap_sample(embeddings_dict, output_path=umap_path)
        else:
            logger.warning("No embeddings found for UMAP")
    
    logger.info("\n✓ Visualization complete!")
    logger.info("="*60)
