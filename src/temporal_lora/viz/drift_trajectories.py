"""Term drift trajectory visualization across time periods."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from sentence_transformers import SentenceTransformer
from umap import UMAP

from ..eval.encoder import load_embeddings
from ..models.lora_model import load_lora_adapter
from ..utils.logging import get_logger

logger = get_logger(__name__)


def extract_term_contexts(
    data_path: Path,
    term: str,
    max_contexts: int = 50,
    context_window: int = 100,
) -> List[str]:
    """Extract contexts containing a specific term from a dataset.
    
    Args:
        data_path: Path to data CSV file.
        term: Term to search for (case-insensitive).
        max_contexts: Maximum number of contexts to extract.
        context_window: Number of characters around term to include.
        
    Returns:
        List of context strings.
    """
    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        return []
    
    df = pd.read_csv(data_path)
    
    if "abstract" not in df.columns:
        logger.warning(f"No 'abstract' column in {data_path}")
        return []
    
    # Search for term (case-insensitive)
    term_lower = term.lower()
    contexts = []
    
    for abstract in df["abstract"]:
        if not isinstance(abstract, str):
            continue
        
        abstract_lower = abstract.lower()
        
        # Find all occurrences
        start_idx = 0
        while start_idx < len(abstract_lower):
            idx = abstract_lower.find(term_lower, start_idx)
            if idx == -1:
                break
            
            # Extract context window
            context_start = max(0, idx - context_window)
            context_end = min(len(abstract), idx + len(term) + context_window)
            context = abstract[context_start:context_end]
            
            contexts.append(context)
            
            if len(contexts) >= max_contexts:
                break
            
            start_idx = idx + 1
        
        if len(contexts) >= max_contexts:
            break
    
    logger.info(f"  Extracted {len(contexts)} contexts for '{term}'")
    return contexts[:max_contexts]


def collect_term_contexts_per_bucket(
    data_dir: Path,
    buckets: List[str],
    terms: List[str],
    contexts_per_bin: int = 50,
) -> Dict[str, Dict[str, List[str]]]:
    """Collect contexts for each term in each bucket.
    
    Args:
        data_dir: Directory containing bucket data.
        buckets: List of bucket names.
        terms: List of terms to track.
        contexts_per_bin: Number of contexts per term per bucket.
        
    Returns:
        Dictionary mapping bucket -> term -> contexts list.
    """
    logger.info(f"Collecting contexts for {len(terms)} terms across {len(buckets)} buckets...")
    
    contexts_dict = {}
    
    for bucket_name in buckets:
        logger.info(f"\nBucket: {bucket_name}")
        
        # Try different file locations
        possible_paths = [
            data_dir / bucket_name / "all.csv",
            data_dir / bucket_name / "train.csv",
            data_dir / bucket_name.replace("bucket_", "") / "all.csv",
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            logger.warning(f"  No data file found for {bucket_name}")
            continue
        
        bucket_contexts = {}
        
        for term in terms:
            contexts = extract_term_contexts(
                data_path=data_path,
                term=term,
                max_contexts=contexts_per_bin,
            )
            bucket_contexts[term] = contexts
        
        contexts_dict[bucket_name] = bucket_contexts
    
    return contexts_dict


def encode_term_contexts(
    contexts_dict: Dict[str, Dict[str, List[str]]],
    adapters_dir: Path,
    base_model_name: str,
    buckets: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Encode contexts using each bucket's adapter.
    
    Args:
        contexts_dict: Bucket -> term -> contexts.
        adapters_dir: Directory with LoRA adapters.
        base_model_name: Base model identifier.
        buckets: List of bucket names.
        
    Returns:
        Dictionary mapping bucket -> term -> embeddings array.
    """
    logger.info("Encoding contexts with bucket-specific adapters...")
    
    embeddings_dict = {}
    
    for bucket_name in buckets:
        if bucket_name not in contexts_dict:
            continue
        
        logger.info(f"\nEncoding with {bucket_name} adapter...")
        
        # Load adapter
        adapter_path = adapters_dir / bucket_name
        if not adapter_path.exists():
            logger.warning(f"  Adapter not found: {adapter_path}")
            continue
        
        model = load_lora_adapter(base_model_name, adapter_path)
        
        bucket_embeddings = {}
        
        for term, contexts in contexts_dict[bucket_name].items():
            if not contexts:
                continue
            
            # Encode
            embeddings = model.encode(
                contexts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            
            bucket_embeddings[term] = embeddings
            logger.info(f"  {term}: {len(embeddings)} embeddings")
        
        embeddings_dict[bucket_name] = bucket_embeddings
    
    return embeddings_dict


def compute_term_centroids(
    embeddings_dict: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute centroid embeddings for each term in each bucket.
    
    Args:
        embeddings_dict: Bucket -> term -> embeddings array.
        
    Returns:
        Dictionary mapping term -> bucket -> centroid vector.
    """
    logger.info("Computing term centroids per bucket...")
    
    term_centroids = {}
    
    # Get all terms
    all_terms = set()
    for bucket_embeddings in embeddings_dict.values():
        all_terms.update(bucket_embeddings.keys())
    
    for term in all_terms:
        term_centroids[term] = {}
        
        for bucket_name, bucket_embeddings in embeddings_dict.items():
            if term in bucket_embeddings:
                # Compute mean embedding (centroid)
                centroid = np.mean(bucket_embeddings[term], axis=0)
                term_centroids[term][bucket_name] = centroid
    
    return term_centroids


def plot_drift_trajectories(
    term_centroids: Dict[str, Dict[str, np.ndarray]],
    buckets: List[str],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10),
    seed: int = 42,
) -> None:
    """Plot term drift trajectories with arrowed polylines.
    
    Args:
        term_centroids: Term -> bucket -> centroid vector.
        buckets: List of bucket names in temporal order.
        output_path: Path to save figure.
        figsize: Figure size.
        seed: Random seed for UMAP.
    """
    logger.info("Creating drift trajectory plot...")
    
    # Collect all centroids for UMAP
    all_centroids = []
    centroid_labels = []  # (term, bucket)
    
    for term in sorted(term_centroids.keys()):
        for bucket_name in buckets:
            if bucket_name in term_centroids[term]:
                all_centroids.append(term_centroids[term][bucket_name])
                centroid_labels.append((term, bucket_name))
    
    if len(all_centroids) == 0:
        logger.warning("No centroids to plot")
        return
    
    all_centroids = np.array(all_centroids)
    logger.info(f"Total centroids: {len(all_centroids)}")
    
    # Run UMAP
    logger.info("Running UMAP on centroids...")
    umap_model = UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=min(15, len(all_centroids) - 1),
        min_dist=0.1,
        metric="cosine",
    )
    centroids_2d = umap_model.fit_transform(all_centroids)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette for terms
    terms_list = sorted(term_centroids.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(terms_list)))
    term_colors = {term: colors[i] for i, term in enumerate(terms_list)}
    
    # Plot trajectories for each term
    for term in terms_list:
        # Get coordinates in temporal order
        term_points = []
        term_bucket_names = []
        
        for bucket_name in buckets:
            if bucket_name in term_centroids[term]:
                # Find index in centroids_2d
                for i, (t, b) in enumerate(centroid_labels):
                    if t == term and b == bucket_name:
                        term_points.append(centroids_2d[i])
                        term_bucket_names.append(bucket_name)
                        break
        
        if len(term_points) < 2:
            # Can't draw trajectory with less than 2 points
            if len(term_points) == 1:
                # Just plot the point
                ax.scatter(
                    term_points[0][0],
                    term_points[0][1],
                    c=[term_colors[term]],
                    s=100,
                    marker="o",
                    edgecolors="black",
                    linewidth=1.5,
                    label=term,
                    zorder=5,
                )
            continue
        
        term_points = np.array(term_points)
        
        # Draw polyline with arrows
        for i in range(len(term_points) - 1):
            start = term_points[i]
            end = term_points[i + 1]
            
            # Draw line segment
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=term_colors[term],
                linewidth=2,
                alpha=0.6,
                zorder=2,
            )
            
            # Add arrow
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                color=term_colors[term],
                linewidth=2,
                mutation_scale=20,
                alpha=0.8,
                zorder=3,
            )
            ax.add_patch(arrow)
        
        # Plot points
        ax.scatter(
            term_points[:, 0],
            term_points[:, 1],
            c=[term_colors[term]] * len(term_points),
            s=150,
            marker="o",
            edgecolors="black",
            linewidth=1.5,
            label=term,
            zorder=5,
        )
        
        # Annotate first and last points
        # First point
        ax.annotate(
            f"{term}\n({term_bucket_names[0]})",
            xy=term_points[0],
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=term_colors[term], alpha=0.3),
            zorder=6,
        )
        
        # Last point
        if len(term_points) > 1:
            ax.annotate(
                f"{term}\n({term_bucket_names[-1]})",
                xy=term_points[-1],
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=term_colors[term], alpha=0.3),
                zorder=6,
            )
    
    ax.set_title(
        "Term Drift Trajectories Across Time Periods",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(
        title="Terms",
        loc="best",
        framealpha=0.95,
        edgecolor="black",
    )
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved drift trajectories to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_drift_trajectories(
    data_dir: Path,
    adapters_dir: Path,
    base_model_name: str,
    buckets: List[str],
    terms: List[str],
    output_path: Path,
    contexts_per_bin: int = 50,
    seed: int = 42,
) -> None:
    """Generate complete drift trajectory visualization.
    
    Args:
        data_dir: Directory with bucket data.
        adapters_dir: Directory with LoRA adapters.
        base_model_name: Base model identifier.
        buckets: List of bucket names in temporal order.
        terms: List of terms to track.
        output_path: Output path for figure.
        contexts_per_bin: Number of contexts per term per bucket.
        seed: Random seed.
    """
    logger.info("\n" + "="*60)
    logger.info("GENERATING TERM DRIFT TRAJECTORIES")
    logger.info("="*60)
    logger.info(f"Terms: {', '.join(terms)}")
    logger.info(f"Buckets: {', '.join(buckets)}")
    logger.info(f"Contexts per bin: {contexts_per_bin}")
    
    # Step 1: Collect contexts
    contexts_dict = collect_term_contexts_per_bucket(
        data_dir=data_dir,
        buckets=buckets,
        terms=terms,
        contexts_per_bin=contexts_per_bin,
    )
    
    # Step 2: Encode with adapters
    embeddings_dict = encode_term_contexts(
        contexts_dict=contexts_dict,
        adapters_dir=adapters_dir,
        base_model_name=base_model_name,
        buckets=buckets,
    )
    
    # Step 3: Compute centroids
    term_centroids = compute_term_centroids(embeddings_dict)
    
    # Step 4: Plot trajectories
    plot_drift_trajectories(
        term_centroids=term_centroids,
        buckets=buckets,
        output_path=output_path,
        seed=seed,
    )
    
    logger.info("\n✓ Drift trajectories complete!")
    logger.info("="*60)
