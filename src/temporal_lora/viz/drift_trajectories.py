"""Term drift trajectory visualization.

Shows how embeddings of specific terms evolve across time periods.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP

from ..models.lora_model import load_lora_adapter
from ..utils.logging import get_logger

logger = get_logger(__name__)


def extract_contexts(
    data_dir: Path,
    bucket_name: str,
    terms: List[str],
    contexts_per_term: int = 50,
    context_window: int = 5,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Extract contexts containing specific terms from a bucket.
    
    Args:
        data_dir: Directory containing processed data.
        bucket_name: Name of the time bucket.
        terms: List of terms to search for.
        contexts_per_term: Number of contexts to extract per term.
        context_window: Number of words around the term to include.
        seed: Random seed for sampling.
        
    Returns:
        Dictionary mapping term -> list of context strings.
    """
    bucket_dir = data_dir / bucket_name
    
    # Load data
    contexts = {term: [] for term in terms}
    
    for split in ["train", "val", "test"]:
        csv_path = bucket_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            text = str(row.get("abstract", ""))
            
            for term in terms:
                # Case-insensitive search
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                matches = list(pattern.finditer(text))
                
                if matches:
                    # Extract context around each match
                    for match in matches:
                        # Get word-level context
                        words = text.split()
                        term_words = term.split()
                        
                        # Find term position in words
                        term_pos = None
                        for i in range(len(words) - len(term_words) + 1):
                            if ' '.join(words[i:i+len(term_words)]).lower() == term.lower():
                                term_pos = i
                                break
                        
                        if term_pos is not None:
                            start = max(0, term_pos - context_window)
                            end = min(len(words), term_pos + len(term_words) + context_window)
                            context = ' '.join(words[start:end])
                            contexts[term].append(context)
    
    # Sample contexts
    rng = np.random.RandomState(seed)
    sampled_contexts = {}
    
    for term, term_contexts in contexts.items():
        if len(term_contexts) > contexts_per_term:
            indices = rng.choice(len(term_contexts), size=contexts_per_term, replace=False)
            sampled_contexts[term] = [term_contexts[i] for i in indices]
        else:
            sampled_contexts[term] = term_contexts
        
        logger.info(f"  {bucket_name} - {term}: {len(sampled_contexts[term])} contexts")
    
    return sampled_contexts


def encode_contexts_with_adapter(
    contexts: List[str],
    base_model_name: str,
    adapter_dir: Optional[Path],
    use_lora: bool = True,
) -> np.ndarray:
    """Encode contexts with a specific adapter.
    
    Args:
        contexts: List of context strings.
        base_model_name: Base model identifier.
        adapter_dir: Path to LoRA adapter (if using LoRA).
        use_lora: Whether to use LoRA adapter.
        
    Returns:
        Embeddings array (n_contexts, dim).
    """
    if not contexts:
        return np.array([])
    
    # Load model
    if use_lora and adapter_dir and adapter_dir.exists():
        model = load_lora_adapter(base_model_name, adapter_dir)
        logger.info(f"  Loaded LoRA adapter from: {adapter_dir.name}")
    else:
        model = SentenceTransformer(base_model_name)
        logger.info(f"  Loaded base model: {base_model_name}")
    
    # Encode
    embeddings = model.encode(
        contexts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    
    return embeddings


def compute_drift_trajectories(
    data_dir: Path,
    adapters_dir: Path,
    buckets: List[str],
    terms: List[str],
    base_model_name: str,
    contexts_per_term: int = 50,
    use_lora: bool = True,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """Compute term drift trajectories across time buckets.
    
    Args:
        data_dir: Directory with processed data.
        adapters_dir: Directory with LoRA adapters.
        buckets: List of bucket names (in temporal order).
        terms: List of terms to track.
        base_model_name: Base model identifier.
        contexts_per_term: Contexts to sample per term per bucket.
        use_lora: Whether to use LoRA adapters.
        seed: Random seed.
        
    Returns:
        Tuple of (embeddings_dict, contexts_dict).
        embeddings_dict: {term -> array of shape (n_buckets, dim)}
        contexts_dict: {term -> list of contexts per bucket}
    """
    logger.info(f"Computing drift trajectories for {len(terms)} terms...")
    
    all_embeddings = {term: [] for term in terms}
    all_contexts = {term: [] for term in terms}
    
    for bucket_name in buckets:
        logger.info(f"\nProcessing bucket: {bucket_name}")
        
        # Extract contexts
        bucket_contexts = extract_contexts(
            data_dir=data_dir,
            bucket_name=bucket_name,
            terms=terms,
            contexts_per_term=contexts_per_term,
            seed=seed,
        )
        
        # Determine adapter path
        adapter_dir = adapters_dir / bucket_name if use_lora else None
        
        # Encode each term's contexts with this bucket's adapter
        for term in terms:
            contexts = bucket_contexts[term]
            
            if not contexts:
                logger.warning(f"  {term}: No contexts found in {bucket_name}")
                # Use zero embedding as placeholder
                embeddings = np.zeros((1, 384))  # Assuming 384-dim embeddings
            else:
                embeddings = encode_contexts_with_adapter(
                    contexts=contexts,
                    base_model_name=base_model_name,
                    adapter_dir=adapter_dir,
                    use_lora=use_lora,
                )
            
            # Average embeddings for this term in this bucket
            avg_embedding = embeddings.mean(axis=0)
            all_embeddings[term].append(avg_embedding)
            all_contexts[term].append(contexts)
    
    # Convert to arrays
    embeddings_dict = {
        term: np.vstack(emb_list) for term, emb_list in all_embeddings.items()
    }
    
    return embeddings_dict, all_contexts


def plot_drift_trajectories(
    embeddings_dict: Dict[str, np.ndarray],
    buckets: List[str],
    terms: List[str],
    output_path: Optional[Path] = None,
    seed: int = 42,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """Plot term drift trajectories with arrowed polylines.
    
    Args:
        embeddings_dict: Dictionary mapping term -> embeddings array (n_buckets, dim).
        buckets: List of bucket names (in temporal order).
        terms: List of terms.
        output_path: Path to save figure.
        seed: Random seed for UMAP.
        figsize: Figure size.
    """
    logger.info("Creating drift trajectory plot...")
    
    # Collect all embeddings for UMAP
    all_embeddings = []
    labels = []
    
    for term in terms:
        term_embeddings = embeddings_dict[term]
        all_embeddings.append(term_embeddings)
        labels.extend([term] * len(term_embeddings))
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Run UMAP
    logger.info("Running UMAP projection...")
    umap_model = UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=min(15, len(all_embeddings) - 1),
        min_dist=0.1,
        metric="cosine",
    )
    embedding_2d = umap_model.fit_transform(all_embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    import seaborn as sns
    colors = sns.color_palette("husl", len(terms))
    
    # Plot each term's trajectory
    idx = 0
    for term_idx, term in enumerate(terms):
        n_buckets = len(buckets)
        term_coords = embedding_2d[idx:idx+n_buckets]
        idx += n_buckets
        
        # Plot points
        ax.scatter(
            term_coords[:, 0],
            term_coords[:, 1],
            c=[colors[term_idx]],
            s=200,
            alpha=0.7,
            edgecolors="black",
            linewidth=1.5,
            label=term,
            zorder=3,
        )
        
        # Plot trajectory with arrows
        for i in range(len(term_coords) - 1):
            x_start, y_start = term_coords[i]
            x_end, y_end = term_coords[i + 1]
            
            # Draw line
            ax.plot(
                [x_start, x_end],
                [y_start, y_end],
                color=colors[term_idx],
                linewidth=2,
                alpha=0.6,
                zorder=2,
            )
            
            # Draw arrow
            dx = x_end - x_start
            dy = y_end - y_start
            ax.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops=dict(
                    arrowstyle="->",
                    color=colors[term_idx],
                    lw=2,
                    alpha=0.7,
                ),
                zorder=2,
            )
        
        # Annotate bucket positions
        for i, bucket_name in enumerate(buckets):
            x, y = term_coords[i]
            ax.annotate(
                bucket_name,
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )
    
    ax.set_title(
        "Term Drift Trajectories Across Time Periods",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", fontsize=11)
    ax.legend(title="Terms", loc="best", framealpha=0.95, edgecolor="black")
    ax.grid(alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved drift trajectories to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def run_drift_analysis(
    data_dir: Path,
    adapters_dir: Path,
    buckets: List[str],
    terms: List[str],
    base_model_name: str,
    output_dir: Path,
    contexts_per_term: int = 50,
    use_lora: bool = True,
    seed: int = 42,
) -> None:
    """Run complete drift analysis and create visualization.
    
    Args:
        data_dir: Directory with processed data.
        adapters_dir: Directory with LoRA adapters.
        buckets: List of bucket names (in temporal order).
        terms: List of terms to track.
        base_model_name: Base model identifier.
        output_dir: Output directory for figure.
        contexts_per_term: Contexts per term per bucket.
        use_lora: Whether to use LoRA adapters.
        seed: Random seed.
    """
    logger.info("\n" + "="*60)
    logger.info("TERM DRIFT TRAJECTORY ANALYSIS")
    logger.info("="*60)
    logger.info(f"Terms: {', '.join(terms)}")
    logger.info(f"Buckets: {', '.join(buckets)}")
    logger.info(f"Contexts per term: {contexts_per_term}")
    logger.info(f"Use LoRA: {use_lora}")
    
    # Compute trajectories
    embeddings_dict, contexts_dict = compute_drift_trajectories(
        data_dir=data_dir,
        adapters_dir=adapters_dir,
        buckets=buckets,
        terms=terms,
        base_model_name=base_model_name,
        contexts_per_term=contexts_per_term,
        use_lora=use_lora,
        seed=seed,
    )
    
    # Create visualization
    output_path = output_dir / "drift_trajectories.png"
    plot_drift_trajectories(
        embeddings_dict=embeddings_dict,
        buckets=buckets,
        terms=terms,
        output_path=output_path,
        seed=seed,
    )
    
    # Save trajectory data
    data_output = output_dir / "drift_trajectories_data.npz"
    np.savez(
        data_output,
        **{term: embeddings_dict[term] for term in terms},
        buckets=buckets,
        terms=terms,
    )
    logger.info(f"✓ Saved trajectory data to: {data_output}")
    
    logger.info("\n✓ Drift analysis complete!")
    logger.info("="*60)
