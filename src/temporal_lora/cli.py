"""Command-line interface for Temporal LoRA pipeline."""

from pathlib import Path
from typing import Optional

import typer

from .data.pipeline import run_data_pipeline
from .train.trainer import train_all_buckets
from .utils.env import dump_environment
from .utils.io import load_config
from .utils.paths import (
    ADAPTERS_DIR,
    CONFIG_DIR,
    DATA_PROCESSED_DIR,
    DELIVERABLES_REPRO_DIR,
    PROJECT_ROOT,
    ensure_dirs,
)

app = typer.Typer(
    name="temporal-lora",
    help="Temporal LoRA for Dynamic Sentence Embeddings",
    add_completion=False,
)


@app.command()
def env_dump() -> None:
    """Dump environment info (CUDA, packages, git SHA) to deliverables/repro/."""
    ensure_dirs()
    dump_environment(DELIVERABLES_REPRO_DIR, repo_path=PROJECT_ROOT)
    typer.echo("✓ Environment info dumped successfully")


@app.command()
def prepare_data(
    max_per_bucket: int = typer.Option(6000, help="Max samples per time bucket"),
    output_dir: Optional[str] = typer.Option(None, help="Override output directory"),
) -> None:
    """Download and preprocess arXiv CS/ML dataset into time buckets."""
    ensure_dirs()
    
    # Load configs
    data_config = load_config("data", CONFIG_DIR)
    
    # Override max_per_bucket if provided
    data_config["sampling"]["max_per_bucket"] = max_per_bucket
    
    # Override output directory if provided
    output_path = Path(output_dir) if output_dir else DATA_PROCESSED_DIR
    
    # Run pipeline
    typer.echo("Starting data preparation pipeline...")
    report = run_data_pipeline(data_config, output_dir=output_path)
    
    typer.echo(f"✓ Data preparation complete. Report saved to: {output_path / 'report.json'}")


@app.command()
def train_adapters(
    epochs: int = typer.Option(2, help="Training epochs"),
    lora_r: int = typer.Option(16, help="LoRA rank"),
    cross_period_negatives: bool = typer.Option(True, help="Use cross-period negatives"),
    data_dir: Optional[str] = typer.Option(None, help="Override data directory"),
    output_dir: Optional[str] = typer.Option(None, help="Override output directory"),
) -> None:
    """Train time-bucket LoRA adapters on frozen sentence encoder."""
    ensure_dirs()
    
    # Load configs
    model_config = load_config("model", CONFIG_DIR)
    train_config = load_config("train", CONFIG_DIR)
    
    # Override config values
    if lora_r:
        model_config["lora"]["r"] = lora_r
    if epochs:
        train_config["training"]["epochs"] = epochs
    if cross_period_negatives is not None:
        train_config["negatives"]["cross_period_negatives"] = cross_period_negatives
    
    # Combine configs
    config = {
        "model": model_config,
        "training": train_config["training"],
        "negatives": train_config["negatives"],
    }
    
    # Set paths
    data_path = Path(data_dir) if data_dir else DATA_PROCESSED_DIR
    output_path = Path(output_dir) if output_dir else ADAPTERS_DIR
    
    # Check data exists
    if not data_path.exists():
        typer.echo(
            f"❌ Data directory not found: {data_path}\n"
            f"Run 'python -m temporal_lora.cli prepare-data' first."
        )
        raise typer.Exit(1)
    
    # Train
    typer.echo(f"Training LoRA adapters with r={lora_r}, epochs={epochs}")
    typer.echo(f"Data directory: {data_path}")
    typer.echo(f"Output directory: {output_path}")
    
    metrics = train_all_buckets(data_path, output_path, config)
    
    # Print summary
    typer.echo("\n" + "=" * 60)
    typer.echo("TRAINING SUMMARY")
    typer.echo("=" * 60)
    for bucket_name, bucket_metrics in metrics.items():
        typer.echo(f"\nBucket: {bucket_name}")
        typer.echo(f"  Examples: {bucket_metrics['train_examples']}")
        typer.echo(f"  Epochs: {bucket_metrics['epochs']}")
        typer.echo(f"  Time: {bucket_metrics['total_time']:.2f}s")
        typer.echo(f"  Output: {bucket_metrics['output_dir']}")
    typer.echo("=" * 60)
    
    typer.echo("\n✓ Training complete!")


@app.command()
def build_indexes(
    checkpoint_dir: Optional[str] = typer.Option(None, help="Path to adapter checkpoints"),
) -> None:
    """Build FAISS IndexFlatIP for each time bucket."""
    raise NotImplementedError("build-indexes not yet implemented")


@app.command()
def evaluate(
    mode: str = typer.Option("multi-index", help="Retrieval mode: time-select or multi-index"),
    merge: str = typer.Option("softmax", help="Multi-index merge: softmax/mean/max/rrf"),
    scenarios: Optional[str] = typer.Option(
        "within,cross,all", help="Comma-separated eval scenarios: within/cross/all"
    ),
    config: Optional[str] = typer.Option(None, help="Override eval config path"),
) -> None:
    """Evaluate retrieval quality with NDCG@10, Recall@10/100, MRR."""
    raise NotImplementedError("evaluate not yet implemented")


@app.command()
def visualize(
    output_dir: Optional[str] = typer.Option(None, help="Override output directory"),
) -> None:
    """Generate heatmaps, UMAP, and drift trajectories."""
    raise NotImplementedError("visualize not yet implemented")


@app.command()
def export_deliverables() -> None:
    """Consolidate results, figures, and repro info into deliverables/."""
    raise NotImplementedError("export-deliverables not yet implemented")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
