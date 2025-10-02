"""Command-line interface for Temporal LoRA pipeline."""

from pathlib import Path
from typing import Optional

import typer

from .data.pipeline import run_data_pipeline
from .train.trainer import train_all_buckets
from .eval.encoder import encode_and_cache_bucket
from .eval.indexes import build_bucket_indexes
from .eval.evaluate import run_evaluation
from .utils.env import dump_environment
from .utils.io import load_config
from .utils.paths import (
    ADAPTERS_DIR,
    CONFIG_DIR,
    DATA_PROCESSED_DIR,
    DELIVERABLES_REPRO_DIR,
    DELIVERABLES_RESULTS_DIR,
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
    balance_per_bin: bool = typer.Option(
        True, "--balance-per-bin/--no-balance-per-bin", help="Enforce equal counts across bins"
    ),
    bins: Optional[str] = typer.Option(
        None, help="Custom bins (comma-separated, e.g., 'pre2016:null-2015,2016-2018:2016-2018')"
    ),
    output_dir: Optional[str] = typer.Option(None, help="Override output directory"),
) -> None:
    """Download and preprocess arXiv CS/ML dataset into time buckets."""
    ensure_dirs()
    
    # Load configs
    data_config = load_config("data", CONFIG_DIR)
    
    # Override max_per_bucket if provided
    data_config["sampling"]["max_per_bucket"] = max_per_bucket
    data_config["sampling"]["balance_per_bin"] = balance_per_bin
    
    # Override bins if provided
    if bins:
        typer.echo(f"Using custom bins: {bins}")
        # Parse custom bins specification
        custom_buckets = []
        for bin_spec in bins.split(","):
            name, year_range = bin_spec.split(":")
            start_year, end_year = year_range.split("-")
            custom_buckets.append({
                "name": name.strip(),
                "range": [
                    None if start_year.strip().lower() == "null" else int(start_year.strip()),
                    None if end_year.strip().lower() == "null" else int(end_year.strip()),
                ]
            })
        data_config["buckets"] = custom_buckets
    
    # Override output directory if provided
    output_path = Path(output_dir) if output_dir else DATA_PROCESSED_DIR
    
    # Run pipeline
    typer.echo("Starting data preparation pipeline...")
    typer.echo(f"Balance per bin: {balance_per_bin}")
    typer.echo(f"Max per bucket: {max_per_bucket}")
    typer.echo(f"Number of bins: {len(data_config['buckets'])}")
    report = run_data_pipeline(data_config, output_dir=output_path)
    
    typer.echo(f"✓ Data preparation complete. Report saved to: {output_path / 'report.json'}")


@app.command()
def train_adapters(
    epochs: int = typer.Option(2, help="Training epochs"),
    lora_r: int = typer.Option(16, help="LoRA rank"),
    cross_period_negatives: bool = typer.Option(
        True,
        "--cross-period-negatives/--no-cross-period-negatives",
        help="Use cross-period negatives",
    ),
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
    use_lora: bool = typer.Option(True, "--lora/--baseline", help="Use LoRA adapters or baseline"),
    adapters_dir: Optional[str] = typer.Option(None, help="Path to adapter checkpoints"),
    data_dir: Optional[str] = typer.Option(None, help="Data directory"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory for indexes"),
) -> None:
    """Build FAISS IndexFlatIP for each time bucket."""
    ensure_dirs()
    
    # Load configs
    data_config = load_config("data", CONFIG_DIR)
    model_config = load_config("model", CONFIG_DIR)
    
    # Set paths
    data_path = Path(data_dir) if data_dir else DATA_PROCESSED_DIR
    adapters_path = Path(adapters_dir) if adapters_dir else ADAPTERS_DIR
    
    embeddings_suffix = "lora" if use_lora else "baseline"
    embeddings_dir = PROJECT_ROOT / "models" / f"embeddings_{embeddings_suffix}"
    indexes_dir = PROJECT_ROOT / "models" / f"indexes_{embeddings_suffix}"
    
    if output_dir:
        indexes_dir = Path(output_dir)
    
    base_model = model_config["base_model"]["name"]
    buckets = [b["name"] for b in data_config["buckets"]]
    
    typer.echo(f"Building {'LoRA' if use_lora else 'baseline'} indexes")
    typer.echo(f"Base model: {base_model}")
    typer.echo(f"Buckets: {', '.join(buckets)}")
    
    # Encode and cache embeddings for each bucket
    for bucket_name in buckets:
        bucket_data_path = data_path / bucket_name
        adapter_dir = adapters_path / bucket_name if use_lora else None
        
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Processing bucket: {bucket_name}")
        typer.echo(f"{'='*60}")
        
        encode_and_cache_bucket(
            bucket_name=bucket_name,
            bucket_data_path=bucket_data_path,
            adapter_dir=adapter_dir,
            base_model_name=base_model,
            output_dir=embeddings_dir,
            use_lora=use_lora,
        )
    
    # Build FAISS indexes
    typer.echo(f"\n{'='*60}")
    typer.echo("Building FAISS indexes...")
    typer.echo(f"{'='*60}")
    
    index_paths = build_bucket_indexes(embeddings_dir, indexes_dir, buckets)
    
    typer.echo(f"\n✓ Indexes built: {len(index_paths)} buckets")
    typer.echo(f"Output directory: {indexes_dir}")


@app.command()
def evaluate(
    mode: str = typer.Option("multi-index", help="Retrieval mode: time-select or multi-index"),
    merge: str = typer.Option("softmax", help="Multi-index merge: softmax/mean/max/rrf"),
    temperature: float = typer.Option(2.0, help="Temperature for softmax merge"),
    scenarios: Optional[str] = typer.Option(
        "within,cross,all", help="Comma-separated eval scenarios: within/cross/all"
    ),
    use_lora: bool = typer.Option(True, "--lora/--baseline", help="Evaluate LoRA or baseline"),
    data_dir: Optional[str] = typer.Option(None, help="Data directory"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory for results"),
) -> None:
    """Evaluate retrieval quality with NDCG@10, Recall@10/100, MRR."""
    ensure_dirs()
    
    # Load configs
    data_config = load_config("data", CONFIG_DIR)
    
    # Set paths
    data_path = Path(data_dir) if data_dir else DATA_PROCESSED_DIR
    
    embeddings_suffix = "lora" if use_lora else "baseline"
    embeddings_dir = PROJECT_ROOT / "models" / f"embeddings_{embeddings_suffix}"
    indexes_dir = PROJECT_ROOT / "models" / f"indexes_{embeddings_suffix}"
    
    output_path = Path(output_dir) if output_dir else DELIVERABLES_RESULTS_DIR
    
    # Parse scenarios
    scenario_list = [s.strip() for s in scenarios.split(",")]
    buckets = [b["name"] for b in data_config["buckets"]]
    
    typer.echo(f"Evaluating {'LoRA' if use_lora else 'baseline'} system")
    typer.echo(f"Mode: {mode}")
    typer.echo(f"Scenarios: {', '.join(scenario_list)}")
    if mode == "multi-index":
        typer.echo(f"Merge: {merge} (temperature={temperature})")
    
    # Run evaluation
    results = run_evaluation(
        data_dir=data_path,
        embeddings_dir=embeddings_dir,
        indexes_dir=indexes_dir,
        buckets=buckets,
        scenarios=scenario_list,
        mode=mode,
        merge_strategy=merge,
        temperature=temperature,
        output_dir=output_path,
    )
    
    # Print summary
    typer.echo(f"\n{'='*60}")
    typer.echo("EVALUATION RESULTS")
    typer.echo(f"{'='*60}")
    for key, metrics in results.items():
        typer.echo(f"\n{key}:")
        for metric_name, value in metrics.items():
            typer.echo(f"  {metric_name}: {value:.4f}")
    
    typer.echo(f"\n✓ Results saved to: {output_path}")


@app.command()
def visualize(
    baseline_results: Optional[str] = typer.Option(
        None, help="Path to baseline results CSV"
    ),
    lora_results: Optional[str] = typer.Option(
        None, help="Path to LoRA results CSV"
    ),
    embeddings_dir: Optional[str] = typer.Option(
        None, help="Path to embeddings directory for UMAP"
    ),
    output_dir: Optional[str] = typer.Option(
        None, help="Output directory for figures"
    ),
) -> None:
    """Generate heatmaps and UMAP visualizations."""
    from .viz.plots import visualize_results
    
    ensure_dirs()
    
    # Default paths
    baseline_path = Path(baseline_results) if baseline_results else DELIVERABLES_RESULTS_DIR / "baseline_results.csv"
    lora_path = Path(lora_results) if lora_results else DELIVERABLES_RESULTS_DIR / "lora_results.csv"
    emb_dir = Path(embeddings_dir) if embeddings_dir else PROJECT_ROOT / "models" / "embeddings_lora"
    output_path = Path(output_dir) if output_dir else PROJECT_ROOT / "deliverables" / "figures"
    
    typer.echo("Generating visualizations...")
    typer.echo(f"Baseline results: {baseline_path}")
    typer.echo(f"LoRA results: {lora_path}")
    typer.echo(f"Embeddings: {emb_dir}")
    typer.echo(f"Output: {output_path}")
    
    visualize_results(
        baseline_results_path=baseline_path,
        lora_results_path=lora_path,
        embeddings_dir=emb_dir,
        output_dir=output_path,
    )
    
    typer.echo(f"\n✓ Visualizations saved to: {output_path}")


@app.command()
def export_deliverables() -> None:
    """Consolidate results, figures, and repro info into deliverables/."""
    import subprocess
    import sys
    
    ensure_dirs()
    
    export_script = PROJECT_ROOT / "scripts" / "export_results.py"
    
    if not export_script.exists():
        typer.echo(f"❌ Export script not found: {export_script}")
        raise typer.Exit(1)
    
    typer.echo("Running export script...\n")
    
    result = subprocess.run(
        [sys.executable, str(export_script)],
        cwd=PROJECT_ROOT,
    )
    
    if result.returncode != 0:
        typer.echo("\n❌ Export failed")
        raise typer.Exit(1)
    
    typer.echo("\n✓ Deliverables exported successfully")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
