"""Command-line interface for Temporal LoRA pipeline."""

from pathlib import Path
from typing import Optional, List

import typer

from .data.pipeline import run_data_pipeline
from .train.trainer import train_all_buckets
from .eval.encoder import encode_and_cache_bucket
from .eval.indexes import build_bucket_indexes
from .eval.evaluate import (
    run_full_evaluation,
    run_temperature_sweep,
)
from .eval.efficiency import generate_efficiency_summary
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
    mode: str = typer.Option(
        "lora",
        help="Training mode: lora (default), full_ft, or seq_ft"
    ),
    epochs: int = typer.Option(2, help="Training epochs"),
    lora_r: int = typer.Option(16, help="LoRA rank (for lora mode)"),
    hard_temporal_negatives: bool = typer.Option(
        False,
        "--hard-temporal-negatives/--no-hard-temporal-negatives",
        help="Use hard temporal negatives from adjacent bins",
    ),
    neg_k: int = typer.Option(4, help="Number of hard negatives per positive"),
    cross_period_negatives: bool = typer.Option(
        True,
        "--cross-period-negatives/--no-cross-period-negatives",
        help="Use cross-period negatives (deprecated, use --hard-temporal-negatives)",
    ),
    data_dir: Optional[str] = typer.Option(None, help="Override data directory"),
    output_dir: Optional[str] = typer.Option(None, help="Override output directory"),
) -> None:
    """Train time-bucket models (LoRA adapters or full fine-tuning)."""
    ensure_dirs()
    
    # Validate mode
    valid_modes = ["lora", "full_ft", "seq_ft"]
    if mode not in valid_modes:
        typer.echo(f"❌ Invalid mode: {mode}. Must be one of: {', '.join(valid_modes)}")
        raise typer.Exit(1)
    
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
    
    if output_dir:
        output_path = Path(output_dir)
    else:
        # Default output based on mode
        if mode == "lora":
            output_path = ADAPTERS_DIR
        elif mode == "full_ft":
            output_path = PROJECT_ROOT / "artifacts" / "full_ft"
        elif mode == "seq_ft":
            output_path = PROJECT_ROOT / "artifacts" / "seq_ft"
    
    # Check data exists
    if not data_path.exists():
        typer.echo(
            f"❌ Data directory not found: {data_path}\n"
            f"Run 'python -m temporal_lora.cli prepare-data' first."
        )
        raise typer.Exit(1)
    
    # Train
    typer.echo(f"Training {mode.upper()} with:")
    typer.echo(f"  Mode: {mode}")
    if mode == "lora":
        typer.echo(f"  LoRA rank: {lora_r}")
    typer.echo(f"  Epochs: {epochs}")
    typer.echo(f"  Hard temporal negatives: {hard_temporal_negatives}")
    if hard_temporal_negatives:
        typer.echo(f"  Negatives per positive: {neg_k}")
    typer.echo(f"  Data directory: {data_path}")
    typer.echo(f"  Output directory: {output_path}")
    typer.echo()
    
    metrics = train_all_buckets(
        data_path,
        output_path,
        config,
        mode=mode,
        use_hard_negatives=hard_temporal_negatives,
        neg_k=neg_k,
    )
    
    # Print summary
    typer.echo("\n" + "=" * 60)
    typer.echo("TRAINING SUMMARY")
    typer.echo("=" * 60)
    for bucket_name, bucket_metrics in metrics.items():
        typer.echo(f"\nBucket: {bucket_name}")
        typer.echo(f"  Mode: {bucket_metrics['mode']}")
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
def evaluate_all_modes(
    modes: str = typer.Option(
        "baseline_frozen,lora",
        help="Comma-separated list of modes to evaluate"
    ),
    temperature_sweep: bool = typer.Option(
        True,
        "--temperature-sweep/--no-temperature-sweep",
        help="Run temperature sweep for multi-index merge"
    ),
    temperatures: str = typer.Option(
        "1.5,2.0,3.0",
        help="Comma-separated temperatures for sweep"
    ),
    cache_dir: Optional[str] = typer.Option(None, help="Cache directory"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory"),
) -> None:
    """Run comprehensive evaluation across all training modes."""
    ensure_dirs()
    
    # Load configs
    data_config = load_config("data", CONFIG_DIR)
    
    # Parse inputs
    mode_list = [m.strip() for m in modes.split(",")]
    temp_list = [float(t.strip()) for t in temperatures.split(",")]
    buckets = [b["name"] for b in data_config["buckets"]]
    
    # Set paths
    data_path = DATA_PROCESSED_DIR
    cache_path = Path(cache_dir) if cache_dir else PROJECT_ROOT / ".cache"
    output_path = Path(output_dir) if output_dir else DELIVERABLES_RESULTS_DIR
    
    typer.echo("Running full evaluation across modes:")
    typer.echo(f"  Modes: {', '.join(mode_list)}")
    typer.echo(f"  Buckets: {', '.join(buckets)}")
    typer.echo(f"  Temperature sweep: {temperature_sweep}")
    if temperature_sweep:
        typer.echo(f"  Temperatures: {temp_list}")
    typer.echo()
    
    # Run evaluation
    run_full_evaluation(
        data_dir=data_path,
        cache_dir=cache_path,
        modes=mode_list,
        buckets=buckets,
        output_dir=output_path,
        run_temperature_sweep_flag=temperature_sweep,
        temperatures=temp_list,
    )
    
    typer.echo(f"\n✓ Full evaluation complete! Results saved to: {output_path}")


@app.command()
def efficiency_summary(
    modes: str = typer.Option(
        "baseline_frozen,lora,full_ft,seq_ft",
        help="Comma-separated list of modes"
    ),
    adapters_dir: Optional[str] = typer.Option(None, help="LoRA adapters directory"),
    full_ft_dir: Optional[str] = typer.Option(None, help="Full FT directory"),
    seq_ft_dir: Optional[str] = typer.Option(None, help="Sequential FT directory"),
    output: Optional[str] = typer.Option(None, help="Output CSV path"),
) -> None:
    """Generate efficiency summary table (params, size, runtime)."""
    ensure_dirs()
    
    # Load configs
    data_config = load_config("data", CONFIG_DIR)
    model_config = load_config("model", CONFIG_DIR)
    
    # Parse inputs
    mode_list = [m.strip() for m in modes.split(",")]
    buckets = [b["name"] for b in data_config["buckets"]]
    base_model = model_config["base_model"]["name"]
    
    # Set paths
    adapters_path = Path(adapters_dir) if adapters_dir else ADAPTERS_DIR
    full_ft_path = Path(full_ft_dir) if full_ft_dir else PROJECT_ROOT / "artifacts" / "full_ft"
    seq_ft_path = Path(seq_ft_dir) if seq_ft_dir else PROJECT_ROOT / "artifacts" / "seq_ft"
    output_path = Path(output) if output else DELIVERABLES_RESULTS_DIR / "efficiency_summary.csv"
    
    typer.echo("Generating efficiency summary...")
    typer.echo(f"  Modes: {', '.join(mode_list)}")
    typer.echo(f"  Base model: {base_model}")
    typer.echo()
    
    # Generate summary
    df = generate_efficiency_summary(
        base_model_name=base_model,
        modes=mode_list,
        adapters_dir=adapters_path,
        full_ft_dir=full_ft_path,
        seq_ft_dir=seq_ft_path,
        buckets=buckets,
        output_path=output_path,
    )
    
    typer.echo(f"\n✓ Efficiency summary saved to: {output_path}")


@app.command()
def visualize(
    results_dir: Optional[str] = typer.Option(
        None, help="Directory with evaluation results"
    ),
    embeddings_dir: Optional[str] = typer.Option(
        None, help="Path to embeddings directory for UMAP"
    ),
    output_dir: Optional[str] = typer.Option(
        None, help="Output directory for figures"
    ),
    baseline_mode: str = typer.Option(
        "baseline_frozen", help="Name of baseline mode"
    ),
    lora_mode: str = typer.Option(
        "lora", help="Name of LoRA mode"
    ),
) -> None:
    """Generate heatmaps and UMAP visualizations."""
    from .viz.plots import visualize_results
    
    ensure_dirs()
    
    # Default paths
    results_path = Path(results_dir) if results_dir else DELIVERABLES_RESULTS_DIR
    emb_dir = Path(embeddings_dir) if embeddings_dir else PROJECT_ROOT / ".cache" / "embeddings" / lora_mode
    output_path = Path(output_dir) if output_dir else PROJECT_ROOT / "deliverables" / "figures"
    
    typer.echo("Generating visualizations...")
    typer.echo(f"Results directory: {results_path}")
    typer.echo(f"Embeddings: {emb_dir}")
    typer.echo(f"Output: {output_path}")
    
    visualize_results(
        results_dir=results_path,
        embeddings_dir=emb_dir,
        output_dir=output_path,
        baseline_mode=baseline_mode,
        lora_mode=lora_mode,
    )
    
    typer.echo(f"\n✓ Visualizations saved to: {output_path}")


@app.command()
def drift_trajectories(
    terms: str = typer.Option(
        "transformer,BERT,LLM",
        help="Comma-separated terms to track"
    ),
    contexts_per_term: int = typer.Option(
        50, help="Number of contexts per term per bucket"
    ),
    data_dir: Optional[str] = typer.Option(None, help="Data directory"),
    adapters_dir: Optional[str] = typer.Option(None, help="LoRA adapters directory"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory"),
    use_lora: bool = typer.Option(True, "--lora/--baseline", help="Use LoRA adapters"),
) -> None:
    """Generate term drift trajectory visualization."""
    from .viz.drift_trajectories import run_drift_analysis
    
    ensure_dirs()
    
    # Load configs
    data_config = load_config("data", CONFIG_DIR)
    model_config = load_config("model", CONFIG_DIR)
    
    # Parse inputs
    term_list = [t.strip() for t in terms.split(",")]
    buckets = [b["name"] for b in data_config["buckets"]]
    base_model = model_config["base_model"]["name"]
    
    # Set paths
    data_path = Path(data_dir) if data_dir else DATA_PROCESSED_DIR
    adapters_path = Path(adapters_dir) if adapters_dir else ADAPTERS_DIR
    output_path = Path(output_dir) if output_dir else PROJECT_ROOT / "deliverables" / "figures"
    
    typer.echo("Generating drift trajectories...")
    typer.echo(f"  Terms: {', '.join(term_list)}")
    typer.echo(f"  Buckets: {', '.join(buckets)}")
    typer.echo(f"  Contexts per term: {contexts_per_term}")
    typer.echo(f"  Use LoRA: {use_lora}")
    
    run_drift_analysis(
        data_dir=data_path,
        adapters_dir=adapters_path,
        base_model_name=base_model,
        buckets=buckets,
        terms=term_list,
        output_dir=output_path,
        contexts_per_term=contexts_per_term,
        use_lora=use_lora,
    )
    
    typer.echo(f"\n✓ Drift trajectories saved to: {output_path}")


@app.command()
def quick_ablation(
    bucket: Optional[str] = typer.Option(None, help="Bucket to use for ablation"),
    ranks: str = typer.Option("8,16,32", help="Comma-separated LoRA ranks"),
    max_eval: int = typer.Option(500, help="Max eval samples"),
    epochs: int = typer.Option(1, help="Training epochs"),
    data_dir: Optional[str] = typer.Option(None, help="Data directory"),
    output: Optional[str] = typer.Option(None, help="Output CSV path"),
) -> None:
    """Run quick ablation study on LoRA hyperparameters."""
    from .ablate.quick import run_quick_ablation
    
    ensure_dirs()
    
    # Load configs
    data_config = load_config("data", CONFIG_DIR)
    model_config = load_config("model", CONFIG_DIR)
    
    # Parse inputs
    rank_list = [int(r.strip()) for r in ranks.split(",")]
    buckets = [b["name"] for b in data_config["buckets"]]
    bucket_name = bucket if bucket else buckets[0]
    base_model = model_config["base_model"]["name"]
    
    # Set paths
    data_path = Path(data_dir) if data_dir else DATA_PROCESSED_DIR
    output_path = Path(output) if output else DELIVERABLES_RESULTS_DIR / "quick_ablation.csv"
    
    typer.echo("Running quick ablation...")
    typer.echo(f"  Bucket: {bucket_name}")
    typer.echo(f"  Ranks: {rank_list}")
    typer.echo(f"  Epochs: {epochs}")
    typer.echo(f"  Max eval: {max_eval}")
    
    # Run ablation
    results_df = run_quick_ablation(
        data_dir=data_path,
        base_model_name=base_model,
        bucket_name=bucket_name,
        output_path=output_path,
        ranks=rank_list,
        epochs=epochs,
        max_eval_samples=max_eval,
    )
    
    typer.echo(f"\n✓ Ablation results saved to: {output_path}")
    typer.echo(f"✓ Summary saved to: {output_path.parent / 'ablation_summary.md'}")


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


@app.command()
def benchmark(
    baseline_models: Optional[str] = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2,sentence-transformers/all-mpnet-base-v2",
        help="Comma-separated list of baseline models to compare"
    ),
    lora_adapters_dir: Optional[str] = typer.Option(
        None,
        help="Path to trained LoRA adapters directory"
    ),
    buckets: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of buckets to evaluate (e.g., 'bucket_0,bucket_1')"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        help="Output directory for benchmark results"
    ),
    generate_report: bool = typer.Option(
        True,
        "--report/--no-report",
        help="Generate comprehensive report"
    ),
) -> None:
    """Run comprehensive benchmark comparison against baseline models."""
    from .benchmark import run_benchmark
    from .benchmark.report_generator import generate_report as gen_report
    
    ensure_dirs()
    
    # Parse baseline models
    baseline_list = baseline_models.split(",") if baseline_models else None
    
    # Parse buckets
    bucket_list = buckets.split(",") if buckets else None
    
    # Set paths
    lora_path = Path(lora_adapters_dir) if lora_adapters_dir else ADAPTERS_DIR
    output_path = Path(output_dir) if output_dir else DELIVERABLES_RESULTS_DIR / "benchmark"
    
    typer.echo("\n" + "="*60)
    typer.echo("Starting Benchmark Comparison")
    typer.echo("="*60)
    typer.echo(f"Baseline models: {baseline_list}")
    typer.echo(f"LoRA adapters: {lora_path}")
    typer.echo(f"Buckets: {bucket_list or 'all'}")
    typer.echo(f"Output: {output_path}")
    typer.echo("="*60 + "\n")
    
    # Run benchmark
    df = run_benchmark(
        data_dir=DATA_PROCESSED_DIR,
        output_dir=output_path,
        lora_adapters_dir=lora_path if lora_path.exists() else None,
        buckets=bucket_list,
    )
    
    typer.echo(f"\n✅ Benchmark results saved to: {output_path / 'benchmark_comparison.csv'}")
    
    # Generate report
    if generate_report:
        typer.echo("\n📝 Generating comprehensive report...")
        report_path = gen_report(
            results_csv=output_path / "benchmark_comparison.csv",
            output_dir=output_path,
            baseline=baseline_list[0].split("/")[-1] if baseline_list else "all-MiniLM-L6-v2"
        )
        typer.echo(f"✅ Report generated: {report_path}")
        typer.echo(f"✅ Visualizations saved to: {output_path / 'figures'}")
    
    typer.echo("\n" + "="*60)
    typer.echo("Benchmark Complete!")
    typer.echo("="*60)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
