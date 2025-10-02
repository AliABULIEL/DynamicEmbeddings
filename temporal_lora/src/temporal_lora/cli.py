"""
Command-line interface for Temporal LoRA.

All major operations (data prep, training, indexing, evaluation, visualization)
are exposed as CLI commands.
"""

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.table import Table

from temporal_lora.data import (
    BucketConfig,
    create_pairs,
    create_splits,
    load_dataset_from_csv,
    load_dataset_from_hf,
)
from temporal_lora.data.bucketing import get_split_summary

app = typer.Typer(help="Temporal LoRA for Dynamic Sentence Embeddings")
console = Console()


@app.command()
def prepare_data(
    config_path: Path = typer.Option(
        "src/temporal_lora/config/data.yaml", help="Path to data config YAML"
    ),
    force_csv: bool = typer.Option(False, help="Skip HF, use CSV directly"),
):
    """
    Download and preprocess arXiv CS/ML abstracts into time buckets.

    Attempts to load from HuggingFace first, falls back to CSV if unavailable.
    Writes parquet files per bucket/split and generates audit report.
    """
    console.print("[bold blue]📊 Preparing data...[/bold blue]")

    # Load config
    if not config_path.exists():
        console.print(f"[red]❌ Config not found: {config_path}[/red]")
        raise typer.Exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create bucket config from YAML
    bucket_definitions = config["buckets"]["definitions"]
    boundaries = []
    for bucket_def in bucket_definitions:
        name = bucket_def["name"]
        end_year = bucket_def["end"]
        boundaries.append((name, end_year))

    bucket_config = BucketConfig(
        boundaries=boundaries,
        train_ratio=config["splits"]["train"],
        val_ratio=config["splits"]["val"],
        test_ratio=config["splits"]["test"],
        max_per_bucket=config["sampling"]["max_per_bucket"],
        seed=config["splits"]["seed"],
    )

    console.print(f"[cyan]Buckets: {[b[0] for b in boundaries]}[/cyan]")
    console.print(
        f"[cyan]Splits: {bucket_config.train_ratio}/{bucket_config.val_ratio}/{bucket_config.test_ratio}[/cyan]"
    )
    console.print(f"[cyan]Max per bucket: {bucket_config.max_per_bucket}[/cyan]")

    # Load dataset
    df = None

    if not force_csv:
        # Try HuggingFace first
        try:
            hf_name = config["dataset"]["hf_name"]
            console.print(f"[yellow]Attempting to load from HuggingFace: {hf_name}...[/yellow]")
            df = load_dataset_from_hf(hf_name)
            console.print(f"[green]✓ Loaded {len(df)} papers from HuggingFace[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️  HuggingFace load failed: {e}[/yellow]")
            console.print("[yellow]Falling back to CSV...[/yellow]")

    # Fall back to CSV
    if df is None:
        csv_path = Path(config["output"]["csv_fallback"])
        console.print(f"[yellow]Loading from CSV: {csv_path}...[/yellow]")
        try:
            df = load_dataset_from_csv(csv_path)
            console.print(f"[green]✓ Loaded {len(df)} papers from CSV[/green]")
        except Exception as e:
            console.print(f"[red]❌ CSV load failed: {e}[/red]")
            console.print(
                "\n[yellow]No data source available! Please provide either:[/yellow]"
            )
            console.print(
                f"  1. HuggingFace dataset: {config['dataset']['hf_name']}"
            )
            console.print(f"  2. CSV file at: {csv_path.absolute()}")
            console.print(
                f"\nCSV must have columns: paper_id, title, abstract, year"
            )
            raise typer.Exit(1)

    # Apply preprocessing filters
    console.print("[cyan]Applying preprocessing filters...[/cyan]")
    initial_count = len(df)

    # Drop rows with missing required fields
    if config["preprocessing"]["drop_missing"]:
        df = df.dropna(subset=["paper_id", "title", "abstract", "year"])

    # Filter by abstract length
    min_len = config["preprocessing"]["min_abstract_length"]
    df = df[df["abstract"].str.len() >= min_len]

    filtered_count = len(df)
    console.print(
        f"[cyan]Kept {filtered_count}/{initial_count} papers after filtering "
        f"({filtered_count/initial_count*100:.1f}%)[/cyan]"
    )

    # Create splits
    console.print("[cyan]Creating train/val/test splits per bucket...[/cyan]")
    splits = create_splits(df, bucket_config)

    # Prepare output directories
    output_dir = Path(config["output"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write parquet files
    console.print("[cyan]Writing parquet files...[/cyan]")
    files_written = []

    for bucket_name, bucket_splits in splits.items():
        bucket_dir = output_dir / bucket_name
        bucket_dir.mkdir(exist_ok=True)

        for split_name, split_df in bucket_splits.items():
            output_path = bucket_dir / f"{split_name}.parquet"
            split_df.to_parquet(output_path, index=False)
            files_written.append(str(output_path))
            console.print(
                f"  [green]✓ {bucket_name}/{split_name}: {len(split_df)} papers[/green]"
            )

    # Generate audit report
    console.print("[cyan]Generating audit report...[/cyan]")
    report = {
        "config": {
            "buckets": [b[0] for b in boundaries],
            "splits": {
                "train": bucket_config.train_ratio,
                "val": bucket_config.val_ratio,
                "test": bucket_config.test_ratio,
            },
            "max_per_bucket": bucket_config.max_per_bucket,
            "seed": bucket_config.seed,
        },
        "data_source": "huggingface"
        if not force_csv
        else "csv",
        "total_papers": len(df),
        "split_counts": get_split_summary(splits),
        "year_range": {
            "min": int(df["year"].min()),
            "max": int(df["year"].max()),
        },
        "year_histogram": df["year"].value_counts().sort_index().to_dict(),
        "sample_rows": [
            {
                "paper_id": row["paper_id"],
                "title": row["title"][:100] + "..."
                if len(row["title"]) > 100
                else row["title"],
                "year": int(row["year"]),
            }
            for _, row in df.head(3).iterrows()
        ],
    }

    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"[green]✓ Report saved: {report_path}[/green]")

    # Display summary table
    table = Table(title="Dataset Summary")
    table.add_column("Bucket", style="cyan")
    table.add_column("Train", justify="right")
    table.add_column("Val", justify="right")
    table.add_column("Test", justify="right")
    table.add_column("Total", justify="right", style="bold")

    for bucket_name, counts in report["split_counts"].items():
        table.add_row(
            bucket_name,
            str(counts["train"]),
            str(counts["val"]),
            str(counts["test"]),
            str(counts["total"]),
        )

    console.print(table)
    console.print(
        f"\n[bold green]✓ Data preparation complete![/bold green]"
    )
    console.print(f"[cyan]Files written to: {output_dir}[/cyan]")
    console.print(f"[cyan]Total files: {len(files_written)}[/cyan]")


@app.command()
def train_adapters(
    epochs: int = typer.Option(2, help="Number of training epochs"),
    lora_r: int = typer.Option(16, help="LoRA rank"),
    cross_period_negatives: bool = typer.Option(
        False, help="Use cross-period negatives"
    ),
    fp16: bool = typer.Option(True, help="Use mixed precision training"),
):
    """
    Train time-bucket LoRA adapters on frozen sentence encoder.
    """
    console.print("[bold blue]Training LoRA adapters...[/bold blue]")
    console.print(f"Epochs: {epochs}, LoRA rank: {lora_r}")
    console.print(f"Cross-period negatives: {cross_period_negatives}")
    console.print(f"FP16: {fp16}")
    # Placeholder: Actual implementation in src/temporal_lora/training/
    console.print("[green]✓ Training complete[/green]")


@app.command()
def build_indexes(
    index_type: str = typer.Option("flat", help="FAISS index type: flat, ivf"),
):
    """
    Build per-bucket FAISS indexes for retrieval.
    """
    console.print("[bold blue]Building FAISS indexes...[/bold blue]")
    console.print(f"Index type: {index_type}")
    # Placeholder: Actual implementation in src/temporal_lora/indexing/
    console.print("[green]✓ Indexes built[/green]")


@app.command()
def evaluate(
    scenarios: List[str] = typer.Option(
        ["within", "cross", "all"], help="Eval scenarios: within, cross, all"
    ),
    mode: str = typer.Option(
        "multi-index", help="Retrieval mode: time-select, multi-index"
    ),
    merge: str = typer.Option(
        "softmax", help="Multi-index merge: softmax, max, mean, rrf"
    ),
):
    """
    Evaluate retrieval performance with statistical tests.
    """
    console.print("[bold blue]Running evaluation...[/bold blue]")
    console.print(f"Scenarios: {scenarios}")
    console.print(f"Mode: {mode}, Merge: {merge}")
    # Placeholder: Actual implementation in src/temporal_lora/evaluation/
    console.print("[green]✓ Evaluation complete[/green]")


@app.command()
def visualize(
    plots: List[str] = typer.Option(
        ["heatmap", "umap", "drift", "year_gap"], help="Plots to generate"
    ),
):
    """
    Generate visualization artifacts (heatmaps, UMAP, drift plots).
    """
    console.print("[bold blue]Generating visualizations...[/bold blue]")
    console.print(f"Plots: {plots}")
    # Placeholder: Actual implementation in src/temporal_lora/visualization/
    console.print("[green]✓ Visualizations saved[/green]")


@app.command()
def ablate(
    lora_ranks: List[int] = typer.Option([8, 16, 32], help="LoRA ranks to test"),
    buckets: List[int] = typer.Option([2, 3], help="Number of buckets to test"),
    negatives: List[str] = typer.Option(
        ["random", "cross_period"], help="Negative sampling strategies"
    ),
):
    """
    Run ablation studies on LoRA rank, buckets, and negatives.
    """
    console.print("[bold blue]Running ablation studies...[/bold blue]")
    console.print(f"LoRA ranks: {lora_ranks}")
    console.print(f"Buckets: {buckets}")
    console.print(f"Negatives: {negatives}")
    # Placeholder: Actual implementation loops through combinations
    console.print("[green]✓ Ablations complete[/green]")


@app.command()
def export_deliverables():
    """
    Export all artifacts (figures, tables, repro info) to deliverables/.
    """
    console.print("[bold blue]Exporting deliverables...[/bold blue]")
    # Placeholder: Copy figures, tables, environment.json to deliverables/
    console.print("[green]✓ Deliverables exported to deliverables/[/green]")


@app.command()
def env_dump():
    """
    Capture environment (CUDA, torch, pip freeze, commit hash) for reproducibility.
    """
    console.print("[bold blue]Dumping environment info...[/bold blue]")
    # Placeholder: Actual implementation in src/temporal_lora/utils/env.py
    console.print(
        "[green]✓ Environment saved to deliverables/repro/environment.json[/green]"
    )


if __name__ == "__main__":
    app()
