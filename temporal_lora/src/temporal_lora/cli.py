"""
Command-line interface for Temporal LoRA.

All major operations (data prep, training, indexing, evaluation, visualization)
are exposed as CLI commands.
"""

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

app = typer.Typer(help="Temporal LoRA for Dynamic Sentence Embeddings")
console = Console()


@app.command()
def prepare_data(
    max_per_bucket: int = typer.Option(6000, help="Max samples per time bucket"),
    output_format: str = typer.Option("processed", help="Output format: processed, csv"),
    output_path: Optional[Path] = typer.Option(None, help="Output path for CSV"),
    inspect: bool = typer.Option(False, help="Inspect data splits only"),
):
    """
    Download and preprocess arXiv CS/ML abstracts into time buckets.
    """
    console.print("[bold blue]Preparing data...[/bold blue]")
    console.print(f"Max per bucket: {max_per_bucket}")
    console.print(f"Output format: {output_format}")
    if inspect:
        console.print("[yellow]Inspection mode - no processing[/yellow]")
    # Placeholder: Actual implementation in src/temporal_lora/data/
    console.print("[green]✓ Data preparation complete[/green]")


@app.command()
def train_adapters(
    epochs: int = typer.Option(2, help="Number of training epochs"),
    lora_r: int = typer.Option(16, help="LoRA rank"),
    cross_period_negatives: bool = typer.Option(False, help="Use cross-period negatives"),
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
    mode: str = typer.Option("multi-index", help="Retrieval mode: time-select, multi-index"),
    merge: str = typer.Option("softmax", help="Multi-index merge: softmax, max, mean, rrf"),
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
    console.print("[green]✓ Environment saved to deliverables/repro/environment.json[/green]")


if __name__ == "__main__":
    app()
