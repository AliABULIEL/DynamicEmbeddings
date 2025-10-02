"""Command-line interface for temporal LoRA training and evaluation."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from temporal_lora.utils.env import dump_environment
from temporal_lora.utils.logging import setup_logging
from temporal_lora.utils.paths import get_deliverables_dir, get_project_root

app = typer.Typer(
    name="temporal-lora",
    help="Temporal LoRA: Dynamic Sentence Embeddings via Time-Bucket Adapters",
    add_completion=False,
)

console = Console()


@app.command()
def env_dump(
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for environment dump",
    ),
) -> None:
    """Dump environment information for reproducibility."""
    setup_logging()
    
    if output_dir is None:
        output_dir = get_deliverables_dir() / "repro"
    
    console.print("[bold blue]Dumping environment information...[/bold blue]")
    env_file = dump_environment(output_dir)
    console.print(f"[green]✓[/green] Environment dump saved to: {env_file}")


@app.command()
def prepare_data(
    max_per_bucket: int = typer.Option(
        6000,
        "--max-per-bucket",
        help="Maximum samples per time bucket",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to data config YAML",
    ),
) -> None:
    """Prepare and process dataset for training."""
    setup_logging()
    console.print("[bold blue]Preparing dataset...[/bold blue]")
    console.print(f"  Max per bucket: {max_per_bucket}")
    console.print("[yellow]⚠[/yellow]  Implementation pending (Prompt 2)")


@app.command()
def train_adapters(
    epochs: int = typer.Option(2, "--epochs", "-e", help="Number of training epochs"),
    lora_r: int = typer.Option(16, "--lora-r", "-r", help="LoRA rank"),
    cross_period_negatives: bool = typer.Option(
        True,
        "--cross-period-negatives/--no-cross-period-negatives",
        help="Use cross-period-biased negative sampling",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to training config YAML",
    ),
) -> None:
    """Train temporal LoRA adapters."""
    setup_logging()
    console.print("[bold blue]Training temporal LoRA adapters...[/bold blue]")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  LoRA rank: {lora_r}")
    console.print(f"  Cross-period negatives: {cross_period_negatives}")
    console.print("[yellow]⚠[/yellow]  Implementation pending (Prompt 3)")


@app.command()
def build_indexes(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to eval config YAML",
    ),
) -> None:
    """Build FAISS indexes for all time buckets."""
    setup_logging()
    console.print("[bold blue]Building FAISS indexes...[/bold blue]")
    console.print("[yellow]⚠[/yellow]  Implementation pending (Prompt 4)")


@app.command()
def evaluate(
    scenarios: list[str] = typer.Option(
        ["within", "cross", "all"],
        "--scenarios",
        "-s",
        help="Evaluation scenarios (within/cross/all)",
    ),
    mode: str = typer.Option(
        "multi-index",
        "--mode",
        "-m",
        help="Retrieval mode (single-index/multi-index)",
    ),
    merge: str = typer.Option(
        "softmax",
        "--merge",
        help="Merge strategy (softmax/mean/max/rrf)",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to eval config YAML",
    ),
) -> None:
    """Evaluate temporal LoRA retrieval performance."""
    setup_logging()
    console.print("[bold blue]Evaluating retrieval performance...[/bold blue]")
    console.print(f"  Scenarios: {', '.join(scenarios)}")
    console.print(f"  Mode: {mode}")
    console.print(f"  Merge strategy: {merge}")
    console.print("[yellow]⚠[/yellow]  Implementation pending (Prompt 5)")


@app.command()
def visualize(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to visualization config",
    ),
) -> None:
    """Generate all visualizations (heatmaps, UMAP, drift plots)."""
    setup_logging()
    console.print("[bold blue]Generating visualizations...[/bold blue]")
    console.print("[yellow]⚠[/yellow]  Implementation pending (Prompt 6)")


@app.command()
def ablate(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to ablation config",
    ),
) -> None:
    """Run ablation studies (rank, buckets, negative strategy)."""
    setup_logging()
    console.print("[bold blue]Running ablation studies...[/bold blue]")
    console.print("[yellow]⚠[/yellow]  Implementation pending (Prompt 7)")


@app.command()
def export_deliverables(
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for deliverables",
    ),
) -> None:
    """Export all deliverables (results, figures, repro info)."""
    setup_logging()
    
    if output_dir is None:
        output_dir = get_deliverables_dir()
    
    console.print("[bold blue]Exporting deliverables...[/bold blue]")
    console.print(f"  Output directory: {output_dir}")
    console.print("[yellow]⚠[/yellow]  Implementation pending (Prompt 8)")


@app.command()
def info() -> None:
    """Show project information and structure."""
    console.print("\n[bold cyan]Temporal LoRA: Dynamic Sentence Embeddings[/bold cyan]\n")
    
    # Project structure table
    table = Table(title="Project Structure", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Description", style="white")
    
    table.add_row("Base Model", "sentence-transformers/all-MiniLM-L6-v2 (frozen)")
    table.add_row("Adapters", "LoRA on Q/K/V attention (rank 16 default)")
    table.add_row("Time Buckets", "≤2018, 2019-2024 (configurable)")
    table.add_row("Retrieval", "Multi-index FAISS with merge strategies")
    table.add_row("Metrics", "NDCG@10, Recall@10/100, MRR + 95% CI")
    
    console.print(table)
    
    # Expected gains table
    gains_table = Table(
        title="\nExpected Performance Gains (NDCG@10)",
        show_header=True,
        header_style="bold green",
    )
    gains_table.add_column("Scenario", style="cyan")
    gains_table.add_column("Expected Gain", style="green")
    
    gains_table.add_row("Within-period", "+3 to +8 points")
    gains_table.add_row("Cross-period", "+10 to +20 points")
    gains_table.add_row("Hard cross-period", "+25 to +30 points")
    gains_table.add_row("All-period", "+6 to +12 points")
    
    console.print(gains_table)
    console.print()


if __name__ == "__main__":
    app()
