"""Command-line interface for Temporal LoRA pipeline."""

from typing import List, Optional

import typer

app = typer.Typer(
    name="temporal-lora",
    help="Temporal LoRA for Dynamic Sentence Embeddings",
    add_completion=False,
)


@app.command()
def env_dump() -> None:
    """Dump environment info (CUDA, packages, git SHA) to deliverables/repro/."""
    raise NotImplementedError("env-dump not yet implemented")


@app.command()
def prepare_data(
    max_per_bucket: int = typer.Option(6000, help="Max samples per time bucket"),
    output_dir: Optional[str] = typer.Option(None, help="Override output directory"),
) -> None:
    """Download and preprocess arXiv CS/ML dataset into time buckets."""
    raise NotImplementedError("prepare-data not yet implemented")


@app.command()
def train_adapters(
    epochs: int = typer.Option(2, help="Training epochs"),
    lora_r: int = typer.Option(16, help="LoRA rank"),
    cross_period_negatives: bool = typer.Option(True, help="Use cross-period negatives"),
    config: Optional[str] = typer.Option(None, help="Override config file path"),
) -> None:
    """Train time-bucket LoRA adapters on frozen sentence encoder."""
    raise NotImplementedError("train-adapters not yet implemented")


@app.command()
def build_indexes(
    checkpoint_dir: Optional[str] = typer.Option(None, help="Path to adapter checkpoints"),
) -> None:
    """Build FAISS IndexFlatIP for each time bucket."""
    raise NotImplementedError("build-indexes not yet implemented")


@app.command()
def evaluate(
    scenarios: List[str] = typer.Option(
        ["within", "cross", "all"], help="Eval scenarios: within/cross/all"
    ),
    mode: str = typer.Option("multi-index", help="Retrieval mode: time-select or multi-index"),
    merge: str = typer.Option("softmax", help="Multi-index merge: softmax/mean/max/rrf"),
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
