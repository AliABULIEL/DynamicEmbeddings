#!/usr/bin/env python3
"""Export deliverables: collect results, figures, and repro info.

This script consolidates:
- CSV result files (baseline and LoRA)
- PNG visualizations
- Environment/reproducibility info
- A summary README with top-line numbers
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def get_top_metrics(results_path: Path) -> Optional[Dict[str, float]]:
    """Extract top-line metrics from results CSV.
    
    Args:
        results_path: Path to results CSV file.
        
    Returns:
        Dictionary with metric names and values, or None if file doesn't exist.
    """
    if not results_path.exists():
        return None
    
    df = pd.read_csv(results_path, index_col=0)
    
    # Get mean across all scenarios for key metrics
    metrics = {}
    for metric in ["ndcg@10", "recall@10", "recall@100", "mrr"]:
        if metric in df.columns:
            metrics[metric] = float(df[metric].mean())
    
    return metrics


def create_results_readme(
    deliverables_dir: Path,
    baseline_metrics: Optional[Dict[str, float]],
    lora_metrics: Optional[Dict[str, float]],
) -> None:
    """Create README with top-line numbers.
    
    Args:
        deliverables_dir: Deliverables directory path.
        baseline_metrics: Baseline metrics dictionary.
        lora_metrics: LoRA metrics dictionary.
    """
    readme_path = deliverables_dir / "README_results.md"
    
    with open(readme_path, "w") as f:
        f.write("# Temporal LoRA for Dynamic Sentence Embeddings - Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Top-line summary
        f.write("## Top-Line Results\n\n")
        
        if baseline_metrics and lora_metrics:
            f.write("| Metric | Baseline | LoRA | Δ Improvement | % Improvement |\n")
            f.write("|--------|----------|------|---------------|---------------|\n")
            
            for metric in ["ndcg@10", "recall@10", "recall@100", "mrr"]:
                if metric in baseline_metrics and metric in lora_metrics:
                    baseline = baseline_metrics[metric]
                    lora = lora_metrics[metric]
                    delta = lora - baseline
                    pct = (delta / baseline * 100) if baseline > 0 else 0
                    
                    f.write(
                        f"| {metric.upper()} | {baseline:.4f} | {lora:.4f} | "
                        f"{delta:+.4f} | {pct:+.2f}% |\n"
                    )
        elif lora_metrics:
            f.write("**LoRA Results (mean across scenarios):**\n\n")
            for metric, value in lora_metrics.items():
                f.write(f"- **{metric.upper()}**: {value:.4f}\n")
        elif baseline_metrics:
            f.write("**Baseline Results (mean across scenarios):**\n\n")
            for metric, value in baseline_metrics.items():
                f.write(f"- **{metric.upper()}**: {value:.4f}\n")
        else:
            f.write("*No results found. Run evaluation first.*\n")
        
        f.write("\n---\n\n")
        
        # File inventory
        f.write("## Files in This Directory\n\n")
        
        f.write("### Results\n")
        f.write("- `results/baseline_results.csv` - Baseline retrieval metrics\n")
        f.write("- `results/lora_results.csv` - LoRA retrieval metrics\n\n")
        
        f.write("### Visualizations\n")
        f.write("- `figures/comparison_heatmaps_*.png` - Performance comparison heatmaps\n")
        f.write("- `figures/umap_embeddings.png` - UMAP projection of embeddings\n\n")
        
        f.write("### Reproducibility\n")
        f.write("- `repro/environment.json` - System info, CUDA, packages, git SHA\n\n")
        
        f.write("---\n\n")
        
        # Methodology
        f.write("## Methodology\n\n")
        f.write("### Model Architecture\n")
        f.write("- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2` (frozen)\n")
        f.write("- **Adaptation**: Time-bucket LoRA on attention Q/K/V layers\n")
        f.write("- **Time Buckets**: ≤2018, 2019–2024\n\n")
        
        f.write("### Evaluation\n")
        f.write("- **Metrics**: NDCG@10, Recall@10/100, MRR\n")
        f.write("- **Modes**: Time-select and multi-index retrieval\n")
        f.write("- **Scenarios**: Within-period, cross-period, and all queries\n\n")
        
        f.write("### Training\n")
        f.write("- **Epochs**: 2-3 with early stopping\n")
        f.write("- **Precision**: FP16 (when CUDA available)\n")
        f.write("- **Negatives**: Cross-period-biased sampling\n\n")
        
        f.write("---\n\n")
        f.write("For more details, see project documentation.\n")
    
    print(f"✓ Created {readme_path}")


def export_deliverables(
    project_root: Path,
    deliverables_dir: Path,
) -> None:
    """Export all deliverables to consolidated directory.
    
    Args:
        project_root: Project root path.
        deliverables_dir: Output deliverables directory.
    """
    print("=" * 60)
    print("EXPORTING DELIVERABLES")
    print("=" * 60)
    
    # Create deliverables structure
    results_dir = deliverables_dir / "results"
    figures_dir = deliverables_dir / "figures"
    repro_dir = deliverables_dir / "repro"
    
    for d in [results_dir, figures_dir, repro_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Copy results CSVs
    print("\n📊 Copying results...")
    source_results = project_root / "deliverables" / "results"
    if source_results.exists():
        for csv_file in source_results.glob("*.csv"):
            dest = results_dir / csv_file.name
            shutil.copy2(csv_file, dest)
            print(f"  ✓ {csv_file.name}")
    else:
        print("  ⚠ No results found (run evaluation first)")
    
    # Copy figures
    print("\n🎨 Copying figures...")
    source_figures = project_root / "deliverables" / "figures"
    if source_figures.exists():
        for fig_file in source_figures.glob("*.png"):
            dest = figures_dir / fig_file.name
            shutil.copy2(fig_file, dest)
            print(f"  ✓ {fig_file.name}")
    else:
        print("  ⚠ No figures found (run visualize first)")
    
    # Copy repro info
    print("\n🔄 Copying reproducibility info...")
    source_repro = project_root / "deliverables" / "repro"
    if source_repro.exists():
        for repro_file in source_repro.glob("*"):
            if repro_file.is_file():
                dest = repro_dir / repro_file.name
                shutil.copy2(repro_file, dest)
                print(f"  ✓ {repro_file.name}")
    else:
        print("  ⚠ No repro info found (run env-dump first)")
    
    # Load metrics for README
    baseline_metrics = get_top_metrics(results_dir / "baseline_results.csv")
    lora_metrics = get_top_metrics(results_dir / "lora_results.csv")
    
    # Create README with top-line numbers
    print("\n📝 Creating summary README...")
    create_results_readme(deliverables_dir, baseline_metrics, lora_metrics)
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nAll deliverables exported to: {deliverables_dir}")
    print("\nContents:")
    print(f"  - results/     ({len(list(results_dir.glob('*.csv')))} CSV files)")
    print(f"  - figures/     ({len(list(figures_dir.glob('*.png')))} PNG files)")
    print(f"  - repro/       ({len(list(repro_dir.glob('*')))} files)")
    print(f"  - README_results.md")


def main():
    """Main entry point."""
    import sys
    
    # Get project root (assumes script is in scripts/)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    deliverables_dir = project_root / "deliverables"
    
    print(f"Project root: {project_root}")
    print(f"Deliverables: {deliverables_dir}\n")
    
    export_deliverables(project_root, deliverables_dir)


if __name__ == "__main__":
    main()
