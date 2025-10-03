#!/usr/bin/env python3
"""Export results, figures, and reproducibility info to deliverables/."""

import json
import shutil
from pathlib import Path

from temporal_lora.utils.logging import get_logger
from temporal_lora.utils.env import dump_environment

logger = get_logger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DELIVERABLES_DIR = PROJECT_ROOT / "deliverables"
RESULTS_DIR = DELIVERABLES_DIR / "results"
FIGURES_DIR = DELIVERABLES_DIR / "figures"
REPRO_DIR = DELIVERABLES_DIR / "repro"


def create_deliverables_structure():
    """Create deliverables directory structure."""
    logger.info("Creating deliverables structure...")
    
    DELIVERABLES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    REPRO_DIR.mkdir(exist_ok=True)
    
    logger.info(f"✓ Structure created at: {DELIVERABLES_DIR}")


def collect_results():
    """Collect result CSVs from various locations."""
    logger.info("Collecting results...")
    
    # Paths to check
    source_dirs = [
        PROJECT_ROOT / "deliverables" / "results",
        PROJECT_ROOT / "results",
    ]
    
    # Patterns to collect
    patterns = [
        "**/baseline_frozen_*.csv",
        "**/lora_*.csv",
        "**/full_ft_*.csv",
        "**/seq_ft_*.csv",
        "**/delta_*.csv",
        "**/efficiency_summary.csv",
        "**/temperature_sweep.csv",
    ]
    
    collected = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        
        for pattern in patterns:
            for csv_file in source_dir.glob(pattern):
                if "deliverables/results" not in str(csv_file):
                    # Copy to deliverables
                    dest = RESULTS_DIR / csv_file.name
                    shutil.copy(csv_file, dest)
                    collected.append(dest.name)
                    logger.info(f"  ✓ Collected: {csv_file.name}")
    
    if not collected:
        logger.warning("No result files found to collect")
    
    return collected


def collect_figures():
    """Collect visualizations."""
    logger.info("Collecting figures...")
    
    # Paths to check
    source_dirs = [
        PROJECT_ROOT / "deliverables" / "figures",
        PROJECT_ROOT / "figures",
        PROJECT_ROOT / "plots",
    ]
    
    # Patterns to collect
    patterns = [
        "**/*.png",
        "**/*.pdf",
        "**/*.svg",
    ]
    
    collected = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        
        for pattern in patterns:
            for fig_file in source_dir.glob(pattern):
                if "deliverables/figures" not in str(fig_file):
                    # Copy to deliverables
                    dest = FIGURES_DIR / fig_file.name
                    shutil.copy(fig_file, dest)
                    collected.append(dest.name)
                    logger.info(f"  ✓ Collected: {fig_file.name}")
    
    if not collected:
        logger.warning("No figure files found to collect")
    
    return collected


def create_summary_readme(result_files, figure_files):
    """Create README summarizing deliverables."""
    logger.info("Creating summary README...")
    
    readme_content = """# Temporal LoRA Evaluation Results

This directory contains the complete evaluation results, visualizations, and reproducibility information for the Temporal LoRA project.

## Directory Structure

```
deliverables/
├── results/          # Evaluation metrics and comparisons
├── figures/          # Visualizations (heatmaps, UMAP plots)
├── repro/            # Reproducibility information
└── README_results.md # This file
```

## Results Files

"""
    
    # Add result files
    if result_files:
        readme_content += "### Evaluation Metrics\n\n"
        
        # Group files
        baseline_files = [f for f in result_files if "baseline" in f]
        lora_files = [f for f in result_files if "lora" in f and "baseline" not in f]
        full_ft_files = [f for f in result_files if "full_ft" in f]
        seq_ft_files = [f for f in result_files if "seq_ft" in f]
        delta_files = [f for f in result_files if "delta" in f]
        other_files = [f for f in result_files if f not in baseline_files + lora_files + full_ft_files + seq_ft_files + delta_files]
        
        if baseline_files:
            readme_content += "**Baseline (Frozen Encoder):**\n"
            for f in sorted(baseline_files):
                readme_content += f"- `{f}`\n"
            readme_content += "\n"
        
        if lora_files:
            readme_content += "**LoRA Adapters:**\n"
            for f in sorted(lora_files):
                readme_content += f"- `{f}`\n"
            readme_content += "\n"
        
        if full_ft_files:
            readme_content += "**Full Fine-Tuning:**\n"
            for f in sorted(full_ft_files):
                readme_content += f"- `{f}`\n"
            readme_content += "\n"
        
        if seq_ft_files:
            readme_content += "**Sequential Fine-Tuning:**\n"
            for f in sorted(seq_ft_files):
                readme_content += f"- `{f}`\n"
            readme_content += "\n"
        
        if delta_files:
            readme_content += "**Comparisons (LoRA - Baseline):**\n"
            for f in sorted(delta_files):
                readme_content += f"- `{f}`\n"
            readme_content += "\n"
        
        if other_files:
            readme_content += "**Other Results:**\n"
            for f in sorted(other_files):
                readme_content += f"- `{f}`\n"
            readme_content += "\n"
    
    # Add figure files
    if figure_files:
        readme_content += "## Visualizations\n\n"
        
        heatmap_files = [f for f in figure_files if "heatmap" in f.lower()]
        umap_files = [f for f in figure_files if "umap" in f.lower()]
        other_figs = [f for f in figure_files if f not in heatmap_files + umap_files]
        
        if heatmap_files:
            readme_content += "**Heatmaps (Query Bucket × Doc Bucket):**\n"
            for f in sorted(heatmap_files):
                readme_content += f"- `{f}`\n"
            readme_content += "\n"
        
        if umap_files:
            readme_content += "**UMAP Embeddings:**\n"
            for f in sorted(umap_files):
                readme_content += f"- `{f}`\n"
            readme_content += "\n"
        
        if other_figs:
            readme_content += "**Other Figures:**\n"
            for f in sorted(other_figs):
                readme_content += f"- `{f}`\n"
            readme_content += "\n"
    
    # Add metrics explanation
    readme_content += """## Metrics

All results include the following metrics:

- **NDCG@10**: Normalized Discounted Cumulative Gain at position 10
- **Recall@10**: Fraction of relevant documents in top 10
- **Recall@100**: Fraction of relevant documents in top 100
- **MRR**: Mean Reciprocal Rank

## Evaluation Modes

Results are organized by training mode:

1. **baseline_frozen**: Static frozen encoder (no time adaptation)
2. **lora**: LoRA adapters per time bucket (main contribution)
3. **full_ft**: Full fine-tuning per bucket (upper bound)
4. **seq_ft**: Sequential fine-tuning (catastrophic forgetting demo)

## Matrix Structure

Each CSV file with `_ndcg_at_10.csv`, `_recall_at_10.csv`, etc. contains a query bucket × doc bucket matrix:

- **Rows**: Query time buckets
- **Columns**: Document time buckets
- **Diagonal**: Within-period retrieval
- **Off-diagonal**: Cross-period retrieval

## Key Findings

Look for:

1. **Delta matrices** show LoRA improvements over baseline
2. **Temperature sweep** results identify optimal merge settings
3. **Efficiency summary** compares parameter counts and training times
4. **Diagonal vs off-diagonal** patterns reveal temporal adaptation

## Reproducibility

See `repro/` directory for:
- Environment snapshot (packages, CUDA)
- Git commit SHA
- Configuration files

---

*Generated by export_results.py*
"""
    
    readme_path = DELIVERABLES_DIR / "README_results.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    logger.info(f"✓ README created: {readme_path}")


def export_environment():
    """Export reproducibility information."""
    logger.info("Exporting environment info...")
    
    dump_environment(REPRO_DIR, repo_path=PROJECT_ROOT)
    
    logger.info(f"✓ Environment exported to: {REPRO_DIR}")


def main():
    """Run export process."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPORTING DELIVERABLES")
    logger.info("=" * 80 + "\n")
    
    # Create structure
    create_deliverables_structure()
    
    # Collect files
    result_files = collect_results()
    figure_files = collect_figures()
    
    # Create summary
    create_summary_readme(result_files, figure_files)
    
    # Export environment
    export_environment()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("EXPORT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Results collected: {len(result_files)}")
    logger.info(f"Figures collected: {len(figure_files)}")
    logger.info(f"Output directory: {DELIVERABLES_DIR}")
    logger.info("=" * 80)
    
    logger.info("\n✓ Export complete!")


if __name__ == "__main__":
    main()
