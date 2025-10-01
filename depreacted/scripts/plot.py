#!/usr/bin/env python3
"""Generate plots from experimental results."""

import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tide_lite.plots.figures import PlotGenerator, PlotConfig


def main():
    parser = argparse.ArgumentParser(description="Generate plots from TIDE-Lite results")
    parser.add_argument(
        "--input",
        type=str,
        default="results/summary.json",
        help="Path to summary JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format for plots"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for saved figures"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be plotted without generating"
    )
    
    args = parser.parse_args()
    
    # Setup plot configuration
    config = PlotConfig(
        save_format=args.format,
        dpi=args.dpi,
        figure_size=(8, 6),
    )
    
    # Create generator
    generator = PlotGenerator(config, args.output_dir)
    
    # Generate mock data if input doesn't exist
    if not Path(args.input).exists() or args.dry_run:
        print(f"\n{'='*70}")
        print("GENERATING MOCK PLOTS (no real data)")
        print(f"{'='*70}\n")
        
        # Create mock data for demonstration
        mock_data = {
            "models": {
                "TIDE-Lite": {
                    "STS-B": {
                        "spearman": 0.785,
                        "pearson": 0.792,
                        "mse": 0.423,
                        "latency_ms": 8.5,
                        "params": 53200
                    }
                },
                "Baseline (MiniLM)": {
                    "STS-B": {
                        "spearman": 0.771,
                        "pearson": 0.778,
                        "mse": 0.451,
                        "latency_ms": 6.2,
                        "params": 0
                    }
                }
            },
            "ablations": {
                "temporal_weight": {
                    0.0: {"spearman": 0.771, "temporal_consistency": 0.45},
                    0.05: {"spearman": 0.778, "temporal_consistency": 0.62},
                    0.1: {"spearman": 0.785, "temporal_consistency": 0.71},
                    0.2: {"spearman": 0.782, "temporal_consistency": 0.78},
                    0.5: {"spearman": 0.768, "temporal_consistency": 0.85}
                },
                "mlp_hidden": {
                    64: {"spearman": 0.776, "params": 28800},
                    128: {"spearman": 0.785, "params": 53200},
                    256: {"spearman": 0.789, "params": 102000}
                }
            }
        }
        
        # Save mock data
        mock_path = Path(args.output_dir) / "mock_summary.json"
        mock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mock_path, 'w') as f:
            json.dump(mock_data, f, indent=2)
        
        # Generate plots from mock data
        if not args.dry_run:
            # Score vs dimension plot
            fig = generator.plot_mlp_size_vs_metrics(
                mock_data["ablations"]["mlp_hidden"],
                ["spearman"],
                title="Performance vs MLP Hidden Dimension",
                save_name="fig_score_vs_dim"
            )
            
            # Latency comparison plot
            latency_data = {
                name: {"latency_ms": model_data["STS-B"]["latency_ms"]}
                for name, model_data in mock_data["models"].items()
            }
            fig = generator.plot_model_comparison(
                mock_data["models"],
                ["latency_ms", "params"],
                title="Model Efficiency Comparison",
                save_name="fig_latency_vs_dim"
            )
            
            # Temporal consistency plot
            fig = generator.plot_lambda_vs_metrics(
                mock_data["ablations"]["temporal_weight"],
                ["spearman", "temporal_consistency"],
                title="Impact of Temporal Weight λ",
                save_name="fig_temporal_ablation"
            )
            
            print(f"\nGenerated plots in: {args.output_dir}")
            print(f"  - fig_score_vs_dim.{args.format}")
            print(f"  - fig_latency_vs_dim.{args.format}")
            print(f"  - fig_temporal_ablation.{args.format}")
    else:
        # Generate from real data
        figures = generator.generate_all_plots(args.input, args.dry_run)
        
        if not args.dry_run:
            print(f"\nGenerated {len(figures)} plots in: {args.output_dir}")
    
    print("\n✓ Plot generation complete!")


if __name__ == "__main__":
    main()
