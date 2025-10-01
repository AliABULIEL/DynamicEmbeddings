"""Command-line interface for plot generation."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ..plots.figures import PlotGenerator, PlotConfig
from ..utils.config import setup_logging

logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup argument parser for plotting CLI.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="tide-lite-plots",
        description="Generate plots from TIDE-Lite experimental results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots from summary JSON
  tide-lite-plots --input results/summary.json --output-dir results/figures/
  
  # Generate only specific plot types
  tide-lite-plots --input summary.json --plot-types model_comparison ablation
  
  # High-quality publication figures
  tide-lite-plots --input summary.json --publication --format pdf --dpi 300
  
  # Custom figure size and style
  tide-lite-plots --input summary.json --size 10 8 --style seaborn-v0_8-whitegrid
  
  # Dry run to see what would be generated
  tide-lite-plots --input summary.json --dry-run
""",
    )
    
    # Input/Output
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input summary JSON file",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for plots",
    )
    
    # Plot selection
    parser.add_argument(
        "--plot-types",
        nargs="+",
        choices=[
            "model_comparison",
            "lambda_sweep",
            "mlp_size",
            "time_encoding",
            "ablation",
            "training_curves",
            "all",
        ],
        default=["all"],
        help="Types of plots to generate",
    )
    
    # Plot configuration
    parser.add_argument(
        "--size",
        nargs=2,
        type=float,
        default=[8, 6],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches",
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for saved figures",
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format for figures",
    )
    
    parser.add_argument(
        "--style",
        type=str,
        default="seaborn-v0_8-darkgrid",
        help="Matplotlib style",
    )
    
    parser.add_argument(
        "--palette",
        type=str,
        default="husl",
        help="Color palette name",
    )
    
    parser.add_argument(
        "--font-size",
        type=int,
        default=10,
        help="Base font size",
    )
    
    # Quality presets
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Use publication-quality settings",
    )
    
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="Use presentation settings (larger fonts)",
    )
    
    # Options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without creating plots",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity",
    )
    
    return parser


def load_and_prepare_data(input_path: Path) -> Dict:
    """Load and prepare data for plotting.
    
    Args:
        input_path: Path to summary JSON.
        
    Returns:
        Loaded data dictionary.
    """
    with open(input_path, "r") as f:
        data = json.load(f)
    
    return data


def extract_ablation_data(data: Dict) -> Optional[Dict]:
    """Extract ablation study data from summary.
    
    Args:
        data: Summary data dictionary.
        
    Returns:
        Ablation data or None if not found.
    """
    ablation_data = {}
    
    if "models" in data:
        for model_name, tasks in data["models"].items():
            if "ablation" in model_name.lower():
                # Try to parse configuration from name
                import re
                
                mlp_match = re.search(r"mlp(\d+)", model_name.lower())
                tw_match = re.search(r"tw([\d.]+)", model_name.lower())
                td_match = re.search(r"td(\d+)", model_name.lower())
                
                if mlp_match and tw_match:
                    mlp_dim = int(mlp_match.group(1))
                    temp_weight = float(tw_match.group(1))
                    
                    if "STS-B" in tasks and "spearman" in tasks["STS-B"]:
                        key = (mlp_dim, temp_weight)
                        ablation_data[key] = tasks["STS-B"]["spearman"]
    
    return ablation_data if ablation_data else None


def print_plot_plan(
    input_path: Path,
    output_dir: Path,
    plot_types: List[str],
    config: PlotConfig,
) -> None:
    """Print plotting plan for dry run.
    
    Args:
        input_path: Input summary file.
        output_dir: Output directory.
        plot_types: Types of plots to generate.
        config: Plot configuration.
    """
    print("\n" + "=" * 70)
    print("PLOT GENERATION PLAN")
    print("=" * 70)
    
    print(f"\n📊 Input file: {input_path}")
    print(f"📁 Output directory: {output_dir}")
    
    print("\n🎨 Plot configuration:")
    print(f"  • Size: {config.figure_size[0]}x{config.figure_size[1]} inches")
    print(f"  • DPI: {config.dpi}")
    print(f"  • Format: {config.save_format}")
    print(f"  • Style: {config.style}")
    print(f"  • Palette: {config.color_palette}")
    
    print("\n📈 Plots to generate:")
    
    if "all" in plot_types or "model_comparison" in plot_types:
        print("  • Model Comparison")
        print("    - Grouped bar chart comparing models")
        print("    - Metrics: Spearman, Pearson, MSE, etc.")
    
    if "all" in plot_types or "lambda_sweep" in plot_types:
        print("  • Lambda Sweep Analysis")
        print("    - Temporal weight vs performance")
        print("    - Multiple subplots for different metrics")
    
    if "all" in plot_types or "mlp_size" in plot_types:
        print("  • MLP Size Impact")
        print("    - Hidden dimension vs performance")
        print("    - Line plot with multiple metrics")
    
    if "all" in plot_types or "time_encoding" in plot_types:
        print("  • Time Encoding Comparison")
        print("    - Encoding dimension analysis")
        print("    - Bar chart and trend analysis")
    
    if "all" in plot_types or "ablation" in plot_types:
        print("  • Ablation Heatmap")
        print("    - 2D heatmap of parameter combinations")
        print("    - Color-coded performance metrics")
    
    if "all" in plot_types or "training_curves" in plot_types:
        print("  • Training Curves")
        print("    - Loss and metric progression")
        print("    - Train vs validation curves")
    
    print("\n[DRY RUN] No plots will be generated")
    print("=" * 70)


def main() -> int:
    """Main entry point for plotting CLI.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Apply quality presets
    if args.publication:
        args.dpi = 300
        args.format = "pdf"
        args.style = "seaborn-v0_8-whitegrid"
        args.font_size = 10
        logger.info("Using publication quality settings")
    elif args.presentation:
        args.size = [12, 8]
        args.font_size = 14
        args.dpi = 150
        logger.info("Using presentation settings")
    
    # Create plot configuration
    config = PlotConfig(
        figure_size=tuple(args.size),
        dpi=args.dpi,
        style=args.style,
        color_palette=args.palette,
        font_size=args.font_size,
        save_format=args.format,
    )
    
    if args.dry_run:
        print_plot_plan(args.input, args.output_dir, args.plot_types, config)
        
        # Load data to show what's available
        try:
            data = load_and_prepare_data(args.input)
            
            print("\n📊 Available data:")
            print(f"  • Models: {len(data.get('models', {}))}")
            
            if "models" in data:
                for model in list(data["models"].keys())[:5]:
                    tasks = list(data["models"][model].keys())
                    print(f"    - {model}: {', '.join(tasks)}")
                
                if len(data["models"]) > 5:
                    print(f"    ... and {len(data['models']) - 5} more models")
            
            if "best_performers" in data:
                print(f"  • Best performers: {len(data['best_performers'])} metrics")
            
            if "comparisons" in data:
                print(f"  • Comparisons: {len(data['comparisons'])} pairs")
            
        except Exception as e:
            print(f"\n⚠️  Could not load input file: {e}")
        
        return 0
    
    try:
        logger.info(f"Starting plot generation from {args.input}")
        
        # Create plot generator
        generator = PlotGenerator(config, args.output_dir)
        
        # Load data
        data = load_and_prepare_data(args.input)
        
        generated_plots = []
        
        # Generate requested plots
        if "all" in args.plot_types:
            figures = generator.generate_all_plots(args.input)
            generated_plots.extend(figures.keys())
        else:
            # Generate specific plot types
            if "model_comparison" in args.plot_types:
                if "models" in data:
                    models_data = {}
                    for model_name, tasks in data["models"].items():
                        if "STS-B" in tasks:
                            models_data[model_name] = tasks["STS-B"]
                    
                    if models_data:
                        fig = generator.plot_model_comparison(
                            models_data,
                            ["spearman", "pearson", "mse"],
                        )
                        generated_plots.append("model_comparison")
            
            if "ablation" in args.plot_types:
                ablation_data = extract_ablation_data(data)
                if ablation_data:
                    fig = generator.plot_ablation_heatmap(ablation_data)
                    generated_plots.append("ablation_heatmap")
            
            # Add other specific plot types as needed...
        
        # Print summary
        print("\n" + "=" * 70)
        print("PLOT GENERATION COMPLETE")
        print("=" * 70)
        print(f"\n✅ Generated {len(generated_plots)} plots:")
        for plot_name in generated_plots:
            print(f"  • {plot_name}.{args.format}")
        
        print(f"\n📁 Saved to: {args.output_dir}")
        
        return 0
        
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        logger.exception("Plot generation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
