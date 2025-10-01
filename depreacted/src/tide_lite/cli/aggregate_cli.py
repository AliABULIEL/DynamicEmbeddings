"""Command-line interface for result aggregation."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from ..plots.aggregate import ResultAggregator
from ..utils.config import setup_logging

logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup argument parser for aggregation CLI.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="tide-lite-aggregate",
        description="Aggregate TIDE-Lite experimental results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate all results in a directory
  tide-lite-aggregate --results-dir results/
  
  # Save to specific output directory
  tide-lite-aggregate --results-dir results/ --output-dir results/summary/
  
  # Include only specific patterns
  tide-lite-aggregate --results-dir results/ --include "**/metrics_stsb_*.json"
  
  # Exclude backup files
  tide-lite-aggregate --results-dir results/ --exclude "**/*backup*"
  
  # Generate only CSV output
  tide-lite-aggregate --results-dir results/ --csv-only
  
  # Dry run to see what would be processed
  tide-lite-aggregate --results-dir results/ --dry-run
""",
    )
    
    # Input/Output
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing results to aggregate",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to results-dir)",
    )
    
    # File patterns
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="Glob patterns for files to include",
    )
    
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help="Glob patterns for files to exclude",
    )
    
    # Output formats
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Generate only CSV output",
    )
    
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Generate only JSON output",
    )
    
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip markdown summary generation",
    )
    
    # Output filenames
    parser.add_argument(
        "--csv-name",
        type=str,
        default="summary.csv",
        help="Name for CSV output file",
    )
    
    parser.add_argument(
        "--json-name",
        type=str,
        default="summary.json",
        help="Name for JSON output file",
    )
    
    parser.add_argument(
        "--markdown-name",
        type=str,
        default="summary.md",
        help="Name for markdown output file",
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
        help="Show what would be processed without aggregating",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity",
    )
    
    return parser


def print_aggregation_plan(
    results_dir: Path,
    output_dir: Path,
    include_patterns: Optional[List[str]],
    exclude_patterns: Optional[List[str]],
    generate_csv: bool,
    generate_json: bool,
    generate_markdown: bool,
) -> None:
    """Print aggregation plan for dry run.
    
    Args:
        results_dir: Results directory.
        output_dir: Output directory.
        include_patterns: Include patterns.
        exclude_patterns: Exclude patterns.
        generate_csv: Whether CSV will be generated.
        generate_json: Whether JSON will be generated.
        generate_markdown: Whether markdown will be generated.
    """
    print("\n" + "=" * 70)
    print("AGGREGATION PLAN")
    print("=" * 70)
    
    print(f"\n📂 Input directory: {results_dir}")
    print(f"📊 Output directory: {output_dir}")
    
    print("\n🔍 File patterns:")
    if include_patterns:
        print("  Include:")
        for pattern in include_patterns:
            print(f"    • {pattern}")
    else:
        print("  Include: **/metrics_*.json (default)")
    
    if exclude_patterns:
        print("  Exclude:")
        for pattern in exclude_patterns:
            print(f"    • {pattern}")
    
    print("\n📄 Output formats:")
    if generate_csv:
        print("  • CSV summary")
    if generate_json:
        print("  • JSON summary")
    if generate_markdown:
        print("  • Markdown report")
    
    print("\n📈 Expected processing:")
    print("  1. Scan for metrics files matching patterns")
    print("  2. Parse model names and task types")
    print("  3. Extract numeric metrics")
    print("  4. Identify best performers")
    print("  5. Calculate improvements over baseline")
    print("  6. Generate output files")
    
    print("\n[DRY RUN] No actual aggregation will occur")
    print("=" * 70)


def main() -> int:
    """Main entry point for aggregation CLI.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Determine output directory
    output_dir = args.output_dir or args.results_dir
    output_dir = Path(output_dir)
    
    # Determine what to generate
    if args.csv_only:
        generate_csv = True
        generate_json = False
        generate_markdown = False
    elif args.json_only:
        generate_csv = False
        generate_json = True
        generate_markdown = False
    else:
        generate_csv = True
        generate_json = True
        generate_markdown = not args.no_markdown
    
    # Set default patterns if not provided
    include_patterns = args.include or [
        "**/metrics_*.json",
        "**/metrics_train.json",
    ]
    
    exclude_patterns = args.exclude or [
        "**/*backup*",
        "**/*tmp*",
    ]
    
    if args.dry_run:
        print_aggregation_plan(
            args.results_dir,
            output_dir,
            include_patterns,
            exclude_patterns,
            generate_csv,
            generate_json,
            generate_markdown,
        )
        
        # Show example files that would be found
        print("\n📁 Example files that would be processed:")
        aggregator = ResultAggregator(
            args.results_dir,
            include_patterns,
            exclude_patterns,
        )
        files = aggregator.find_metrics_files()
        
        for i, file in enumerate(files[:10], 1):
            print(f"  {i}. {file.relative_to(args.results_dir)}")
        
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        
        print(f"\n📊 Total files to aggregate: {len(files)}")
        
        return 0
    
    try:
        logger.info(f"Starting aggregation from {args.results_dir}")
        
        # Create aggregator
        aggregator = ResultAggregator(
            args.results_dir,
            include_patterns,
            exclude_patterns,
        )
        
        # Find files
        files = aggregator.find_metrics_files()
        
        if not files:
            print(f"Warning: No metrics files found in {args.results_dir}")
            return 1
        
        print(f"Found {len(files)} metrics files to aggregate")
        
        # Perform aggregation
        aggregated = aggregator.aggregate()
        
        # Generate outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if generate_csv:
            csv_path = output_dir / args.csv_name
            aggregator.save_csv(csv_path)
            print(f"✅ Saved CSV to {csv_path}")
        
        if generate_json:
            json_path = output_dir / args.json_name
            aggregator.save_json(json_path)
            print(f"✅ Saved JSON to {json_path}")
        
        if generate_markdown:
            markdown_path = output_dir / args.markdown_name
            markdown = aggregator.generate_markdown_summary()
            with open(markdown_path, "w") as f:
                f.write(markdown)
            print(f"✅ Saved markdown to {markdown_path}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("AGGREGATION SUMMARY")
        print("=" * 70)
        print(f"\nModels found: {len(aggregated.models)}")
        for model in aggregated.models:
            tasks = list(aggregated.models[model].keys())
            print(f"  • {model}: {', '.join(tasks)}")
        
        if aggregated.best_performers:
            print("\nBest performers:")
            for metric, best in aggregated.best_performers.items():
                print(f"  • {metric}: {best}")
        
        print("\n✅ Aggregation complete!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        logger.exception("Aggregation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
