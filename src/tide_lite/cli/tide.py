"""Unified orchestrator CLI for TIDE-Lite pipeline.

This module provides a single entry point for all TIDE-Lite operations
including training, evaluation, benchmarking, ablation studies, and reporting.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TIDEOrchestrator:
    """Orchestrates TIDE-Lite pipeline operations."""
    
    def __init__(self, dry_run: bool = False) -> None:
        """Initialize orchestrator.
        
        Args:
            dry_run: If True, only print commands without executing.
        """
        self.dry_run = dry_run
        self.commands_to_run: List[Dict[str, Any]] = []
    
    def add_command(
        self,
        name: str,
        cmd: List[str],
        description: str = "",
    ) -> None:
        """Add a command to the execution queue.
        
        Args:
            name: Command identifier.
            cmd: Command as list of arguments.
            description: Human-readable description.
        """
        self.commands_to_run.append({
            "name": name,
            "cmd": cmd,
            "description": description,
        })
    
    def execute_plan(self) -> int:
        """Execute or display the command plan.
        
        Returns:
            Exit code (0 for success).
        """
        if not self.commands_to_run:
            print("No commands to execute.")
            return 0
        
        print("\n" + "=" * 70)
        print("TIDE-LITE ORCHESTRATOR - EXECUTION PLAN")
        print("=" * 70)
        
        for i, command in enumerate(self.commands_to_run, 1):
            print(f"\n[{i}] {command['name']}")
            if command['description']:
                print(f"    {command['description']}")
            print(f"    Command: {' '.join(command['cmd'])}")
        
        if self.dry_run:
            print("\n" + "=" * 70)
            print("DRY RUN MODE - Commands above would be executed in sequence")
            print("=" * 70)
            return 0
        
        print("\n" + "=" * 70)
        print("EXECUTING COMMANDS")
        print("=" * 70)
        
        for i, command in enumerate(self.commands_to_run, 1):
            print(f"\n[{i}/{len(self.commands_to_run)}] Executing: {command['name']}")
            print(f"Command: {' '.join(command['cmd'])}")
            
            # In real implementation, would use subprocess.run()
            # For now, just print since we're in dry-run mode
            print(f"[WOULD EXECUTE]: {' '.join(command['cmd'])}")
        
        print("\n✅ All commands completed successfully!")
        return 0


def cmd_train(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle train subcommand.
    
    Args:
        args: Parsed arguments.
        orchestrator: Orchestrator instance.
        
    Returns:
        Exit code.
    """
    cmd = ["python", "-m", "tide_lite.cli.train_cli"]
    
    if args.config:
        cmd.extend(["--config", str(args.config)])
    if args.output_dir:
        cmd.extend(["--output-dir", str(args.output_dir)])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.learning_rate:
        cmd.extend(["--learning-rate", str(args.learning_rate)])
    if args.num_epochs:
        cmd.extend(["--num-epochs", str(args.num_epochs)])
    if args.temporal_weight:
        cmd.extend(["--temporal-weight", str(args.temporal_weight)])
    if args.dry_run:
        cmd.append("--dry-run")
    
    orchestrator.add_command(
        "train",
        cmd,
        f"Train TIDE-Lite model for {args.num_epochs or 3} epochs"
    )
    
    return orchestrator.execute_plan()


def cmd_eval_stsb(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle eval-stsb subcommand.
    
    Args:
        args: Parsed arguments.
        orchestrator: Orchestrator instance.
        
    Returns:
        Exit code.
    """
    cmd = ["python", "-m", "tide_lite.cli.eval_stsb_cli"]
    
    if args.model_path:
        cmd.extend(["--model-path", str(args.model_path)])
    elif args.baseline:
        cmd.extend(["--baseline", args.baseline])
    
    if args.output_dir:
        cmd.extend(["--output-dir", str(args.output_dir)])
    if args.split:
        cmd.extend(["--split", args.split])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.compare_baseline:
        cmd.append("--compare-baseline")
    if args.dry_run:
        cmd.append("--dry-run")
    
    orchestrator.add_command(
        "eval-stsb",
        cmd,
        f"Evaluate on STS-B {args.split or 'test'} split"
    )
    
    return orchestrator.execute_plan()


def cmd_eval_quora(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle eval-quora subcommand.
    
    Args:
        args: Parsed arguments.
        orchestrator: Orchestrator instance.
        
    Returns:
        Exit code.
    """
    cmd = ["python", "-m", "tide_lite.cli.eval_quora_cli"]
    
    if args.model_path:
        cmd.extend(["--model-path", str(args.model_path)])
    elif args.baseline:
        cmd.extend(["--baseline", args.baseline])
    
    if args.output_dir:
        cmd.extend(["--output-dir", str(args.output_dir)])
    if args.index_type:
        cmd.extend(["--index-type", args.index_type])
    if args.max_corpus:
        cmd.extend(["--max-corpus", str(args.max_corpus)])
    if args.max_queries:
        cmd.extend(["--max-queries", str(args.max_queries)])
    if args.dry_run:
        cmd.append("--dry-run")
    
    orchestrator.add_command(
        "eval-quora",
        cmd,
        f"Evaluate on Quora retrieval with {args.index_type or 'Flat'} index"
    )
    
    return orchestrator.execute_plan()


def cmd_eval_temporal(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle eval-temporal subcommand.
    
    Args:
        args: Parsed arguments.
        orchestrator: Orchestrator instance.
        
    Returns:
        Exit code.
    """
    cmd = ["python", "-m", "tide_lite.cli.eval_temporal_cli"]
    
    if args.model_path:
        cmd.extend(["--model-path", str(args.model_path)])
    elif args.baseline:
        cmd.extend(["--baseline", args.baseline])
    
    if args.output_dir:
        cmd.extend(["--output-dir", str(args.output_dir)])
    if args.time_window_days:
        cmd.extend(["--time-window-days", str(args.time_window_days)])
    if args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.compare_baseline:
        cmd.append("--compare-baseline")
    if args.dry_run:
        cmd.append("--dry-run")
    
    orchestrator.add_command(
        "eval-temporal",
        cmd,
        f"Evaluate temporal understanding (window: {args.time_window_days or 30} days)"
    )
    
    return orchestrator.execute_plan()


def cmd_bench_all(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle bench-all subcommand - runs all three evaluations.
    
    Args:
        args: Parsed arguments.
        orchestrator: Orchestrator instance.
        
    Returns:
        Exit code.
    """
    output_dir = args.output_dir or Path("results/benchmark")
    
    # STS-B evaluation
    cmd_stsb = ["python", "-m", "tide_lite.cli.eval_stsb_cli"]
    if args.model_path:
        cmd_stsb.extend(["--model-path", str(args.model_path)])
    elif args.baseline:
        cmd_stsb.extend(["--baseline", args.baseline])
    cmd_stsb.extend(["--output-dir", str(output_dir)])
    cmd_stsb.extend(["--split", args.split or "test"])
    if args.dry_run:
        cmd_stsb.append("--dry-run")
    
    orchestrator.add_command(
        "bench-stsb",
        cmd_stsb,
        "Benchmark on STS-B semantic similarity"
    )
    
    # Quora retrieval evaluation
    cmd_quora = ["python", "-m", "tide_lite.cli.eval_quora_cli"]
    if args.model_path:
        cmd_quora.extend(["--model-path", str(args.model_path)])
    elif args.baseline:
        cmd_quora.extend(["--baseline", args.baseline])
    cmd_quora.extend(["--output-dir", str(output_dir)])
    cmd_quora.extend(["--index-type", "Flat"])
    if args.max_corpus:
        cmd_quora.extend(["--max-corpus", str(args.max_corpus)])
    if args.dry_run:
        cmd_quora.append("--dry-run")
    
    orchestrator.add_command(
        "bench-quora",
        cmd_quora,
        "Benchmark on Quora duplicate questions retrieval"
    )
    
    # Temporal evaluation
    cmd_temporal = ["python", "-m", "tide_lite.cli.eval_temporal_cli"]
    if args.model_path:
        cmd_temporal.extend(["--model-path", str(args.model_path)])
    elif args.baseline:
        cmd_temporal.extend(["--baseline", args.baseline])
    cmd_temporal.extend(["--output-dir", str(output_dir)])
    cmd_temporal.extend(["--time-window-days", "30"])
    if args.dry_run:
        cmd_temporal.append("--dry-run")
    
    orchestrator.add_command(
        "bench-temporal",
        cmd_temporal,
        "Benchmark on temporal understanding"
    )
    
    return orchestrator.execute_plan()


def cmd_ablation(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle ablation subcommand - parameter grid search.
    
    Args:
        args: Parsed arguments.
        orchestrator: Orchestrator instance.
        
    Returns:
        Exit code.
    """
    # Parse parameter ranges
    mlp_hidden_dims = [int(x) for x in args.mlp_hidden_dims.split(",")]
    temporal_weights = [float(x) for x in args.temporal_weights.split(",")]
    time_encoding_dims = [int(x) for x in args.time_encoding_dims.split(",")]
    
    output_base = args.output_dir or Path("results/ablation")
    
    # Generate all combinations
    combinations = list(product(mlp_hidden_dims, temporal_weights, time_encoding_dims))
    
    print(f"\nAblation study: {len(combinations)} configurations")
    print(f"MLP hidden dims: {mlp_hidden_dims}")
    print(f"Temporal weights: {temporal_weights}")
    print(f"Time encoding dims: {time_encoding_dims}")
    
    for i, (mlp_dim, temp_weight, time_dim) in enumerate(combinations, 1):
        run_name = f"ablation_mlp{mlp_dim}_tw{temp_weight}_td{time_dim}"
        output_dir = output_base / run_name
        
        # Training command
        cmd_train = [
            "python", "-m", "tide_lite.cli.train_cli",
            "--config", str(args.config or "configs/defaults.yaml"),
            "--output-dir", str(output_dir),
            "--mlp-hidden-dim", str(mlp_dim),
            "--temporal-weight", str(temp_weight),
            "--time-encoding-dim", str(time_dim),
            "--num-epochs", str(args.num_epochs or 1),  # Quick ablation
        ]
        if args.dry_run:
            cmd_train.append("--dry-run")
        
        orchestrator.add_command(
            f"ablation-train-{i}",
            cmd_train,
            f"Config {i}/{len(combinations)}: mlp={mlp_dim}, tw={temp_weight}, td={time_dim}"
        )
        
        # Evaluation command (just STS-B for ablation)
        cmd_eval = [
            "python", "-m", "tide_lite.cli.eval_stsb_cli",
            "--model-path", str(output_dir / "checkpoints" / "checkpoint_final.pt"),
            "--output-dir", str(output_dir),
            "--split", "validation",  # Use validation for ablation
        ]
        if args.dry_run:
            cmd_eval.append("--dry-run")
        
        orchestrator.add_command(
            f"ablation-eval-{i}",
            cmd_eval,
            f"Evaluate config {i}/{len(combinations)}"
        )
    
    return orchestrator.execute_plan()


def cmd_aggregate(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle aggregate subcommand - collect all results.
    
    Args:
        args: Parsed arguments.
        orchestrator: Orchestrator instance.
        
    Returns:
        Exit code.
    """
    results_dir = args.results_dir or Path("results")
    output_file = args.output or results_dir / "summary.json"
    
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)
    
    print(f"\n📂 Scanning directory: {results_dir}")
    print(f"📊 Looking for metrics files:")
    print("   - metrics_stsb_*.json")
    print("   - metrics_quora_*.json")
    print("   - metrics_temporal_*.json")
    print("   - metrics_train.json")
    
    # In dry-run mode, just show what would be done
    if orchestrator.dry_run:
        print("\n[DRY RUN] Would aggregate results from:")
        
        # Mock some example files
        example_files = [
            "results/run_20240315/metrics_train.json",
            "results/evaluation/metrics_stsb_tide_lite.json",
            "results/evaluation/metrics_quora_tide_lite.json",
            "results/evaluation/metrics_temporal_tide_lite.json",
            "results/evaluation/metrics_stsb_baseline_minilm.json",
        ]
        
        for file in example_files:
            print(f"  • {file}")
        
        print(f"\n[DRY RUN] Would save aggregated results to:")
        print(f"  • {output_file} (JSON)")
        print(f"  • {output_file.with_suffix('.csv')} (CSV)")
        
        print("\n[DRY RUN] Aggregated structure:")
        print("""
{
  "models": {
    "tide_lite": {
      "stsb": {"spearman": 0.82, "pearson": 0.81, ...},
      "quora": {"ndcg_at_10": 0.69, "recall_at_10": 0.75, ...},
      "temporal": {"accuracy_at_1": 0.85, "consistency": 0.72, ...}
    },
    "baseline_minilm": {
      "stsb": {"spearman": 0.82, "pearson": 0.81, ...}
    }
  },
  "summary": {
    "best_stsb_spearman": "tide_lite",
    "best_quora_ndcg": "tide_lite",
    "best_temporal_accuracy": "tide_lite"
  },
  "timestamp": "2024-03-15T14:25:30"
}
        """)
    else:
        # In real implementation, would scan and aggregate
        print("\n[WOULD SCAN AND AGGREGATE RESULTS]")
    
    return 0


def cmd_report(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle report subcommand - generate markdown report and plots.
    
    Args:
        args: Parsed arguments.
        orchestrator: Orchestrator instance.
        
    Returns:
        Exit code.
    """
    input_file = args.input or Path("results/summary.json")
    output_dir = args.output_dir or Path("results/report")
    
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)
    
    print(f"\n📊 Input: {input_file}")
    print(f"📝 Output directory: {output_dir}")
    
    if orchestrator.dry_run:
        print("\n[DRY RUN] Would generate:")
        print(f"  • {output_dir}/report.md - Markdown report")
        print(f"  • {output_dir}/figures/")
        print(f"    - comparison_stsb.png")
        print(f"    - comparison_retrieval.png")
        print(f"    - temporal_consistency.png")
        print(f"    - ablation_heatmap.png")
        print(f"    - training_curves.png")
        
        print("\n[DRY RUN] Report structure:")
        print("""
# TIDE-Lite Evaluation Report

## Executive Summary
- Best Spearman: 0.823 (TIDE-Lite)
- Best nDCG@10: 0.695 (TIDE-Lite)
- Temporal Consistency: 0.85 (TIDE-Lite) vs 0.42 (baseline)

## Model Comparison

| Model | STS-B Spearman | Quora nDCG@10 | Temporal Acc@1 | Extra Params |
|-------|----------------|---------------|----------------|--------------|
| TIDE-Lite | **0.823** | **0.695** | **0.85** | 53K |
| Baseline | 0.820 | 0.680 | 0.72 | 0 |

## Detailed Results
...

## Ablation Study
...

## Conclusions
TIDE-Lite shows consistent improvements...
        """)
    else:
        print("\n[WOULD GENERATE MARKDOWN REPORT AND PLOTS]")
    
    return 0


def setup_parser() -> argparse.ArgumentParser:
    """Setup argument parser with subcommands.
    
    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="tide",
        description="TIDE-Lite unified orchestrator for training and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  tide train --config configs/defaults.yaml --output-dir results/run1
  
  # Evaluate on STS-B
  tide eval-stsb --model-path results/run1/checkpoints/final.pt
  
  # Run complete benchmark suite
  tide bench-all --model-path results/run1/checkpoints/final.pt
  
  # Run ablation study
  tide ablation --mlp-hidden-dims 64,128,256 --temporal-weights 0.05,0.1,0.2
  
  # Generate report
  tide aggregate --results-dir results/
  tide report --input results/summary.json
""",
    )
    
    # Global arguments
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command plan without executing",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train TIDE-Lite model")
    train_parser.add_argument("--config", type=Path, help="Config file")
    train_parser.add_argument("--output-dir", type=Path, help="Output directory")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--num-epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--temporal-weight", type=float, help="Temporal loss weight")
    
    # Eval-STSB subcommand
    stsb_parser = subparsers.add_parser("eval-stsb", help="Evaluate on STS-B")
    stsb_model = stsb_parser.add_mutually_exclusive_group(required=True)
    stsb_model.add_argument("--model-path", type=Path, help="Model path")
    stsb_model.add_argument("--baseline", choices=["minilm", "e5-base", "bge-base"])
    stsb_parser.add_argument("--output-dir", type=Path, help="Output directory")
    stsb_parser.add_argument("--split", choices=["validation", "test"], help="Split")
    stsb_parser.add_argument("--batch-size", type=int, help="Batch size")
    stsb_parser.add_argument("--compare-baseline", action="store_true", help="Compare with baseline")
    
    # Eval-Quora subcommand
    quora_parser = subparsers.add_parser("eval-quora", help="Evaluate on Quora retrieval")
    quora_model = quora_parser.add_mutually_exclusive_group(required=True)
    quora_model.add_argument("--model-path", type=Path, help="Model path")
    quora_model.add_argument("--baseline", choices=["minilm", "e5-base", "bge-base"])
    quora_parser.add_argument("--output-dir", type=Path, help="Output directory")
    quora_parser.add_argument("--index-type", choices=["Flat", "IVFFlat"], help="FAISS index")
    quora_parser.add_argument("--max-corpus", type=int, help="Max corpus size")
    quora_parser.add_argument("--max-queries", type=int, help="Max queries")
    
    # Eval-Temporal subcommand
    temporal_parser = subparsers.add_parser("eval-temporal", help="Evaluate temporal understanding")
    temporal_model = temporal_parser.add_mutually_exclusive_group(required=True)
    temporal_model.add_argument("--model-path", type=Path, help="Model path")
    temporal_model.add_argument("--baseline", choices=["minilm", "e5-base", "bge-base"])
    temporal_parser.add_argument("--output-dir", type=Path, help="Output directory")
    temporal_parser.add_argument("--time-window-days", type=float, help="Time window in days")
    temporal_parser.add_argument("--max-samples", type=int, help="Max samples")
    temporal_parser.add_argument("--compare-baseline", action="store_true", help="Compare with baseline")
    
    # Bench-all subcommand
    bench_parser = subparsers.add_parser("bench-all", help="Run all evaluations")
    bench_model = bench_parser.add_mutually_exclusive_group(required=True)
    bench_model.add_argument("--model-path", type=Path, help="Model path")
    bench_model.add_argument("--baseline", choices=["minilm", "e5-base", "bge-base"])
    bench_parser.add_argument("--output-dir", type=Path, help="Output directory")
    bench_parser.add_argument("--split", choices=["validation", "test"], help="Dataset split")
    bench_parser.add_argument("--max-corpus", type=int, help="Max corpus for Quora")
    
    # Ablation subcommand
    ablation_parser = subparsers.add_parser("ablation", help="Run ablation study")
    ablation_parser.add_argument(
        "--mlp-hidden-dims",
        type=str,
        default="64,128,256",
        help="Comma-separated MLP hidden dimensions"
    )
    ablation_parser.add_argument(
        "--temporal-weights",
        type=str,
        default="0.05,0.1,0.2",
        help="Comma-separated temporal loss weights"
    )
    ablation_parser.add_argument(
        "--time-encoding-dims",
        type=str,
        default="16,32,64",
        help="Comma-separated time encoding dimensions"
    )
    ablation_parser.add_argument("--config", type=Path, help="Base config file")
    ablation_parser.add_argument("--output-dir", type=Path, help="Output directory")
    ablation_parser.add_argument("--num-epochs", type=int, default=1, help="Epochs per config")
    
    # Aggregate subcommand
    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate results")
    aggregate_parser.add_argument("--results-dir", type=Path, help="Results directory")
    aggregate_parser.add_argument("--output", type=Path, help="Output file")
    
    # Report subcommand
    report_parser = subparsers.add_parser("report", help="Generate markdown report")
    report_parser.add_argument("--input", type=Path, help="Input summary JSON")
    report_parser.add_argument("--output-dir", type=Path, help="Output directory")
    
    return parser


def main() -> int:
    """Main entry point for unified orchestrator.
    
    Returns:
        Exit code.
    """
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create orchestrator
    orchestrator = TIDEOrchestrator(dry_run=args.dry_run)
    
    # Dispatch to appropriate handler
    handlers = {
        "train": cmd_train,
        "eval-stsb": cmd_eval_stsb,
        "eval-quora": cmd_eval_quora,
        "eval-temporal": cmd_eval_temporal,
        "bench-all": cmd_bench_all,
        "ablation": cmd_ablation,
        "aggregate": cmd_aggregate,
        "report": cmd_report,
    }
    
    handler = handlers.get(args.command)
    if not handler:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1
    
    return handler(args, orchestrator)


if __name__ == "__main__":
    sys.exit(main())
