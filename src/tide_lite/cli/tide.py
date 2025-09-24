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
    
    def __init__(self, dry_run: bool = True, run: bool = False) -> None:
        """Initialize orchestrator.
        
        Args:
            dry_run: If True, only print commands without executing.
            run: If True, actually execute commands (overrides dry_run).
        """
        self.dry_run = not run if run else dry_run
        self.commands_to_run: List[Dict[str, Any]] = []
    
    def add_command(
        self,
        name: str,
        cmd: List[str],
        description: str = "",
        output_path: Optional[str] = None,
    ) -> None:
        """Add a command to the execution queue.
        
        Args:
            name: Command identifier.
            cmd: Command as list of arguments.
            description: Human-readable description.
            output_path: Expected output file path.
        """
        self.commands_to_run.append({
            "name": name,
            "cmd": cmd,
            "description": description,
            "output_path": output_path,
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
                print(f"    📝 {command['description']}")
            print(f"    💻 Command: {' '.join(command['cmd'])}")
            if command['output_path']:
                print(f"    📄 Output: {command['output_path']}")
        
        if self.dry_run:
            print("\n" + "=" * 70)
            print("🔍 DRY RUN MODE - Commands above would be executed in sequence")
            print("💡 Use --run to actually execute these commands")
            print("=" * 70)
            return 0
        
        print("\n" + "=" * 70)
        print("🚀 EXECUTING COMMANDS")
        print("=" * 70)
        
        for i, command in enumerate(self.commands_to_run, 1):
            print(f"\n[{i}/{len(self.commands_to_run)}] Executing: {command['name']}")
            
            try:
                result = subprocess.run(
                    command['cmd'],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print(f"✅ Success: {command['name']}")
                if command['output_path'] and Path(command['output_path']).exists():
                    print(f"   📄 Created: {command['output_path']}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed: {command['name']}")
                print(f"   Error: {e.stderr}")
                return e.returncode
            except FileNotFoundError:
                print(f"❌ Command not found: {command['cmd'][0]}")
                return 127
        
        print("\n✅ All commands completed successfully!")
        return 0


def cmd_train(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle train subcommand."""
    cmd = ["python", "-m", "tide_lite.cli.train"]
    
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
    if args.consistency_weight:
        cmd.extend(["--consistency-weight", str(args.consistency_weight)])
    
    if not orchestrator.dry_run:
        cmd.append("--run")
    
    output_dir = args.output_dir or Path("results") / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    orchestrator.add_command(
        "train",
        cmd,
        f"Train TIDE-Lite model for {args.num_epochs or 3} epochs",
        str(output_dir / "metrics_train.json"),
    )
    
    return orchestrator.execute_plan()


def cmd_eval_stsb(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle eval-stsb subcommand."""
    cmd = ["python", "-m", "tide_lite.cli.eval_stsb"]
    
    cmd.extend(["--model", str(args.model)])
    
    if args.type:
        cmd.extend(["--type", args.type])
    if args.output_dir:
        cmd.extend(["--output-dir", str(args.output_dir)])
    if args.split:
        cmd.extend(["--split", args.split])
    
    if not orchestrator.dry_run:
        cmd.append("--run")
    
    model_name = Path(str(args.model)).stem if "/" in str(args.model) else str(args.model)
    output_dir = args.output_dir or Path("results")
    
    orchestrator.add_command(
        "eval-stsb",
        cmd,
        f"Evaluate on STS-B {args.split or 'test'} split",
        str(output_dir / f"metrics_stsb_{model_name}.json"),
    )
    
    return orchestrator.execute_plan()


def cmd_eval_quora(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle eval-quora subcommand."""
    cmd = ["python", "-m", "tide_lite.cli.eval_quora"]
    
    cmd.extend(["--model", str(args.model)])
    
    if args.type:
        cmd.extend(["--type", args.type])
    if args.output_dir:
        cmd.extend(["--output-dir", str(args.output_dir)])
    if args.index_type:
        cmd.extend(["--index-type", args.index_type])
    if args.max_corpus:
        cmd.extend(["--max-corpus", str(args.max_corpus)])
    if args.max_queries:
        cmd.extend(["--max-queries", str(args.max_queries)])
    
    if not orchestrator.dry_run:
        cmd.append("--run")
    
    model_name = Path(str(args.model)).stem if "/" in str(args.model) else str(args.model)
    output_dir = args.output_dir or Path("results")
    
    orchestrator.add_command(
        "eval-quora",
        cmd,
        f"Evaluate on Quora retrieval with {args.index_type or 'Flat'} index",
        str(output_dir / f"metrics_quora_{model_name}.json"),
    )
    
    return orchestrator.execute_plan()


def cmd_eval_temporal(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle eval-temporal subcommand."""
    cmd = ["python", "-m", "tide_lite.cli.eval_temporal"]
    
    cmd.extend(["--model", str(args.model)])
    
    if args.type:
        cmd.extend(["--type", args.type])
    if args.output_dir:
        cmd.extend(["--output-dir", str(args.output_dir)])
    if args.time_window_days:
        cmd.extend(["--time-window-days", str(args.time_window_days)])
    
    if not orchestrator.dry_run:
        cmd.append("--run")
    
    model_name = Path(str(args.model)).stem if "/" in str(args.model) else str(args.model)
    output_dir = args.output_dir or Path("results")
    
    orchestrator.add_command(
        "eval-temporal",
        cmd,
        f"Evaluate temporal understanding (window: {args.time_window_days or 30} days)",
        str(output_dir / f"metrics_temporal_{model_name}.json"),
    )
    
    return orchestrator.execute_plan()


def cmd_bench_all(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle bench-all subcommand - runs all three evaluations."""
    output_dir = args.output_dir or Path("results")
    model_type = args.type or ("baseline" if args.model in ["minilm", "e5-base", "bge-base"] else "tide_lite")
    
    # STS-B evaluation
    cmd_stsb = ["python", "-m", "tide_lite.cli.eval_stsb"]
    cmd_stsb.extend(["--model", str(args.model)])
    cmd_stsb.extend(["--type", model_type])
    cmd_stsb.extend(["--output-dir", str(output_dir)])
    if not orchestrator.dry_run:
        cmd_stsb.append("--run")
    
    orchestrator.add_command(
        "bench-stsb",
        cmd_stsb,
        "Benchmark on STS-B semantic similarity"
    )
    
    # Quora retrieval evaluation
    cmd_quora = ["python", "-m", "tide_lite.cli.eval_quora"]
    cmd_quora.extend(["--model", str(args.model)])
    cmd_quora.extend(["--type", model_type])
    cmd_quora.extend(["--output-dir", str(output_dir)])
    cmd_quora.extend(["--max-corpus", "10000"])
    cmd_quora.extend(["--max-queries", "1000"])
    if not orchestrator.dry_run:
        cmd_quora.append("--run")
    
    orchestrator.add_command(
        "bench-quora",
        cmd_quora,
        "Benchmark on Quora retrieval"
    )
    
    # Temporal evaluation (if not skipped)
    if not args.skip_temporal:
        cmd_temporal = ["python", "-m", "tide_lite.cli.eval_temporal"]
        cmd_temporal.extend(["--model", str(args.model)])
        cmd_temporal.extend(["--type", model_type])
        cmd_temporal.extend(["--output-dir", str(output_dir)])
        if not orchestrator.dry_run:
            cmd_temporal.append("--run")
        
        orchestrator.add_command(
            "bench-temporal",
            cmd_temporal,
            "Benchmark on temporal understanding"
        )
    else:
        print("⚠️ Skipping temporal evaluation (--skip-temporal flag set)")
    
    return orchestrator.execute_plan()


def cmd_ablation(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle ablation subcommand - parameter grid search."""
    # Parse parameter ranges
    mlp_hidden_dims = [int(x) for x in args.time_mlp_hidden.split(",")]
    consistency_weights = [float(x) for x in args.consistency_weight.split(",")]
    time_encodings = args.time_encoding.split(",")
    
    output_base = args.output_dir or Path("results/ablation")
    
    # Generate all combinations
    combinations = list(product(mlp_hidden_dims, consistency_weights, time_encodings))
    
    print(f"\nAblation study: {len(combinations)} configurations")
    print(f"MLP hidden dims: {mlp_hidden_dims}")
    print(f"Consistency weights: {consistency_weights}")
    print(f"Time encodings: {time_encodings}")
    
    for i, (mlp_dim, weight, encoding) in enumerate(combinations, 1):
        run_name = f"ablation_mlp{mlp_dim}_w{weight}_enc{encoding}"
        output_dir = output_base / run_name
        
        # Training command
        cmd_train = [
            "python", "-m", "tide_lite.cli.train",
            "--output-dir", str(output_dir),
            "--time-mlp-hidden", str(mlp_dim),
            "--consistency-weight", str(weight),
            "--time-encoding", encoding,
            "--num-epochs", "1",  # Quick ablation
        ]
        if not orchestrator.dry_run:
            cmd_train.append("--run")
        
        orchestrator.add_command(
            f"ablation-train-{i}",
            cmd_train,
            f"Config {i}/{len(combinations)}: mlp={mlp_dim}, weight={weight}, enc={encoding}"
        )
        
        # Evaluation command (STS-B validation)
        cmd_eval = [
            "python", "-m", "tide_lite.cli.eval_stsb",
            "--model", str(output_dir / "checkpoints" / "best_model.pt"),
            "--output-dir", str(output_dir),
            "--split", "validation",
        ]
        if not orchestrator.dry_run:
            cmd_eval.append("--run")
        
        orchestrator.add_command(
            f"ablation-eval-{i}",
            cmd_eval,
            f"Evaluate config {i}/{len(combinations)}"
        )
    
    # Save ablation results summary
    cmd_summary = [
        "python", "-m", "tide_lite.cli.aggregate",
        "--results-dir", str(output_base),
        "--output", str(output_base / "ablation_summary.json"),
    ]
    if not orchestrator.dry_run:
        cmd_summary.append("--run")
    
    orchestrator.add_command(
        "ablation-summary",
        cmd_summary,
        "Aggregate ablation results"
    )
    
    return orchestrator.execute_plan()


def cmd_aggregate(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle aggregate subcommand - collect all results."""
    cmd = ["python", "-m", "tide_lite.cli.aggregate"]
    
    if args.results_dir:
        cmd.extend(["--results-dir", str(args.results_dir)])
    if args.output:
        cmd.extend(["--output", str(args.output)])
    
    if not orchestrator.dry_run:
        cmd.append("--run")
    
    orchestrator.add_command(
        "aggregate",
        cmd,
        "Aggregate all metrics into summary",
        str(args.output or Path("results/summary.json"))
    )
    
    return orchestrator.execute_plan()


def cmd_report(args: argparse.Namespace, orchestrator: TIDEOrchestrator) -> int:
    """Handle report subcommand - generate markdown report."""
    cmd = ["python", "-m", "tide_lite.cli.report"]
    
    if args.input:
        cmd.extend(["--input", str(args.input)])
    if args.output_dir:
        cmd.extend(["--output-dir", str(args.output_dir)])
    
    if not orchestrator.dry_run:
        cmd.append("--run")
    
    output_dir = args.output_dir or Path("reports")
    
    orchestrator.add_command(
        "report",
        cmd,
        "Generate markdown report and plots",
        str(output_dir / "report.md")
    )
    
    return orchestrator.execute_plan()


def setup_parser() -> argparse.ArgumentParser:
    """Setup argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="tide",
        description="TIDE-Lite unified orchestrator for training and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Global arguments
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print command plan without executing (default)",
    )
    
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually execute commands",
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
    train_parser.add_argument("--consistency-weight", type=float, help="Temporal consistency weight")
    
    # Eval-STSB subcommand
    stsb_parser = subparsers.add_parser("eval-stsb", help="Evaluate on STS-B")
    stsb_parser.add_argument("--model", required=True, help="Model path or ID")
    stsb_parser.add_argument("--type", choices=["tide_lite", "baseline"], help="Model type")
    stsb_parser.add_argument("--output-dir", type=Path, help="Output directory")
    stsb_parser.add_argument("--split", choices=["validation", "test"], help="Dataset split")
    
    # Eval-Quora subcommand
    quora_parser = subparsers.add_parser("eval-quora", help="Evaluate on Quora retrieval")
    quora_parser.add_argument("--model", required=True, help="Model path or ID")
    quora_parser.add_argument("--type", choices=["tide_lite", "baseline"], help="Model type")
    quora_parser.add_argument("--output-dir", type=Path, help="Output directory")
    quora_parser.add_argument("--index-type", choices=["Flat", "IVF"], help="FAISS index type")
    quora_parser.add_argument("--max-corpus", type=int, help="Max corpus size")
    quora_parser.add_argument("--max-queries", type=int, help="Max queries")
    
    # Eval-Temporal subcommand
    temporal_parser = subparsers.add_parser("eval-temporal", help="Evaluate temporal understanding")
    temporal_parser.add_argument("--model", required=True, help="Model path or ID")
    temporal_parser.add_argument("--type", choices=["tide_lite", "baseline"], help="Model type")
    temporal_parser.add_argument("--output-dir", type=Path, help="Output directory")
    temporal_parser.add_argument("--time-window-days", type=float, help="Time window in days")
    
    # Bench-all subcommand
    bench_parser = subparsers.add_parser("bench-all", help="Run all evaluations")
    bench_parser.add_argument("--model", required=True, help="Model path or ID")
    bench_parser.add_argument("--type", choices=["tide_lite", "baseline"], help="Model type")
    bench_parser.add_argument("--output-dir", type=Path, help="Output directory")
    bench_parser.add_argument(
        "--skip-temporal",
        action="store_true",
        help="Skip temporal evaluation if datasets are not available"
    )
    
    # Ablation subcommand
    ablation_parser = subparsers.add_parser("ablation", help="Run ablation study")
    ablation_parser.add_argument(
        "--time-mlp-hidden",
        type=str,
        default="64,128,256",
        help="Comma-separated MLP hidden dimensions"
    )
    ablation_parser.add_argument(
        "--consistency-weight",
        type=str,
        default="0.05,0.1,0.2",
        help="Comma-separated consistency weights"
    )
    ablation_parser.add_argument(
        "--time-encoding",
        type=str,
        default="sinusoidal,learnable",
        help="Comma-separated time encoding types"
    )
    ablation_parser.add_argument("--output-dir", type=Path, help="Output directory")
    
    # Aggregate subcommand
    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate results")
    aggregate_parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Results directory")
    aggregate_parser.add_argument("--output", type=Path, help="Output file")
    
    # Report subcommand
    report_parser = subparsers.add_parser("report", help="Generate markdown report")
    report_parser.add_argument("--input", type=Path, help="Input summary JSON")
    report_parser.add_argument("--output-dir", type=Path, help="Output directory")
    
    return parser


def main() -> int:
    """Main entry point for unified orchestrator."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create orchestrator (dry-run by default unless --run is specified)
    orchestrator = TIDEOrchestrator(dry_run=not args.run, run=args.run)
    
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
