"""Result aggregation utilities for TIDE-Lite experiments.

This module aggregates metrics from multiple evaluation runs
into combined CSV and JSON summaries.
"""

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AggregatedMetrics:
    """Container for aggregated experiment metrics.

    Attributes:
        models: Nested dict of model -> task -> metrics.
        best_performers: Dict of metric -> best model name.
        comparisons: Pairwise model comparisons.
        metadata: Aggregation metadata (timestamp, file count, etc.).
    """
    models: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    best_performers: Dict[str, str] = field(default_factory=dict)
    comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResultAggregator:
    """Aggregates results from multiple TIDE-Lite evaluation runs."""

    def __init__(
        self,
        results_dir: Path,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> None:
        """Initialize result aggregator.

        Args:
            results_dir: Root directory containing results.
            include_patterns: Glob patterns for files to include.
            exclude_patterns: Glob patterns for files to exclude.
        """
        self.results_dir = Path(results_dir)
        self.include_patterns = include_patterns or [
            "**/metrics_*.json",
            "**/metrics_train.json",
        ]
        self.exclude_patterns = exclude_patterns or [
            "**/*backup*",
            "**/*tmp*",
        ]

        self.metrics_files: List[Path] = []
        self.aggregated: Optional[AggregatedMetrics] = None

        logger.info(f"Initialized aggregator for directory: {results_dir}")

    def find_metrics_files(self) -> List[Path]:
        """Find all metrics JSON files in results directory.

        Returns:
            List of paths to metrics files.
        """
        all_files = set()

        # Find files matching include patterns
        for pattern in self.include_patterns:
            matching = self.results_dir.glob(pattern)
            all_files.update(matching)

        # Remove files matching exclude patterns
        if self.exclude_patterns:
            excluded = set()
            for pattern in self.exclude_patterns:
                excluded.update(self.results_dir.glob(pattern))
            all_files -= excluded

        # Sort for consistent ordering
        self.metrics_files = sorted(all_files)

        logger.info(f"Found {len(self.metrics_files)} metrics files")
        return self.metrics_files

    def parse_metrics_file(self, filepath: Path) -> Tuple[str, str, Dict[str, Any]]:
        """Parse a single metrics file.

        Args:
            filepath: Path to metrics JSON file.

        Returns:
            Tuple of (model_name, task_name, metrics_dict).
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Extract model and task from filename or content
        filename = filepath.stem  # e.g., "metrics_stsb_tide_lite"

        # Try to get from file content first
        model_name = data.get("model_name", "unknown")
        task_name = data.get("task", "unknown")

        # Fallback to parsing filename
        if model_name == "unknown" or task_name == "unknown":
            parts = filename.split("_")
            if len(parts) >= 3 and parts[0] == "metrics":
                task_name = parts[1]
                model_name = "_".join(parts[2:])
            elif filename == "metrics_train":
                task_name = "training"
                # Get model from parent directory
                model_name = filepath.parent.parent.name

        # Clean up metrics (remove metadata fields)
        metrics = {
            k: v for k, v in data.items()
            if k not in ["model_name", "task", "timestamp"]
        }

        return model_name, task_name, metrics

    def aggregate(self) -> AggregatedMetrics:
        """Aggregate all metrics files.

        Returns:
            AggregatedMetrics object with all results.
        """
        if not self.metrics_files:
            self.find_metrics_files()

        aggregated = AggregatedMetrics()
        aggregated.metadata["timestamp"] = datetime.now().isoformat()
        aggregated.metadata["results_dir"] = str(self.results_dir)
        aggregated.metadata["num_files"] = len(self.metrics_files)

        # Collect all metrics
        for filepath in self.metrics_files:
            try:
                model_name, task_name, metrics = self.parse_metrics_file(filepath)

                if model_name not in aggregated.models:
                    aggregated.models[model_name] = {}

                aggregated.models[model_name][task_name] = metrics

                logger.debug(f"Loaded {task_name} metrics for {model_name}")

            except Exception as e:
                logger.warning(f"Failed to parse {filepath}: {e}")

        # Find best performers for key metrics
        key_metrics = {
            "stsb_spearman": ("STS-B", "spearman"),
            "quora_ndcg": ("Quora Retrieval", "ndcg_at_10"),
            "temporal_accuracy": ("Temporal QA", "temporal_accuracy_at_1"),
            "temporal_consistency": ("Temporal QA", "temporal_consistency_score"),
        }

        for metric_name, (task, field) in key_metrics.items():
            best_value = -float("inf")
            best_model = None

            for model_name, tasks in aggregated.models.items():
                if task in tasks and field in tasks[task]:
                    value = tasks[task][field]
                    if value > best_value:
                        best_value = value
                        best_model = model_name

            if best_model:
                aggregated.best_performers[metric_name] = f"{best_model} ({best_value:.4f})"

        # Compute comparisons (TIDE vs baseline)
        tide_models = [m for m in aggregated.models if "tide" in m.lower()]
        baseline_models = [m for m in aggregated.models if "baseline" in m.lower()]

        if tide_models and baseline_models:
            # Take first of each for comparison
            tide_model = tide_models[0]
            baseline_model = baseline_models[0]

            for task in aggregated.models.get(tide_model, {}):
                if task in aggregated.models.get(baseline_model, {}):
                    tide_metrics = aggregated.models[tide_model][task]
                    baseline_metrics = aggregated.models[baseline_model][task]

                    comparison_key = f"{task}_improvement"
                    aggregated.comparisons[comparison_key] = {}

                    # Compute improvements for numeric metrics
                    for metric in tide_metrics:
                        if (
                            isinstance(tide_metrics[metric], (int, float)) and
                            metric in baseline_metrics and
                            isinstance(baseline_metrics[metric], (int, float))
                        ):
                            improvement = tide_metrics[metric] - baseline_metrics[metric]
                            aggregated.comparisons[comparison_key][metric] = improvement

        self.aggregated = aggregated
        logger.info(f"Aggregated {len(aggregated.models)} models across {len(self.metrics_files)} files")

        return aggregated

    def to_dataframe(self) -> pd.DataFrame:
        """Convert aggregated metrics to pandas DataFrame.

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        if not self.aggregated:
            self.aggregate()

        rows = []

        for model_name, tasks in self.aggregated.models.items():
            row = {"model": model_name}

            for task_name, metrics in tasks.items():
                for metric_name, value in metrics.items():
                    # Create column name as task_metric
                    col_name = f"{task_name}_{metric_name}"
                    row[col_name] = value

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by model name
        df = df.sort_values("model")

        # Reorder columns to put model first
        cols = ["model"] + [c for c in df.columns if c != "model"]
        df = df[cols]

        return df

    def save_csv(self, output_path: Path) -> None:
        """Save aggregated results to CSV file.

        Args:
            output_path: Path for output CSV file.
        """
        df = self.to_dataframe()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Saved CSV to {output_path}")

    def save_json(self, output_path: Path) -> None:
        """Save aggregated results to JSON file.

        Args:
            output_path: Path for output JSON file.
        """
        if not self.aggregated:
            self.aggregate()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        output_data = {
            "models": self.aggregated.models,
            "best_performers": self.aggregated.best_performers,
            "comparisons": self.aggregated.comparisons,
            "metadata": self.aggregated.metadata,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved JSON summary to {output_path}")

    def generate_markdown_summary(self) -> str:
        """Generate markdown summary of results.

        Returns:
            Markdown-formatted summary string.
        """
        if not self.aggregated:
            self.aggregate()

        lines = []
        lines.append("# TIDE-Lite Results Summary")
        lines.append("")
        lines.append(f"Generated: {self.aggregated.metadata['timestamp']}")
        lines.append(f"Files analyzed: {self.aggregated.metadata['num_files']}")
        lines.append("")

        # Best performers
        if self.aggregated.best_performers:
            lines.append("## Best Performers")
            lines.append("")
            for metric, best in self.aggregated.best_performers.items():
                lines.append(f"- **{metric}**: {best}")
            lines.append("")

        # Model comparison table
        lines.append("## Model Comparison")
        lines.append("")

        df = self.to_dataframe()

        # Select key columns for display
        display_cols = ["model"]
        for col in df.columns:
            if any(key in col.lower() for key in ["spearman", "ndcg", "accuracy", "consistency"]):
                display_cols.append(col)

        if len(display_cols) > 1:
            display_df = df[display_cols].copy()

            # Format numeric columns
            for col in display_cols[1:]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) and isinstance(x, (int, float)) else str(x)
                    )

            lines.append(display_df.to_markdown(index=False))

        # Improvements
        if self.aggregated.comparisons:
            lines.append("")
            lines.append("## Improvements over Baseline")
            lines.append("")

            for comparison_key, metrics in self.aggregated.comparisons.items():
                task = comparison_key.replace("_improvement", "")
                lines.append(f"### {task}")
                lines.append("")

                for metric, improvement in metrics.items():
                    sign = "+" if improvement > 0 else ""
                    lines.append(f"- {metric}: {sign}{improvement:.4f}")
                lines.append("")

        return "\n".join(lines)

    def filter_ablation_results(self) -> Dict[str, Dict[str, Any]]:
        """Extract and organize ablation study results.

        Returns:
            Dictionary mapping ablation configs to metrics.
        """
        if not self.aggregated:
            self.aggregate()

        ablation_results = {}

        for model_name, tasks in self.aggregated.models.items():
            # Check if this is an ablation run
            if "ablation" in model_name.lower():
                # Try to parse configuration from name
                # Expected format: ablation_mlp128_tw0.1_td32
                config = self._parse_ablation_config(model_name)

                if config:
                    ablation_results[model_name] = {
                        "config": config,
                        "metrics": tasks,
                    }

        logger.info(f"Found {len(ablation_results)} ablation configurations")
        return ablation_results

    def _parse_ablation_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Parse ablation configuration from model name.

        Args:
            model_name: Model name containing config info.

        Returns:
            Dictionary with parsed configuration or None.
        """
        import re

        config = {}

        # Parse MLP hidden dimension
        mlp_match = re.search(r"mlp(\d+)", model_name.lower())
        if mlp_match:
            config["mlp_hidden_dim"] = int(mlp_match.group(1))

        # Parse temporal weight
        tw_match = re.search(r"tw([\d.]+)", model_name.lower())
        if tw_match:
            config["temporal_weight"] = float(tw_match.group(1))

        # Parse time encoding dimension
        td_match = re.search(r"td(\d+)", model_name.lower())
        if td_match:
            config["time_encoding_dim"] = int(td_match.group(1))

        return config if config else None


def aggregate_results(
    results_dir,
    output_dir = None,
    save_csv: bool = True,
    save_json: bool = True,
    save_markdown: bool = True,
) -> AggregatedMetrics:
    """Convenience function to aggregate results.

    Args:
        results_dir: Directory containing results.
        output_dir: Output directory (defaults to results_dir).
        save_csv: Whether to save CSV output.
        save_json: Whether to save JSON output.
        save_markdown: Whether to save markdown summary.

    Returns:
        AggregatedMetrics object.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir

    aggregator = ResultAggregator(results_dir)
    aggregated = aggregator.aggregate()

    if save_csv:
        aggregator.save_csv(output_dir / "summary.csv")

    if save_json:
        aggregator.save_json(output_dir / "summary.json")

    if save_markdown:
        markdown = aggregator.generate_markdown_summary()
        with open(output_dir / "summary.md", "w") as f:
            f.write(markdown)
        logger.info(f"Saved markdown summary to {output_dir / 'summary.md'}")

    return aggregated
