"""Aggregation module for collecting and merging evaluation results.

This module scans results directories and merges all metrics into
unified summary files (JSON and CSV formats).
"""

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of aggregation process."""
    
    summary_json: Path
    summary_csv: Path
    num_files_processed: int
    models_found: List[str]
    metrics_found: List[str]


class MetricsAggregator:
    """Aggregates metrics from multiple evaluation runs."""
    
    def __init__(self, results_dir: Path = Path("results")) -> None:
        """Initialize aggregator.
        
        Args:
            results_dir: Directory containing metrics files.
        """
        self.results_dir = Path(results_dir)
        self.metrics_files: List[Path] = []
        self.aggregated_data: Dict[str, Dict[str, Any]] = {}
    
    def scan_metrics_files(self) -> List[Path]:
        """Scan for metrics JSON files.
        
        Returns:
            List of found metrics files.
        """
        patterns = [
            "metrics_*.json",
            "*/metrics_*.json",
            "*/*/metrics_*.json",
        ]
        
        files = []
        for pattern in patterns:
            files.extend(self.results_dir.glob(pattern))
        
        # Also look for ablation results
        ablation_dirs = self.results_dir.glob("ablation_*")
        for ablation_dir in ablation_dirs:
            files.extend(ablation_dir.glob("metrics_*.json"))
        
        self.metrics_files = sorted(set(files))
        logger.info(f"Found {len(self.metrics_files)} metrics files")
        
        return self.metrics_files
    
    def load_metrics_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a single metrics file.
        
        Args:
            file_path: Path to metrics JSON file.
            
        Returns:
            Loaded metrics dictionary.
        """
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            # Extract metadata from filename
            filename = file_path.stem  # e.g., "metrics_stsb_minilm"
            parts = filename.split("_")
            
            if len(parts) >= 3:
                task = parts[1]  # stsb, quora, temporal
                model = "_".join(parts[2:])  # model name
            else:
                task = "unknown"
                model = filename
            
            # Add metadata
            data["_metadata"] = {
                "file": str(file_path),
                "task": task,
                "model": model,
            }
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            return {}
    
    def aggregate(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate all metrics.
        
        Returns:
            Nested dictionary: {model: {task: metrics}}.
        """
        self.scan_metrics_files()
        
        aggregated = {}
        
        for file_path in self.metrics_files:
            data = self.load_metrics_file(file_path)
            if not data:
                continue
            
            metadata = data.get("_metadata", {})
            model = metadata.get("model", "unknown")
            task = metadata.get("task", "unknown")
            
            if model not in aggregated:
                aggregated[model] = {}
            
            # Extract key metrics based on task
            if task == "stsb":
                metrics = {
                    "spearman": data.get("metrics", {}).get("spearman_correlation", 0),
                    "pearson": data.get("metrics", {}).get("pearson_correlation", 0),
                    "mse": data.get("metrics", {}).get("mse", 0),
                }
            elif task == "quora":
                metrics = {
                    "ndcg_at_10": data.get("metrics", {}).get("ndcg_at_10", 0),
                    "recall_at_10": data.get("metrics", {}).get("recall_at_10", 0),
                    "mrr_at_10": data.get("metrics", {}).get("mrr_at_10", 0),
                    "latency_ms": data.get("metrics", {}).get("latency_median_ms", 0),
                }
            elif task == "temporal":
                metrics = {
                    "accuracy_at_1": data.get("metrics", {}).get("temporal_accuracy_at_1", 0),
                    "accuracy_at_5": data.get("metrics", {}).get("temporal_accuracy_at_5", 0),
                    "consistency": data.get("metrics", {}).get("temporal_consistency_score", 0),
                    "drift_days": data.get("metrics", {}).get("time_drift_mae_days", 0),
                }
            elif task == "train":
                metrics = {
                    "final_loss": data.get("final_loss", 0),
                    "best_val_score": data.get("best_val_score", 0),
                    "total_epochs": data.get("total_epochs", 0),
                    "training_time_s": data.get("training_time_s", 0),
                }
            else:
                # Generic metrics extraction
                metrics = data.get("metrics", data)
            
            aggregated[model][task] = metrics
        
        self.aggregated_data = aggregated
        return aggregated
    
    def compute_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics across models.
        
        Returns:
            Summary statistics.
        """
        if not self.aggregated_data:
            self.aggregate()
        
        summary = {
            "num_models": len(self.aggregated_data),
            "models": list(self.aggregated_data.keys()),
            "best_performers": {},
        }
        
        # Find best performer for each metric
        metrics_to_compare = {
            "stsb": ["spearman", "pearson"],
            "quora": ["ndcg_at_10", "recall_at_10"],
            "temporal": ["accuracy_at_1", "consistency"],
        }
        
        for task, metrics in metrics_to_compare.items():
            for metric in metrics:
                best_model = None
                best_value = -float("inf")
                
                for model, tasks in self.aggregated_data.items():
                    if task in tasks and metric in tasks[task]:
                        value = tasks[task][metric]
                        if value > best_value:
                            best_value = value
                            best_model = model
                
                if best_model:
                    summary["best_performers"][f"{task}_{metric}"] = {
                        "model": best_model,
                        "value": best_value,
                    }
        
        return summary
    
    def save_json(self, output_path: Path) -> None:
        """Save aggregated results to JSON.
        
        Args:
            output_path: Output JSON file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "models": self.aggregated_data,
            "summary": self.compute_summary_stats(),
            "metadata": {
                "num_files_processed": len(self.metrics_files),
                "source_dir": str(self.results_dir),
            },
        }
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved aggregated results to {output_path}")
    
    def save_csv(self, output_path: Path) -> None:
        """Save aggregated results to CSV.
        
        Args:
            output_path: Output CSV file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten nested structure for CSV
        rows = []
        for model, tasks in self.aggregated_data.items():
            for task, metrics in tasks.items():
                row = {"model": model, "task": task}
                if isinstance(metrics, dict):
                    row.update(metrics)
                else:
                    row["value"] = metrics
                rows.append(row)
        
        # Convert to DataFrame and save
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved aggregated results to {output_path}")
        else:
            logger.warning("No data to save to CSV")
    
    def run(
        self,
        output_json: Optional[Path] = None,
        output_csv: Optional[Path] = None,
        dry_run: bool = False,
    ) -> AggregationResult:
        """Run aggregation process.
        
        Args:
            output_json: JSON output path (default: results/summary.json).
            output_csv: CSV output path (default: results/summary.csv).
            dry_run: If True, only show what would be done.
            
        Returns:
            AggregationResult with paths and statistics.
        """
        if output_json is None:
            output_json = self.results_dir / "summary.json"
        if output_csv is None:
            output_csv = self.results_dir / "summary.csv"
        
        # Scan and aggregate
        self.scan_metrics_files()
        
        if dry_run:
            logger.info("[DRY RUN] Would aggregate metrics from:")
            for file in self.metrics_files[:10]:  # Show first 10
                logger.info(f"  - {file}")
            if len(self.metrics_files) > 10:
                logger.info(f"  ... and {len(self.metrics_files) - 10} more files")
            
            logger.info(f"[DRY RUN] Would save to:")
            logger.info(f"  - {output_json}")
            logger.info(f"  - {output_csv}")
            
            return AggregationResult(
                summary_json=output_json,
                summary_csv=output_csv,
                num_files_processed=len(self.metrics_files),
                models_found=[],
                metrics_found=[],
            )
        
        # Actually aggregate
        self.aggregate()
        
        # Save results
        self.save_json(output_json)
        self.save_csv(output_csv)
        
        return AggregationResult(
            summary_json=output_json,
            summary_csv=output_csv,
            num_files_processed=len(self.metrics_files),
            models_found=list(self.aggregated_data.keys()),
            metrics_found=list(set(
                metric
                for tasks in self.aggregated_data.values()
                for metrics in tasks.values()
                if isinstance(metrics, dict)
                for metric in metrics.keys()
            )),
        )


def aggregate_metrics(
    results_dir: Path = Path("results"),
    output_json: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    dry_run: bool = False,
) -> AggregationResult:
    """Convenience function to aggregate metrics.
    
    Args:
        results_dir: Directory containing metrics files.
        output_json: JSON output path.
        output_csv: CSV output path.
        dry_run: If True, only show what would be done.
        
    Returns:
        AggregationResult with paths and statistics.
    """
    aggregator = MetricsAggregator(results_dir)
    return aggregator.run(output_json, output_csv, dry_run)
