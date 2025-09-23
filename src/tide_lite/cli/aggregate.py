"""Aggregate metrics from multiple evaluation runs.

This module merges all results/metrics_*.json files and ablation results
into summary.json and summary.csv for easy analysis.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """Aggregates metrics from multiple evaluation runs."""
    
    def __init__(self, results_dir: Path = Path("results")) -> None:
        """Initialize aggregator.
        
        Args:
            results_dir: Directory containing metrics files.
        """
        self.results_dir = Path(results_dir)
    
    def find_metrics_files(self) -> Dict[str, List[Path]]:
        """Find all metrics JSON files.
        
        Returns:
            Dictionary mapping metric types to file paths.
        """
        patterns = {
            "stsb": "metrics_stsb_*.json",
            "quora": "metrics_quora_*.json",
            "temporal": "metrics_temporal_*.json",
            "train": "metrics_train*.json",
            "ablation": "ablation_*/metrics_*.json",
        }
        
        found_files = {}
        for metric_type, pattern in patterns.items():
            files = list(self.results_dir.rglob(pattern))
            if files:
                found_files[metric_type] = files
                logger.info(f"Found {len(files)} {metric_type} metrics files")
        
        return found_files
    
    def load_metrics(self, file_path: Path) -> Dict[str, Any]:
        """Load metrics from JSON file.
        
        Args:
            file_path: Path to JSON file.
            
        Returns:
            Metrics dictionary.
        """
        with open(file_path) as f:
            return json.load(f)
    
    def aggregate(self) -> Dict[str, Any]:
        """Aggregate all metrics into summary.
        
        Returns:
            Aggregated metrics dictionary.
        """
        files = self.find_metrics_files()
        summary = {"models": {}, "ablations": {}, "metadata": {}}
        
        # Process STS-B metrics
        for file_path in files.get("stsb", []):
            metrics = self.load_metrics(file_path)
            model_name = metrics.get("model", file_path.stem.replace("metrics_stsb_", ""))
            
            if model_name not in summary["models"]:
                summary["models"][model_name] = {}
            
            summary["models"][model_name]["stsb"] = {
                "spearman": metrics.get("metrics", {}).get("spearman_rho", 0),
                "pearson": metrics.get("metrics", {}).get("pearson_r", 0),
                "mse": metrics.get("metrics", {}).get("mse", 0),
            }
        
        # Process Quora retrieval metrics
        for file_path in files.get("quora", []):
            metrics = self.load_metrics(file_path)
            model_name = metrics.get("model", file_path.stem.replace("metrics_quora_", ""))
            
            if model_name not in summary["models"]:
                summary["models"][model_name] = {}
            
            summary["models"][model_name]["quora"] = {
                "ndcg_at_10": metrics.get("metrics", {}).get("ndcg_at_10", 0),
                "recall_at_10": metrics.get("metrics", {}).get("recall_at_10", 0),
                "mrr_at_10": metrics.get("metrics", {}).get("mrr_at_10", 0),
                "latency_median_ms": metrics.get("metrics", {}).get("latency_median_ms", 0),
            }
        
        # Process temporal metrics
        for file_path in files.get("temporal", []):
            metrics = self.load_metrics(file_path)
            model_name = metrics.get("model", file_path.stem.replace("metrics_temporal_", ""))
            
            if model_name not in summary["models"]:
                summary["models"][model_name] = {}
            
            summary["models"][model_name]["temporal"] = {
                "accuracy_at_1": metrics.get("metrics", {}).get("temporal_accuracy_at_1", 0),
                "accuracy_at_5": metrics.get("metrics", {}).get("temporal_accuracy_at_5", 0),
                "consistency_score": metrics.get("metrics", {}).get("temporal_consistency_score", 0),
                "time_drift_mae": metrics.get("metrics", {}).get("time_drift_mae_days", 0),
            }
        
        # Process ablation results
        for file_path in files.get("ablation", []):
            metrics = self.load_metrics(file_path)
            ablation_name = file_path.parent.name
            
            summary["ablations"][ablation_name] = {
                "config": metrics.get("config", {}),
                "metrics": metrics.get("metrics", {}),
            }
        
        # Add best model per metric
        summary["best_models"] = self._find_best_models(summary["models"])
        
        # Add metadata
        summary["metadata"]["num_models"] = len(summary["models"])
        summary["metadata"]["num_ablations"] = len(summary["ablations"])
        summary["metadata"]["timestamp"] = pd.Timestamp.now().isoformat()
        
        return summary
    
    def _find_best_models(self, models: Dict[str, Any]) -> Dict[str, str]:
        """Find best model for each metric.
        
        Args:
            models: Dictionary of model metrics.
            
        Returns:
            Dictionary mapping metrics to best model names.
        """
        best = {}
        
        # STS-B: highest Spearman
        best_spearman = 0
        best_spearman_model = None
        for model, metrics in models.items():
            if "stsb" in metrics:
                if metrics["stsb"]["spearman"] > best_spearman:
                    best_spearman = metrics["stsb"]["spearman"]
                    best_spearman_model = model
        if best_spearman_model:
            best["stsb_spearman"] = best_spearman_model
        
        # Quora: highest nDCG@10
        best_ndcg = 0
        best_ndcg_model = None
        for model, metrics in models.items():
            if "quora" in metrics:
                if metrics["quora"]["ndcg_at_10"] > best_ndcg:
                    best_ndcg = metrics["quora"]["ndcg_at_10"]
                    best_ndcg_model = model
        if best_ndcg_model:
            best["quora_ndcg"] = best_ndcg_model
        
        # Temporal: highest consistency
        best_consistency = 0
        best_consistency_model = None
        for model, metrics in models.items():
            if "temporal" in metrics:
                if metrics["temporal"]["consistency_score"] > best_consistency:
                    best_consistency = metrics["temporal"]["consistency_score"]
                    best_consistency_model = model
        if best_consistency_model:
            best["temporal_consistency"] = best_consistency_model
        
        return best
    
    def save_json(self, summary: Dict[str, Any], output_path: Path) -> None:
        """Save summary as JSON.
        
        Args:
            summary: Summary dictionary.
            output_path: Output file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved JSON summary to {output_path}")
    
    def save_csv(self, summary: Dict[str, Any], output_path: Path) -> None:
        """Save summary as CSV.
        
        Args:
            summary: Summary dictionary.
            output_path: Output file path.
        """
        # Convert to flat table
        rows = []
        for model_name, model_metrics in summary["models"].items():
            row = {"model": model_name}
            
            # Flatten nested metrics
            for task, task_metrics in model_metrics.items():
                for metric_name, value in task_metrics.items():
                    row[f"{task}_{metric_name}"] = value
            
            rows.append(row)
        
        # Save as CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved CSV summary to {output_path}")
    
    def run(
        self,
        output_json: Optional[Path] = None,
        output_csv: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run aggregation and save results.
        
        Args:
            output_json: Path for JSON output.
            output_csv: Path for CSV output.
            
        Returns:
            Aggregated summary.
        """
        summary = self.aggregate()
        
        if output_json:
            self.save_json(summary, output_json)
        
        if output_csv:
            self.save_csv(summary, output_csv)
        
        return summary


def main() -> None:
    """Command-line interface for aggregation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate evaluation metrics")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/summary.json"),
        help="Output file path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show plan without executing (default)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually execute aggregation",
    )
    
    args = parser.parse_args()
    
    if not args.run:
        print("[DRY RUN] Would aggregate metrics from:", args.results_dir)
        print("[DRY RUN] Would save to:")
        print(f"  JSON: {args.output}")
        print(f"  CSV: {args.output.with_suffix('.csv')}")
        return
    
    aggregator = MetricsAggregator(args.results_dir)
    summary = aggregator.run(
        output_json=args.output,
        output_csv=args.output.with_suffix(".csv"),
    )
    
    print(f"Aggregated {summary['metadata']['num_models']} models")
    print(f"Best models:")
    for metric, model in summary.get("best_models", {}).items():
        print(f"  {metric}: {model}")


if __name__ == "__main__":
    main()
