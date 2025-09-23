"""Result aggregation module for TIDE-Lite.

This module aggregates metrics from multiple evaluation runs
into summary JSON and CSV files for analysis and reporting.
"""

import csv
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelResults:
    """Container for all results from a single model."""
    model_name: str
    stsb: Optional[Dict[str, float]] = None
    quora: Optional[Dict[str, float]] = None
    temporal: Optional[Dict[str, float]] = None
    training: Optional[Dict[str, float]] = None
    extra_params: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "model_name": self.model_name,
            "stsb": self.stsb or {},
            "quora": self.quora or {},
            "temporal": self.temporal or {},
            "training": self.training or {},
            "extra_params": self.extra_params,
        }


class ResultAggregator:
    """Aggregates evaluation results from multiple models."""
    
    def __init__(self, results_dir: Union[str, Path]) -> None:
        """Initialize aggregator.
        
        Args:
            results_dir: Directory containing result files.
        """
        self.results_dir = Path(results_dir)
        self.models: Dict[str, ModelResults] = {}
        
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")
    
    def scan_results(self) -> None:
        """Scan results directory for metrics files."""
        logger.info(f"Scanning {self.results_dir} for metrics files")
        
        # Find all metrics files
        stsb_files = list(self.results_dir.glob("**/metrics_stsb_*.json"))
        quora_files = list(self.results_dir.glob("**/metrics_quora_*.json"))
        temporal_files = list(self.results_dir.glob("**/metrics_temporal_*.json"))
        train_files = list(self.results_dir.glob("**/metrics_train.json"))
        
        logger.info(f"Found {len(stsb_files)} STS-B results")
        logger.info(f"Found {len(quora_files)} Quora results")
        logger.info(f"Found {len(temporal_files)} Temporal results")
        logger.info(f"Found {len(train_files)} Training results")
        
        # Process STS-B results
        for file in stsb_files:
            self._process_stsb_file(file)
        
        # Process Quora results
        for file in quora_files:
            self._process_quora_file(file)
        
        # Process Temporal results
        for file in temporal_files:
            self._process_temporal_file(file)
        
        # Process Training results
        for file in train_files:
            self._process_train_file(file)
    
    def _extract_model_name(self, filename: str) -> str:
        """Extract model name from filename."""
        # Remove metrics_ prefix and .json suffix
        name = filename.replace("metrics_", "").replace(".json", "")
        # Remove task prefix
        for prefix in ["stsb_", "quora_", "temporal_"]:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name
    
    def _process_stsb_file(self, file: Path) -> None:
        """Process STS-B metrics file."""
        try:
            with open(file) as f:
                data = json.load(f)
            
            model_name = self._extract_model_name(file.name)
            
            if model_name not in self.models:
                self.models[model_name] = ModelResults(model_name)
            
            self.models[model_name].stsb = {
                "spearman": data.get("spearman_correlation", 0.0),
                "pearson": data.get("pearson_correlation", 0.0),
                "mse": data.get("mse", 0.0),
            }
            
            logger.debug(f"Loaded STS-B results for {model_name}")
        except Exception as e:
            logger.warning(f"Failed to process {file}: {e}")
    
    def _process_quora_file(self, file: Path) -> None:
        """Process Quora retrieval metrics file."""
        try:
            with open(file) as f:
                data = json.load(f)
            
            model_name = self._extract_model_name(file.name)
            
            if model_name not in self.models:
                self.models[model_name] = ModelResults(model_name)
            
            metrics = data.get("metrics", data)
            self.models[model_name].quora = {
                "ndcg_at_10": metrics.get("ndcg_at_10", 0.0),
                "recall_at_10": metrics.get("recall_at_10", 0.0),
                "mrr_at_10": metrics.get("mrr_at_10", 0.0),
                "map_at_10": metrics.get("map_at_10", 0.0),
                "latency_median_ms": metrics.get("latency_median_ms", 0.0),
            }
            
            logger.debug(f"Loaded Quora results for {model_name}")
        except Exception as e:
            logger.warning(f"Failed to process {file}: {e}")
    
    def _process_temporal_file(self, file: Path) -> None:
        """Process temporal evaluation metrics file."""
        try:
            with open(file) as f:
                data = json.load(f)
            
            model_name = self._extract_model_name(file.name)
            
            if model_name not in self.models:
                self.models[model_name] = ModelResults(model_name)
            
            metrics = data.get("metrics", data)
            self.models[model_name].temporal = {
                "accuracy_at_1": metrics.get("temporal_accuracy_at_1", 0.0),
                "accuracy_at_5": metrics.get("temporal_accuracy_at_5", 0.0),
                "consistency_score": metrics.get("temporal_consistency_score", 0.0),
                "time_drift_mae": metrics.get("time_drift_mae_days", 0.0),
            }
            
            logger.debug(f"Loaded Temporal results for {model_name}")
        except Exception as e:
            logger.warning(f"Failed to process {file}: {e}")
    
    def _process_train_file(self, file: Path) -> None:
        """Process training metrics file."""
        try:
            with open(file) as f:
                data = json.load(f)
            
            # Extract model name from parent directory
            model_name = file.parent.parent.name
            
            if model_name not in self.models:
                self.models[model_name] = ModelResults(model_name)
            
            self.models[model_name].training = {
                "final_loss": data.get("final_loss", 0.0),
                "best_val_score": data.get("best_val_score", 0.0),
                "total_epochs": data.get("num_epochs", 0),
                "training_time": data.get("total_time", 0.0),
            }
            
            # Extract extra parameters if available
            if "model_info" in data:
                self.models[model_name].extra_params = data["model_info"].get("trainable_params", 0)
            
            logger.debug(f"Loaded Training results for {model_name}")
        except Exception as e:
            logger.warning(f"Failed to process {file}: {e}")
    
    def compute_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics across all models."""
        summary = {
            "best_models": {},
            "averages": {},
            "improvements": {},
        }
        
        # Find best models for each metric
        if any(m.stsb for m in self.models.values()):
            stsb_scores = {
                name: model.stsb["spearman"] 
                for name, model in self.models.items() 
                if model.stsb
            }
            if stsb_scores:
                summary["best_models"]["stsb_spearman"] = max(stsb_scores, key=stsb_scores.get)
        
        if any(m.quora for m in self.models.values()):
            quora_scores = {
                name: model.quora["ndcg_at_10"] 
                for name, model in self.models.items() 
                if model.quora
            }
            if quora_scores:
                summary["best_models"]["quora_ndcg"] = max(quora_scores, key=quora_scores.get)
        
        if any(m.temporal for m in self.models.values()):
            temporal_scores = {
                name: model.temporal["consistency_score"] 
                for name, model in self.models.items() 
                if model.temporal
            }
            if temporal_scores:
                summary["best_models"]["temporal_consistency"] = max(temporal_scores, key=temporal_scores.get)
        
        # Calculate improvements (TIDE-Lite vs baselines)
        if "tide_lite" in self.models and "minilm" in self.models:
            tide = self.models["tide_lite"]
            baseline = self.models["minilm"]
            
            if tide.stsb and baseline.stsb:
                summary["improvements"]["stsb_spearman"] = (
                    tide.stsb["spearman"] - baseline.stsb["spearman"]
                )
            
            if tide.quora and baseline.quora:
                summary["improvements"]["quora_ndcg"] = (
                    tide.quora["ndcg_at_10"] - baseline.quora["ndcg_at_10"]
                )
            
            if tide.temporal and baseline.temporal:
                summary["improvements"]["temporal_consistency"] = (
                    tide.temporal["consistency_score"] - baseline.temporal["consistency_score"]
                )
        
        return summary
    
    def save_json(self, output_file: Union[str, Path]) -> None:
        """Save aggregated results to JSON file.
        
        Args:
            output_file: Output JSON file path.
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            "models": {name: model.to_dict() for name, model in self.models.items()},
            "summary": self.compute_summary_stats(),
        }
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved aggregated results to {output_file}")
    
    def save_csv(self, output_file: Union[str, Path]) -> None:
        """Save aggregated results to CSV file.
        
        Args:
            output_file: Output CSV file path.
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for name, model in self.models.items():
            row = {"model": name}
            
            if model.stsb:
                row.update({f"stsb_{k}": v for k, v in model.stsb.items()})
            
            if model.quora:
                row.update({f"quora_{k}": v for k, v in model.quora.items()})
            
            if model.temporal:
                row.update({f"temporal_{k}": v for k, v in model.temporal.items()})
            
            if model.training:
                row.update({f"train_{k}": v for k, v in model.training.items()})
            
            row["extra_params"] = model.extra_params
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
            logger.info(f"Saved CSV results to {output_file}")
        else:
            logger.warning("No results to save to CSV")
    
    def print_summary(self) -> None:
        """Print summary of aggregated results."""
        print("\n" + "=" * 70)
        print("AGGREGATED RESULTS SUMMARY")
        print("=" * 70)
        
        for name, model in self.models.items():
            print(f"\n📊 {name}")
            
            if model.stsb:
                print(f"  STS-B: Spearman={model.stsb['spearman']:.4f}")
            
            if model.quora:
                print(f"  Quora: nDCG@10={model.quora['ndcg_at_10']:.4f}")
            
            if model.temporal:
                print(f"  Temporal: Consistency={model.temporal['consistency_score']:.4f}")
            
            if model.extra_params:
                print(f"  Extra Params: {model.extra_params:,}")
        
        summary = self.compute_summary_stats()
        
        if summary["best_models"]:
            print("\n🏆 Best Models:")
            for metric, model in summary["best_models"].items():
                print(f"  {metric}: {model}")
        
        if summary["improvements"]:
            print("\n📈 TIDE-Lite Improvements:")
            for metric, improvement in summary["improvements"].items():
                print(f"  {metric}: {improvement:+.4f}")
        
        print("\n" + "=" * 70)


def aggregate_results(
    results_dir: Union[str, Path],
    output_json: Optional[Union[str, Path]] = None,
    output_csv: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Aggregate evaluation results from a directory.
    
    Args:
        results_dir: Directory containing result files.
        output_json: Output JSON file path.
        output_csv: Output CSV file path.
        dry_run: If True, only show plan without saving.
        
    Returns:
        Aggregated results dictionary.
    """
    if dry_run:
        logger.info("[DRY RUN] Would aggregate results from:")
        logger.info(f"  Directory: {results_dir}")
        if output_json:
            logger.info(f"  JSON output: {output_json}")
        if output_csv:
            logger.info(f"  CSV output: {output_csv}")
        return {}
    
    aggregator = ResultAggregator(results_dir)
    aggregator.scan_results()
    aggregator.print_summary()
    
    if output_json:
        aggregator.save_json(output_json)
    
    if output_csv:
        aggregator.save_csv(output_csv)
    
    return {
        "models": {name: model.to_dict() for name, model in aggregator.models.items()},
        "summary": aggregator.compute_summary_stats(),
    }
