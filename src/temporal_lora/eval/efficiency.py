"""Efficiency metrics for parameter count, size, and runtime analysis."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from ..models.lora_model import load_lora_adapter
from ..utils.logging import get_logger

logger = get_logger(__name__)


def count_parameters(model: SentenceTransformer) -> Dict[str, int]:
    """Count total and trainable parameters in a model.
    
    Args:
        model: SentenceTransformer model.
        
    Returns:
        Dictionary with parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
    }


def get_directory_size_mb(directory: Path) -> float:
    """Get total size of a directory in MB.
    
    Args:
        directory: Directory path.
        
    Returns:
        Size in MB.
    """
    if not directory.exists():
        return 0.0
    
    total_size = 0
    for file in directory.rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size
    
    return total_size / (1024 * 1024)  # Convert to MB


def get_model_size_mb(model_path: Path, mode: str) -> float:
    """Get model/adapter size in MB.
    
    Args:
        model_path: Path to model or adapter directory.
        mode: Training mode (lora/full_ft/seq_ft/baseline_frozen).
        
    Returns:
        Size in MB.
    """
    if not model_path.exists():
        logger.warning(f"Model path not found: {model_path}")
        return 0.0
    
    if mode == "lora":
        # LoRA adapter size (typically adapter_model.safetensors or .bin)
        adapter_files = list(model_path.glob("adapter_model.*"))
        if adapter_files:
            return sum(f.stat().st_size for f in adapter_files) / (1024 * 1024)
        else:
            # Fall back to directory size
            return get_directory_size_mb(model_path)
    elif mode in ["full_ft", "seq_ft"]:
        # Full model checkpoint size
        return get_directory_size_mb(model_path)
    elif mode == "baseline_frozen":
        # No additional parameters
        return 0.0
    else:
        return get_directory_size_mb(model_path)


def load_training_metrics(model_path: Path) -> Dict[str, Any]:
    """Load training metrics from logs.
    
    Args:
        model_path: Path to model/adapter directory.
        
    Returns:
        Dictionary with training metrics.
    """
    metrics = {
        "wall_clock_seconds": 0.0,
        "epochs": 0,
        "train_examples": 0,
    }
    
    # Try to load from training log CSV
    log_path = model_path / "training_log.csv"
    if log_path.exists():
        try:
            df = pd.read_csv(log_path)
            if "time" in df.columns:
                metrics["wall_clock_seconds"] = df["time"].max()
            if "epoch" in df.columns:
                metrics["epochs"] = df["epoch"].max()
        except Exception as e:
            logger.warning(f"Failed to load training log: {e}")
    
    return metrics


def compute_efficiency_metrics(
    base_model_name: str,
    mode: str,
    model_path: Optional[Path] = None,
    bucket_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute efficiency metrics for a training mode.
    
    Args:
        base_model_name: Base model identifier.
        mode: Training mode.
        model_path: Path to model/adapter (if applicable).
        bucket_name: Bucket name (for loading adapter).
        
    Returns:
        Dictionary with efficiency metrics.
    """
    logger.info(f"Computing efficiency metrics for mode: {mode}")
    
    metrics = {
        "mode": mode,
        "base_model": base_model_name,
        "bucket": bucket_name or "all",
    }
    
    try:
        # Load model based on mode
        if mode == "baseline_frozen":
            model = SentenceTransformer(base_model_name)
        elif mode == "lora" and model_path and model_path.exists():
            model = load_lora_adapter(base_model_name, model_path)
        elif mode in ["full_ft", "seq_ft"] and model_path and model_path.exists():
            model = SentenceTransformer(str(model_path))
        else:
            logger.warning(f"Cannot load model for mode {mode}, path: {model_path}")
            model = None
        
        # Count parameters
        if model is not None:
            param_counts = count_parameters(model)
            metrics.update(param_counts)
            metrics["trainable_percent"] = param_counts["trainable_ratio"] * 100
        else:
            metrics.update({
                "total_params": 0,
                "trainable_params": 0,
                "trainable_ratio": 0.0,
                "trainable_percent": 0.0,
            })
        
        # Get model/adapter size
        if model_path:
            metrics["size_mb"] = get_model_size_mb(model_path, mode)
            
            # Load training metrics
            train_metrics = load_training_metrics(model_path)
            metrics.update(train_metrics)
        else:
            metrics["size_mb"] = 0.0
            metrics["wall_clock_seconds"] = 0.0
            metrics["epochs"] = 0
            metrics["train_examples"] = 0
        
        # GPU memory (if available)
        if torch.cuda.is_available() and model is not None:
            try:
                # Move to GPU and measure memory
                model.to("cuda")
                torch.cuda.reset_peak_memory_stats()
                
                # Dummy forward pass
                dummy_text = ["This is a test sentence."]
                _ = model.encode(dummy_text, convert_to_numpy=True, show_progress_bar=False)
                
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                metrics["peak_gpu_memory_mb"] = peak_memory
                
                # Clean up
                model.to("cpu")
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to measure GPU memory: {e}")
                metrics["peak_gpu_memory_mb"] = 0.0
        else:
            metrics["peak_gpu_memory_mb"] = 0.0
        
    except Exception as e:
        logger.error(f"Error computing efficiency metrics: {e}")
        metrics.update({
            "total_params": 0,
            "trainable_params": 0,
            "trainable_ratio": 0.0,
            "trainable_percent": 0.0,
            "size_mb": 0.0,
            "wall_clock_seconds": 0.0,
            "epochs": 0,
            "train_examples": 0,
            "peak_gpu_memory_mb": 0.0,
        })
    
    return metrics


def generate_efficiency_summary(
    base_model_name: str,
    modes: List[str],
    adapters_dir: Path,
    full_ft_dir: Path,
    seq_ft_dir: Path,
    buckets: List[str],
    output_path: Path,
) -> pd.DataFrame:
    """Generate efficiency summary table for all modes.
    
    Args:
        base_model_name: Base model identifier.
        modes: List of modes to evaluate.
        adapters_dir: Directory with LoRA adapters.
        full_ft_dir: Directory with full FT checkpoints.
        seq_ft_dir: Directory with sequential FT checkpoints.
        buckets: List of bucket names.
        output_path: Output CSV path.
        
    Returns:
        DataFrame with efficiency metrics.
    """
    logger.info("Generating efficiency summary...")
    
    all_metrics = []
    
    for mode in modes:
        logger.info(f"\nProcessing mode: {mode}")
        
        if mode == "baseline_frozen":
            # Single row for baseline
            metrics = compute_efficiency_metrics(
                base_model_name=base_model_name,
                mode=mode,
            )
            all_metrics.append(metrics)
        
        elif mode == "lora":
            # One row per bucket
            for bucket in buckets:
                model_path = adapters_dir / bucket
                metrics = compute_efficiency_metrics(
                    base_model_name=base_model_name,
                    mode=mode,
                    model_path=model_path,
                    bucket_name=bucket,
                )
                all_metrics.append(metrics)
        
        elif mode == "full_ft":
            # One row per bucket
            for bucket in buckets:
                model_path = full_ft_dir / bucket
                metrics = compute_efficiency_metrics(
                    base_model_name=base_model_name,
                    mode=mode,
                    model_path=model_path,
                    bucket_name=bucket,
                )
                all_metrics.append(metrics)
        
        elif mode == "seq_ft":
            # One row per sequential step
            for i, bucket in enumerate(buckets):
                model_path = seq_ft_dir / f"step_{bucket}"
                metrics = compute_efficiency_metrics(
                    base_model_name=base_model_name,
                    mode=mode,
                    model_path=model_path,
                    bucket_name=f"step{i+1}_{bucket}",
                )
                all_metrics.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Efficiency summary saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EFFICIENCY SUMMARY")
    print("=" * 80)
    
    # Aggregate by mode
    if len(df) > 0:
        summary = df.groupby("mode").agg({
            "trainable_percent": "mean",
            "size_mb": "sum",
            "wall_clock_seconds": "sum",
            "peak_gpu_memory_mb": "max",
        }).round(2)
        
        print("\nAggregated by Mode:")
        print(summary.to_string())
    
    print("\n" + "=" * 80)
    
    return df


def load_efficiency_summary(path: Path) -> pd.DataFrame:
    """Load efficiency summary from CSV.
    
    Args:
        path: Path to efficiency summary CSV.
        
    Returns:
        DataFrame with efficiency metrics.
    """
    if not path.exists():
        raise FileNotFoundError(f"Efficiency summary not found: {path}")
    
    return pd.read_csv(path)
