#!/usr/bin/env python3
"""Evaluation script for TIDE-Lite models."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
# Add numpy arrays to safe globals for PyTorch 2.6+
try:
    # Try new numpy API first
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
except (AttributeError, ImportError):
    # Fall back to old API for older numpy versions
    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except:
        pass  # Ignore if both fail
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tide_lite.models import TIDELite, TIDELiteConfig
from src.tide_lite.data.datasets import DatasetConfig, load_stsb_with_timestamps, load_quora
from src.tide_lite.data.collate import STSBCollator, TextBatcher

# Try to import FAISS for retrieval
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_stsb(model, dataset, device, batch_size=64):
    """Evaluate on STS-B dataset."""
    # Create TextBatcher instance with proper configuration
    text_batcher = TextBatcher(
        model_name=model.config.encoder_name,
        max_length=128,
        padding="max_length",
        truncation=True
    )
    collator = STSBCollator(text_batcher, include_timestamps=True)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collator,
        shuffle=False
    )
    
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Extract tokenized inputs from BatchEncoding objects
            sent1_inputs = batch["sentence1_inputs"]
            sent2_inputs = batch["sentence2_inputs"]
            
            # Move batch to device
            input_ids1 = sent1_inputs["input_ids"].to(device)
            attention_mask1 = sent1_inputs["attention_mask"].to(device)
            timestamps1 = batch["timestamps1"].to(device)
            
            input_ids2 = sent2_inputs["input_ids"].to(device)
            attention_mask2 = sent2_inputs["attention_mask"].to(device)
            timestamps2 = batch["timestamps2"].to(device)
            
            # Get embeddings
            emb1, _ = model(input_ids1, attention_mask1, timestamps1)
            emb2, _ = model(input_ids2, attention_mask2, timestamps2)
            
            # Compute cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2)
            
            predictions.extend(cos_sim.cpu().numpy())
            labels.extend(batch["labels"].numpy() / 5.0)  # Normalize to [0, 1]
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Compute metrics
    spearman_corr, _ = spearmanr(predictions, labels)
    pearson_corr, _ = pearsonr(predictions, labels)
    mse = np.mean((predictions - labels) ** 2)
    
    return {
        "spearman": spearman_corr,
        "pearson": pearson_corr,
        "mse": mse
    }


def load_checkpoint_model(checkpoint_path):
    """Load model from checkpoint file."""
    # Handle PyTorch 2.6+ weights_only issue
    # Fix for PyTorch 2.6+: explicitly set weights_only=False for numpy arrays
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # First, try to extract dimensions from the actual checkpoint weights
    # This ensures compatibility even if config is missing or incorrect
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "temporal_gate_state_dict" in checkpoint:
        state_dict = checkpoint["temporal_gate_state_dict"]
    else:
        state_dict = checkpoint
    
    # Infer dimensions from checkpoint weights
    time_encoding_dim = None
    mlp_hidden_dim = None
    hidden_dim = 384  # Default
    
    for key, value in state_dict.items():
        if "time_encoder.scales" in key or "scales" in key:
            time_encoding_dim = value.shape[0]
        if "temporal_gate.mlp.0.weight" in key or "mlp.0.weight" in key:
            mlp_hidden_dim = value.shape[0]
            time_encoding_dim = time_encoding_dim or value.shape[1]
        if "temporal_gate.mlp.3.weight" in key or "mlp.3.weight" in key or "temporal_gate.mlp.2.weight" in key:
            hidden_dim = value.shape[0]
    
    # Use inferred dimensions or defaults
    if time_encoding_dim is None:
        time_encoding_dim = 64 if "config" in checkpoint and checkpoint.get("config", {}).get("time_encoding_dim") == 64 else 32
    if mlp_hidden_dim is None:
        mlp_hidden_dim = 256 if "config" in checkpoint and checkpoint.get("config", {}).get("mlp_hidden_dim") == 256 else 128
    
    # Build config with correct dimensions
    config_dict = {
        "encoder_name": "sentence-transformers/all-MiniLM-L6-v2",
        "hidden_dim": hidden_dim,
        "time_encoding_dim": time_encoding_dim,
        "mlp_hidden_dim": mlp_hidden_dim,
        "mlp_dropout": 0.1,
        "gate_activation": "sigmoid",
        "freeze_encoder": True,
        "pooling_strategy": "mean"
    }
    
    # Override with checkpoint config if available
    if "config" in checkpoint:
        for key, value in checkpoint["config"].items():
            if key in config_dict:
                config_dict[key] = value
    
    # Create model config
    model_config = TIDELiteConfig(**{k: v for k, v in config_dict.items() if k in TIDELiteConfig.__dataclass_fields__})
    
    # Initialize model with correct architecture
    model = TIDELite(model_config)
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "temporal_gate_state_dict" in checkpoint:
        # Only load temporal gate weights
        model.temporal_gate.load_state_dict(checkpoint["temporal_gate_state_dict"])
    else:
        # Try loading as full state dict
        model.load_state_dict(checkpoint)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate TIDE-Lite models")
    parser.add_argument("--model", choices=["tide-lite", "baseline"], default="tide-lite")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--task", choices=["stsb", "quora", "all"], default="stsb")
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    # Check FAISS
    if HAS_FAISS:
        logger.info("FAISS-CPU detected")
    else:
        logger.info("FAISS not found - retrieval tasks will be skipped")
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    if args.model == "tide-lite":
        if not args.checkpoint:
            logger.error("--checkpoint required for TIDE-Lite evaluation")
            return 1
        
        # Handle checkpoint file directly
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.is_file() and checkpoint_path.suffix == ".pt":
            model = load_checkpoint_model(checkpoint_path)
        else:
            # Try loading as directory (old format)
            model = TIDELite.from_pretrained(args.checkpoint)
    else:
        # Baseline model
        from src.tide_lite.models.baselines import load_minilm_baseline
        model = load_minilm_baseline(args.encoder)
    
    model = model.to(device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Evaluate on STS-B
    if args.task in ["stsb", "all"]:
        logger.info("Evaluating on STS-B...")
        dataset_config = DatasetConfig(
            seed=42,
            max_samples=args.max_samples,
            cache_dir="./data"
        )
        datasets = load_stsb_with_timestamps(dataset_config)
        
        stsb_results = evaluate_stsb(model, datasets["validation"], device)
        results["stsb"] = stsb_results
        
        logger.info(f"STS-B Results:")
        logger.info(f"  Spearman: {stsb_results['spearman']:.4f}")
        logger.info(f"  Pearson: {stsb_results['pearson']:.4f}")
        logger.info(f"  MSE: {stsb_results['mse']:.4f}")
    
    # Save results (convert numpy types to Python native types for JSON)
    results_path = output_dir / "eval_results.json"
    
    # Convert numpy types to native Python types
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_json = convert_to_native(results)
    
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
