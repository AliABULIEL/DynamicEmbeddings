#!/usr/bin/env python3
"""Simple evaluation script that works with checkpoint files directly."""

import sys
from pathlib import Path
import torch
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tide_lite.models import TIDELite, TIDELiteConfig
from src.tide_lite.eval.eval_stsb import evaluate_stsb

def evaluate_checkpoint(checkpoint_path, output_dir=None):
    """Evaluate a TIDE-Lite checkpoint on STS-B."""
    
    print("=" * 70)
    print("TIDE-LITE CHECKPOINT EVALUATION")
    print("=" * 70)
    
    checkpoint_path = Path(checkpoint_path)
    
    # Setup output directory
    if output_dir is None:
        output_dir = checkpoint_path.parent.parent / "eval"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    
    # Load checkpoint
    print("\n📊 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract config from checkpoint
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
    else:
        # Use default config
        config_dict = {
            "encoder_name": "sentence-transformers/all-MiniLM-L6-v2",
            "hidden_dim": 384,
            "time_encoding_dim": 32,
            "mlp_hidden_dim": 128,
            "mlp_dropout": 0.1,
            "gate_activation": "sigmoid",
            "freeze_encoder": True,
            "pooling_strategy": "mean"
        }
    
    # Create model config
    model_config = TIDELiteConfig(
        encoder_name=config_dict.get("encoder_name", "sentence-transformers/all-MiniLM-L6-v2"),
        hidden_dim=config_dict.get("hidden_dim", 384),
        time_encoding_dim=config_dict.get("time_encoding_dim", 32),
        mlp_hidden_dim=config_dict.get("mlp_hidden_dim", 128),
        mlp_dropout=config_dict.get("mlp_dropout", 0.1),
        gate_activation=config_dict.get("gate_activation", "sigmoid"),
        freeze_encoder=config_dict.get("freeze_encoder", True),
        pooling_strategy=config_dict.get("pooling_strategy", "mean")
    )
    
    # Initialize model
    print("📊 Initializing model...")
    model = TIDELite(model_config)
    
    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "temporal_gate_state_dict" in checkpoint:
        # Load only temporal gate if that's what was saved
        model.temporal_gate.load_state_dict(checkpoint["temporal_gate_state_dict"])
    else:
        print("⚠️  Warning: No model weights found in checkpoint")
    
    print(f"   Model loaded with {model.count_extra_parameters():,} extra parameters")
    
    # Evaluate on STS-B
    print("\n📊 Evaluating on STS-B...")
    from src.tide_lite.data.datasets import DatasetConfig, load_stsb_with_timestamps
    
    dataset_config = DatasetConfig(
        seed=42,
        cache_dir="./data"
    )
    
    datasets = load_stsb_with_timestamps(dataset_config)
    
    # Run evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    results = evaluate_stsb(model, datasets["validation"], device=device, batch_size=64)
    
    print(f"\n📊 Results:")
    print(f"   Spearman: {results['spearman']:.4f}")
    print(f"   Pearson:  {results['pearson']:.4f}")
    print(f"   MSE:      {results['mse']:.4f}")
    
    # Save results
    results_path = output_dir / "stsb_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Also extract training metrics if available
    if "metrics" in checkpoint:
        metrics_path = output_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(checkpoint["metrics"], f, indent=2)
        print(f"✓ Training metrics saved to: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate TIDE-Lite checkpoint")
    parser.add_argument("--checkpoint", default="outputs/smoke_test/checkpoints/checkpoint_final.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--output-dir", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Find checkpoint if needed
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Try to find in standard location
        base_dir = Path("outputs/smoke_test/checkpoints")
        if base_dir.exists():
            checkpoints = list(base_dir.glob("checkpoint_*.pt"))
            if checkpoints:
                checkpoint_path = checkpoints[-1]
                print(f"Using checkpoint: {checkpoint_path}")
            else:
                print(f"❌ No checkpoints found in {base_dir}")
                return 1
        else:
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            return 1
    
    evaluate_checkpoint(checkpoint_path, args.output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
