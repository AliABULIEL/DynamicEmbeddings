#!/usr/bin/env python
"""Diagnose loss imbalance issues in TIDE-Lite training.

This script helps identify why training loss might be higher than validation loss
by analyzing the individual loss components.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.tide_lite.models.tide_lite import TIDELite, TIDELiteConfig
from src.tide_lite.data.datasets import load_stsb_with_timestamps, DatasetConfig
from src.tide_lite.data.collate import STSBCollator, TextBatcher
from src.tide_lite.train.losses import combined_tide_loss, cosine_regression_loss


def analyze_loss_components(
    model_path: str = None,
    num_batches: int = 10,
    device: str = None
):
    """Analyze loss components on training and validation data.
    
    Args:
        model_path: Path to saved model checkpoint (optional)
        num_batches: Number of batches to analyze
        device: Device to use (auto-detect if None)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*60)
    print("TIDE-LITE LOSS COMPONENT ANALYSIS")
    print("="*60)
    
    # Initialize model
    print("\nInitializing model...")
    config = TIDELiteConfig(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        mlp_hidden_dim=128,
        freeze_encoder=True
    )
    model = TIDELite(config).to(device)
    
    # Load checkpoint if provided
    if model_path and Path(model_path).exists():
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    
    # Load datasets
    print("\nLoading datasets...")
    dataset_config = DatasetConfig(seed=42)
    datasets = load_stsb_with_timestamps(dataset_config)
    
    # Create dataloaders
    tokenizer = TextBatcher(
        model_name=config.encoder_name,
        max_length=128
    )
    collator = STSBCollator(tokenizer, include_timestamps=True)
    
    train_loader = DataLoader(
        datasets["train"],
        batch_size=32,
        shuffle=False,
        collate_fn=collator,
        num_workers=2
    )
    
    val_loader = DataLoader(
        datasets["validation"],
        batch_size=32,
        shuffle=False,
        collate_fn=collator,
        num_workers=2
    )
    
    # Analyze both splits
    for split_name, loader in [("Training", train_loader), ("Validation", val_loader)]:
        print(f"\n\n{split_name} Set Analysis")
        print("-"*40)
        
        task_losses = []
        temporal_losses = []
        preservation_losses = []
        total_losses = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader, desc=f"Analyzing {split_name}")):
                if i >= num_batches:
                    break
                
                # Process batch
                sent1_inputs = {k: v.to(device) for k, v in batch["sentence1_inputs"].items()}
                sent2_inputs = {k: v.to(device) for k, v in batch["sentence2_inputs"].items()}
                labels = batch["labels"].to(device) / 5.0
                timestamps1 = batch["timestamps1"].to(device)
                timestamps2 = batch["timestamps2"].to(device)
                
                # Forward pass
                temporal_emb1, base_emb1 = model(
                    sent1_inputs["input_ids"],
                    sent1_inputs["attention_mask"],
                    timestamps1
                )
                temporal_emb2, base_emb2 = model(
                    sent2_inputs["input_ids"],
                    sent2_inputs["attention_mask"],
                    timestamps2
                )
                
                # Concatenate for loss computation
                temporal_emb = torch.cat([temporal_emb1, temporal_emb2], dim=0)
                base_emb = torch.cat([base_emb1, base_emb2], dim=0)
                timestamps = torch.cat([timestamps1, timestamps2], dim=0)
                
                # Compute losses with different weight configurations
                for alpha, beta in [(0.1, 0.05), (0.01, 0.005), (0.001, 0.001)]:
                    loss, components = combined_tide_loss(
                        temporal_emb,
                        base_emb,
                        timestamps,
                        target_scores=labels,
                        alpha=alpha,
                        beta=beta,
                        tau_seconds=86400.0
                    )
                    
                    if alpha == 0.01 and beta == 0.005:  # Track main configuration
                        task_losses.append(components["task_loss"])
                        temporal_losses.append(components["temporal_loss"])
                        preservation_losses.append(components["preservation_loss"])
                        total_losses.append(loss.item())
        
        # Print statistics
        print(f"\nLoss Component Statistics ({split_name}):")
        print(f"  Task Loss:         {np.mean(task_losses):.4f} ± {np.std(task_losses):.4f}")
        print(f"  Temporal Loss:     {np.mean(temporal_losses):.4f} ± {np.std(temporal_losses):.4f}")
        print(f"  Preservation Loss: {np.mean(preservation_losses):.4f} ± {np.std(preservation_losses):.4f}")
        print(f"  Total Loss:        {np.mean(total_losses):.4f} ± {np.std(total_losses):.4f}")
        
        # Analyze contribution percentages
        print(f"\nLoss Contributions (with α=0.01, β=0.005):")
        task_contrib = np.mean(task_losses)
        temporal_contrib = 0.01 * np.mean(temporal_losses)
        preservation_contrib = 0.005 * np.mean(preservation_losses)
        total = task_contrib + temporal_contrib + preservation_contrib
        
        print(f"  Task:         {task_contrib:.4f} ({100*task_contrib/total:.1f}%)")
        print(f"  Temporal:     {temporal_contrib:.4f} ({100*temporal_contrib/total:.1f}%)")
        print(f"  Preservation: {preservation_contrib:.4f} ({100*preservation_contrib/total:.1f}%)")
    
    print("\n" + "="*60)
    print("\nDIAGNOSIS SUMMARY:")
    print("-"*60)
    print("""
If training loss >> validation loss, check:
1. Temporal/preservation weights might be too high
2. Temporal loss might be computed differently between train/val
3. Batch normalization or dropout effects (if any)

Recommended fixes:
1. Ensure validation uses same loss computation as training
2. Reduce temporal_weight to 0.01 or lower
3. Reduce preservation_weight to 0.005 or lower
4. Monitor individual loss components during training
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose TIDE-Lite loss issues")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of batches to analyze"
    )
    
    args = parser.parse_args()
    analyze_loss_components(args.model, args.num_batches)
