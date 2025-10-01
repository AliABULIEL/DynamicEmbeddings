#!/usr/bin/env python
"""Monitor training progress with detailed loss breakdowns.

This script helps visualize training and validation losses
including all components (task, temporal, preservation).
"""

import json
from pathlib import Path
import argparse
import time
import sys

def monitor_metrics(results_dir: Path, refresh_interval: int = 5):
    """Monitor training metrics from saved JSON files.
    
    Args:
        results_dir: Directory containing metrics files
        refresh_interval: Seconds between refreshes
    """
    metrics_file = results_dir / "metrics_train.json"
    
    print("\n" + "="*60)
    print("TIDE-LITE TRAINING MONITOR")
    print("="*60)
    print(f"Monitoring: {metrics_file}")
    print(f"Refresh every {refresh_interval} seconds (Ctrl+C to stop)\n")
    
    try:
        while True:
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Clear screen (works on Unix/Linux/Mac)
                print("\033[2J\033[H")
                
                print("="*60)
                print("TRAINING PROGRESS")
                print("="*60)
                
                if metrics.get("train_loss"):
                    n_epochs = len(metrics["train_loss"])
                    print(f"\nEpochs completed: {n_epochs}")
                    
                    # Show last 5 epochs
                    print("\nRecent epochs:")
                    print("-"*50)
                    print("Epoch | Train Loss | Val Loss | Val Spearman | LR")
                    print("-"*50)
                    
                    start_idx = max(0, n_epochs - 5)
                    for i in range(start_idx, n_epochs):
                        train_loss = metrics["train_loss"][i]
                        val_loss = metrics["val_loss"][i] if i < len(metrics["val_loss"]) else 0
                        val_spear = metrics["val_spearman"][i] if i < len(metrics["val_spearman"]) else 0
                        lr = metrics["learning_rate"][i] if i < len(metrics["learning_rate"]) else 0
                        
                        print(f"{i+1:5d} | {train_loss:10.4f} | {val_loss:8.4f} | {val_spear:12.4f} | {lr:.2e}")
                    
                    # Show best validation
                    if metrics.get("val_spearman"):
                        best_spearman = max(metrics["val_spearman"])
                        best_epoch = metrics["val_spearman"].index(best_spearman) + 1
                        print(f"\n✨ Best validation Spearman: {best_spearman:.4f} (Epoch {best_epoch})")
                    
                    # Training time
                    if metrics.get("epoch_times"):
                        total_time = sum(metrics["epoch_times"])
                        avg_time = total_time / len(metrics["epoch_times"])
                        print(f"\n⏱  Total time: {total_time/60:.1f} min")
                        print(f"   Avg per epoch: {avg_time:.1f} sec")
                
            else:
                print("Waiting for training to start...")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Monitor TIDE-Lite training")
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Results directory to monitor"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Refresh interval in seconds"
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Error: Directory {args.results_dir} does not exist")
        return 1
    
    return monitor_metrics(args.results_dir, args.interval)

if __name__ == "__main__":
    sys.exit(main())
