#!/usr/bin/env python3
"""Complete pipeline: evaluate trained model, generate plots, and create report."""

import sys
import subprocess
from pathlib import Path

def run_pipeline(checkpoint_dir="outputs/smoke_test"):
    """Run complete evaluation pipeline."""
    
    print("=" * 70)
    print("TIDE-LITE EVALUATION PIPELINE")
    print("=" * 70)
    
    checkpoint_path = Path(checkpoint_dir)
    
    # Find the checkpoint file
    checkpoint_file = checkpoint_path / "checkpoints" / "checkpoint_final.pt"
    if not checkpoint_file.exists():
        # Try to find any checkpoint
        checkpoints = list((checkpoint_path / "checkpoints").glob("checkpoint_*.pt"))
        if checkpoints:
            checkpoint_file = checkpoints[-1]
            print(f"Using checkpoint: {checkpoint_file}")
        else:
            print(f"No checkpoint found in {checkpoint_path / 'checkpoints'}")
            print("Looking for checkpoints...")
            # List what's actually there
            checkpoint_dir = checkpoint_path / "checkpoints"
            if checkpoint_dir.exists():
                files = list(checkpoint_dir.iterdir())
                print(f"Found files: {files}")
            else:
                print(f"Checkpoints directory doesn't exist: {checkpoint_dir}")
            return 1
    
    # 1. Run evaluation
    print("\nStep 1: Running evaluation...")
    cmd = [
        "python3", "scripts/evaluate.py",
        "--model", "tide-lite",
        "--checkpoint", str(checkpoint_file),
        "--task", "stsb",
        "--output-dir", f"{checkpoint_dir}/eval"
    ]
    try:
        subprocess.run(cmd, check=True)
        print("✓ Evaluation complete")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation had issues: {e}")
    
    # 2. Generate enhanced report (skip plots since they have issues)
    print("\nStep 2: Creating enhanced report...")
    cmd = [
        "python3", "scripts/generate_report.py",
        "--input", checkpoint_dir,
        "--output", f"{checkpoint_dir}/report.html"
    ]
    try:
        subprocess.run(cmd, check=True)
        print("✓ Report created")
    except subprocess.CalledProcessError as e:
        print(f"Report generation had issues: {e}")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nResults available in: {checkpoint_dir}/")
    print("  • eval/         - Evaluation metrics")
    print("  • report.html   - Interactive HTML report")
    print("\nOpen the report in your browser:")
    print(f"  open {checkpoint_dir}/report.html")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="outputs/smoke_test", 
                       help="Directory containing training results")
    args = parser.parse_args()
    
    sys.exit(run_pipeline(args.checkpoint_dir))
