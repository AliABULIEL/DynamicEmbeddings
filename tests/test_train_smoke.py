#!/usr/bin/env python
"""Test script to verify TIDE-Lite training pipeline works end-to-end.

Usage:
    python tests/test_train_smoke.py
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tide_lite.models.tide_lite import TIDELite, TIDELiteConfig
from src.tide_lite.train.trainer import TIDETrainer, TrainingConfig
from src.tide_lite.data.datasets import DatasetConfig, load_stsb_with_timestamps


def test_smoke():
    """Run minimal training to verify pipeline."""
    print("Running smoke test for TIDE-Lite training pipeline...")
    
    # Create temp directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configuration
        model_config = TIDELiteConfig(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            hidden_dim=384,
            time_encoding_dim=32,
            mlp_hidden_dim=128,
        )
        
        training_config = TrainingConfig(
            output_dir=temp_dir,
            num_epochs=1,
            batch_size=4,
            eval_batch_size=8,
            save_every_n_steps=10,
            eval_every_n_steps=10,
            use_amp=False,  # Disable for CPU testing
            dry_run=False,
        )
        
        # Initialize model
        print("  ✓ Initializing model...")
        model = TIDELite(model_config)
        param_summary = model.get_parameter_summary()
        print(f"    - Extra parameters: {param_summary['extra_params']:,}")
        
        # Initialize trainer
        print("  ✓ Initializing trainer...")
        trainer = TIDETrainer(model, training_config)
        
        # Run training (just a few steps)
        print("  ✓ Running training...")
        metrics = trainer.train()
        
        # Check outputs exist
        output_path = Path(temp_dir)
        assert (output_path / "config_used.json").exists(), "Config not saved"
        assert (output_path / "metrics_train.json").exists(), "Metrics not saved"
        assert len(list((output_path / "checkpoints").glob("*.pt"))) > 0, "No checkpoints saved"
        
        print(f"  ✓ Training complete! Final Spearman: {metrics['final_val_spearman']:.4f}")
    
    print("\n✅ Smoke test passed!")
    return 0


if __name__ == "__main__":
    sys.exit(test_smoke())
