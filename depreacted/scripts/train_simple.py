#!/usr/bin/env python3
"""Simple training script for TIDE-Lite."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tide_lite.models import TIDELite, TIDELiteConfig
from src.tide_lite.train import TIDETrainer, TrainingConfig

def main():
    # Create model configuration
    model_config = TIDELiteConfig(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        time_encoding_dim=32,
        mlp_hidden_dim=128,
        mlp_dropout=0.1,
        freeze_encoder=True,
        pooling_strategy="mean",
        gate_activation="sigmoid"
    )
    
    # Create training configuration
    train_config = TrainingConfig(
        # Model params (must match for compatibility)
        encoder_name=model_config.encoder_name,
        hidden_dim=model_config.hidden_dim,
        time_encoding_dim=model_config.time_encoding_dim,
        mlp_hidden_dim=model_config.mlp_hidden_dim,
        mlp_dropout=model_config.mlp_dropout,
        freeze_encoder=model_config.freeze_encoder,
        pooling_strategy=model_config.pooling_strategy,
        gate_activation=model_config.gate_activation,
        
        # Training params
        batch_size=8,
        eval_batch_size=16,
        num_epochs=1,
        learning_rate=1e-4,
        warmup_steps=10,
        
        # Loss weights
        temporal_weight=0.1,
        preservation_weight=0.05,
        
        # Output
        output_dir="results/simple_run",
        
        # Misc
        use_amp=False,  # Disable for CPU
        dry_run=True,  # Enable dry run for testing
        seed=42
    )
    
    print("=" * 60)
    print("TIDE-LITE SIMPLE TRAINING")
    print("=" * 60)
    
    # Initialize model
    print("\n📊 Initializing model...")
    model = TIDELite(model_config)
    param_summary = model.get_parameter_summary()
    print(f"   Total params: {param_summary['total_params']:,}")
    print(f"   Extra params: {param_summary['extra_params']:,}")
    
    # Initialize trainer
    print("\n🔄 Setting up trainer...")
    trainer = TIDETrainer(model, train_config)
    
    # Run training
    if train_config.dry_run:
        print("\n🧪 Running dry run (no actual training)...")
        summary = trainer.dry_run_summary()
        print("\n✅ Dry run complete!")
        print(f"   Check {train_config.output_dir}/dry_run_summary.json")
    else:
        print("\n🚀 Starting training...")
        metrics = trainer.train()
        print(f"\n✅ Training complete!")
        print(f"   Final Spearman: {metrics.get('final_val_spearman', 'N/A')}")
    
    print("\n" + "=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
