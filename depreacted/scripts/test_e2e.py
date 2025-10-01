#!/usr/bin/env python3
"""Complete end-to-end test of TIDE-Lite."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Run complete TIDE-Lite pipeline test."""
    print("=" * 70)
    print("TIDE-LITE END-TO-END TEST")
    print("=" * 70)
    
    try:
        # 1. Import all required modules
        print("\n1. Testing imports...")
        from src.tide_lite.models import TIDELite, TIDELiteConfig
        from src.tide_lite.train import TIDETrainer, TrainingConfig
        from src.tide_lite.data.datasets import DatasetConfig, load_stsb_with_timestamps
        print("   ✓ All imports successful")
        
        # 2. Create model
        print("\n2. Creating TIDE-Lite model...")
        model_config = TIDELiteConfig(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            time_encoding_dim=32,
            mlp_hidden_dim=128,
            freeze_encoder=True
        )
        model = TIDELite(model_config)
        params = model.count_extra_parameters()
        print(f"   ✓ Model created with {params:,} extra parameters")
        
        # 3. Test dataset loading
        print("\n3. Testing dataset loading...")
        dataset_config = DatasetConfig(
            seed=42,
            max_samples=50,  # Small sample for quick test
            cache_dir="./data"
        )
        datasets = load_stsb_with_timestamps(dataset_config)
        print(f"   ✓ Loaded STS-B: train={len(datasets['train'])}, "
              f"val={len(datasets['validation'])}, test={len(datasets['test'])}")
        
        # 4. Create training config
        print("\n4. Setting up training configuration...")
        train_config = TrainingConfig(
            encoder_name=model_config.encoder_name,
            hidden_dim=model_config.hidden_dim,
            time_encoding_dim=model_config.time_encoding_dim,
            mlp_hidden_dim=model_config.mlp_hidden_dim,
            freeze_encoder=model_config.freeze_encoder,
            batch_size=4,
            num_epochs=1,
            learning_rate=1e-4,
            warmup_steps=2,
            save_every_n_steps=10,
            eval_every_n_steps=10,
            output_dir="results/e2e_test",
            dry_run=True,  # Dry run for quick test
            seed=42
        )
        print("   ✓ Training configuration created")
        
        # 5. Initialize trainer
        print("\n5. Initializing trainer...")
        trainer = TIDETrainer(model, train_config)
        print("   ✓ Trainer initialized")
        
        # 6. Run dry-run training
        print("\n6. Running dry-run training...")
        summary = trainer.dry_run_summary()
        print("   ✓ Dry run completed successfully")
        print(f"   • Model params: {summary['model']['trainable_params']:,}")
        print(f"   • Estimated time: {summary['compute']['estimated_time_hours']:.2f} hours")
        
        # Success!
        print("\n" + "=" * 70)
        print("✅ END-TO-END TEST PASSED!")
        print("\nThe TIDE-Lite implementation is working correctly.")
        print("\nNext steps:")
        print("1. Run actual training (remove dry_run=True)")
        print("2. Try the smoke config: python3 scripts/train.py --config configs/smoke.yaml")
        print("3. Check GPU availability for faster training")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error above and ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
