#!/usr/bin/env python3
"""Quick test script for TIDE-Lite training."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tide_lite.models import TIDELite, TIDELiteConfig
from src.tide_lite.train import TIDETrainer, TrainingConfig

def main():
    print("Testing TIDE-Lite setup...")
    
    # Test 1: Model initialization
    print("\n1. Testing model initialization...")
    try:
        config = TIDELiteConfig(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            mlp_hidden_dim=128,
            freeze_encoder=True
        )
        model = TIDELite(config)
        params = model.count_extra_parameters()
        print(f"   ✓ Model created with {params:,} extra parameters")
    except Exception as e:
        print(f"   ✗ Model initialization failed: {e}")
        return 1
    
    # Test 2: Training config
    print("\n2. Testing training configuration...")
    try:
        train_config = TrainingConfig(
            num_epochs=1,
            batch_size=8,
            dry_run=True,
            output_dir="test_output"
        )
        print(f"   ✓ Training config created")
    except Exception as e:
        print(f"   ✗ Training config failed: {e}")
        return 1
    
    # Test 3: Trainer initialization
    print("\n3. Testing trainer initialization...")
    try:
        trainer = TIDETrainer(model, train_config)
        print(f"   ✓ Trainer initialized")
    except Exception as e:
        print(f"   ✗ Trainer initialization failed: {e}")
        return 1
    
    print("\n✅ All tests passed! You can now run:")
    print("   python3 scripts/train.py --config configs/smoke.yaml --dry-run")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
