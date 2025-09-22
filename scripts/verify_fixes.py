#!/usr/bin/env python3
"""Quick test to verify fixes work."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from src.tide_lite.models import TIDELite, TIDELiteConfig
        print("  ✓ TIDELite and TIDELiteConfig imported successfully")
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    
    try:
        from src.tide_lite.train import TIDETrainer, TrainingConfig
        print("  ✓ TIDETrainer and TrainingConfig imported successfully")
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    
    return True

def test_seed_generation():
    """Test that timestamp generation works with various seeds."""
    print("\nTesting seed generation...")
    try:
        from src.tide_lite.data.datasets import _generate_synthetic_timestamps
        import numpy as np
        
        # Test with various seed values including edge cases
        test_seeds = [42, 0, 2**32-2, -1, None, 1000000000]
        
        for seed in test_seeds:
            try:
                timestamps = _generate_synthetic_timestamps(
                    n_samples=10,
                    start_date="2020-01-01",
                    end_date="2024-01-01",
                    seed=seed
                )
                print(f"  ✓ Seed {seed}: Generated {len(timestamps)} timestamps")
            except Exception as e:
                print(f"  ✗ Seed {seed} failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_dataset_loading():
    """Test that STS-B dataset loading works."""
    print("\nTesting dataset loading...")
    try:
        from src.tide_lite.data.datasets import DatasetConfig, load_stsb_with_timestamps
        
        # Small config for quick test
        config = DatasetConfig(
            seed=42,
            max_samples=10,  # Just load 10 samples for quick test
            cache_dir="./data"
        )
        
        print("  Loading STS-B with timestamps...")
        datasets = load_stsb_with_timestamps(config)
        
        for split in ["train", "validation", "test"]:
            if split in datasets:
                print(f"  ✓ {split}: {len(datasets[split])} samples")
                # Check required columns
                required_cols = ["sentence1", "sentence2", "label", "timestamp1", "timestamp2"]
                for col in required_cols:
                    if col not in datasets[split].column_names:
                        print(f"  ✗ Missing column: {col}")
                        return False
        
        print("  ✓ All required columns present")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TIDE-LITE FIX VERIFICATION")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Imports
    if not test_imports():
        all_passed = False
    
    # Test 2: Seed generation
    if not test_seed_generation():
        all_passed = False
    
    # Test 3: Dataset loading  
    if not test_dataset_loading():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYou can now run:")
        print("  python3 scripts/train.py --config configs/smoke.yaml")
        print("  python3 scripts/train_simple.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
