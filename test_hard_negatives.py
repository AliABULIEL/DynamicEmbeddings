"""Test hard negatives implementation to verify it's fast and functional."""

import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sentence_transformers import InputExample
from temporal_lora.train.hard_negatives import add_hard_temporal_negatives
from temporal_lora.utils.logging import get_logger

logger = get_logger(__name__)


def test_hard_negatives_speed():
    """Test that hard negative generation is fast (< 2 minutes for small dataset)."""
    
    print("\n" + "="*60)
    print("TESTING HARD NEGATIVES IMPLEMENTATION")
    print("="*60 + "\n")
    
    # Path to your processed data
    data_dir = Path(__file__).parent / "data" / "processed"
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run data preparation first:")
        print("  python -m temporal_lora.cli prepare-data")
        return False
    
    # Test with small subset
    all_bins = ["pre2016", "2016-2018", "2019-2021", "2022-2023", "2024+"]
    test_bucket = "2019-2021"
    
    # Create dummy training examples (small set for testing)
    print(f"Creating test examples from bucket: {test_bucket}")
    
    train_examples = [
        InputExample(
            texts=[
                "Machine learning advances",
                "We propose a new approach to machine learning using neural networks"
            ],
            label=1.0,
            guid=f"test_{i}"
        )
        for i in range(50)  # Just 50 examples for testing
    ]
    
    print(f"Created {len(train_examples)} test examples")
    print(f"Testing hard negative mining...")
    print(f"Target: < 2 minutes (old implementation: 30+ minutes)\n")
    
    # Time the operation
    start_time = time.time()
    
    try:
        augmented = add_hard_temporal_negatives(
            data_dir=data_dir,
            all_bins=all_bins,
            bucket_name=test_bucket,
            train_examples=train_examples,
            neg_k=4,
            seed=42,
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"✅ Hard negative mining completed!")
        print(f"⏱️  Time taken: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"📊 Original examples: {len(train_examples)}")
        print(f"📊 Augmented examples: {len(augmented)}")
        print(f"📊 Ratio: {len(augmented)/len(train_examples):.1f}x")
        print(f"{'='*60}\n")
        
        # Validate results
        if elapsed > 120:  # > 2 minutes
            print("⚠️  WARNING: Still too slow (>2 min for 50 examples)")
            print("Expected: <30 seconds for 50 examples")
            return False
        
        if len(augmented) <= len(train_examples):
            print("⚠️  WARNING: No hard negatives were added!")
            return False
        
        if elapsed < 1:
            print("⚠️  WARNING: Too fast - might not be working correctly")
            return False
        
        # Success criteria
        if 1 <= elapsed <= 120 and len(augmented) > len(train_examples):
            print("✅ TEST PASSED!")
            print(f"   - Fast enough: {elapsed:.2f}s << 2 minutes")
            print(f"   - Augmented correctly: {len(train_examples)} → {len(augmented)}")
            print("\n🎉 Hard negatives implementation is working correctly!")
            print("\nYou can now retrain with:")
            print("  python -m temporal_lora.cli train-adapters \\")
            print("      --mode lora \\")
            print("      --epochs 5 \\")
            print("      --lora-r 32 \\")
            print("      --hard-temporal-negatives")
            return True
        
        return False
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ TEST FAILED after {elapsed:.2f} seconds")
        print(f"Error: {e}")
        print(f"\nFull traceback:")
        import traceback
        traceback.print_exc()
        return False


def test_functionality():
    """Test basic functionality without actual mining."""
    print("\n" + "="*60)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*60 + "\n")
    
    from temporal_lora.train.hard_negatives import get_adjacent_bins
    
    # Test adjacent bins logic
    all_bins = ["pre2016", "2016-2018", "2019-2021", "2022-2023", "2024+"]
    
    test_cases = [
        ("2019-2021", ["2016-2018", "2022-2023"]),  # Middle
        ("pre2016", ["2016-2018"]),  # First
        ("2024+", ["2022-2023"]),  # Last
    ]
    
    all_passed = True
    
    for current, expected in test_cases:
        result = get_adjacent_bins(current, all_bins)
        passed = result == expected
        status = "✅" if passed else "❌"
        print(f"{status} {current}: {result} {'==' if passed else '!='} {expected}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All functionality tests passed!")
    else:
        print("\n❌ Some functionality tests failed!")
    
    return all_passed


if __name__ == "__main__":
    print("\n🧪 Starting Hard Negatives Tests\n")
    
    # Run functionality tests first
    func_passed = test_functionality()
    
    if not func_passed:
        print("\n❌ Basic functionality tests failed!")
        print("Please check the implementation.")
        sys.exit(1)
    
    # Run speed test
    speed_passed = test_hard_negatives_speed()
    
    if speed_passed:
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nYour hard negatives implementation is:")
        print("  ✅ Fast (seconds instead of hours)")
        print("  ✅ Functional (correctly augments examples)")
        print("  ✅ Ready for training")
        print("\nNext step: Retrain your models!")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("❌ TESTS FAILED")
        print("="*60)
        print("\nThe implementation may still have issues.")
        print("Check the error messages above.")
        sys.exit(1)
