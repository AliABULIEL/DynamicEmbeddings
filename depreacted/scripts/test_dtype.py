#!/usr/bin/env python3
"""Test that dtype issues are fixed."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

def test_dtype_consistency():
    """Test that all components use consistent dtypes."""
    print("Testing dtype consistency...")
    
    from src.tide_lite.models import TIDELite, TIDELiteConfig
    from src.tide_lite.data.collate import STSBCollator, TextBatcher
    
    # Create model
    config = TIDELiteConfig(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        time_encoding_dim=32,
        mlp_hidden_dim=128
    )
    model = TIDELite(config)
    
    # Create mock batch
    tokenizer = TextBatcher(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_length=64
    )
    collator = STSBCollator(tokenizer, include_timestamps=True)
    
    # Mock data
    batch_data = [
        {
            "sentence1": "This is a test.",
            "sentence2": "This is another test.",
            "label": 4.0,
            "timestamp1": 1609459200.0,  # 2021-01-01
            "timestamp2": 1609459200.0,
        }
    ]
    
    # Collate batch
    collated = collator(batch_data)
    
    # Check dtypes
    print(f"  timestamps1 dtype: {collated['timestamps1'].dtype}")
    print(f"  timestamps2 dtype: {collated['timestamps2'].dtype}")
    print(f"  labels dtype: {collated['labels'].dtype}")
    
    # Test forward pass
    try:
        with torch.no_grad():
            temporal_emb, base_emb = model(
                collated["sentence1_inputs"]["input_ids"],
                collated["sentence1_inputs"]["attention_mask"],
                collated["timestamps1"]
            )
        print(f"  ✓ Forward pass successful!")
        print(f"  Output shape: {temporal_emb.shape}")
        print(f"  Output dtype: {temporal_emb.dtype}")
        return True
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False

def test_time_encoding_dtypes():
    """Test time encoding with various input dtypes."""
    print("\nTesting time encoding dtypes...")
    
    from src.tide_lite.models.tide_lite import SinusoidalTimeEncoding
    
    encoder = SinusoidalTimeEncoding(encoding_dim=32)
    
    # Test with different input dtypes
    test_cases = [
        ("float32", torch.tensor([1609459200.0], dtype=torch.float32)),
        ("float64", torch.tensor([1609459200.0], dtype=torch.float64)),
        ("int32", torch.tensor([1609459200], dtype=torch.int32)),
        ("int64", torch.tensor([1609459200], dtype=torch.int64)),
    ]
    
    for name, timestamps in test_cases:
        try:
            encoding = encoder(timestamps)
            print(f"  ✓ {name}: Input shape={timestamps.shape}, Output shape={encoding.shape}, dtype={encoding.dtype}")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            return False
    
    return True

def main():
    """Run all dtype tests."""
    print("=" * 60)
    print("DTYPE CONSISTENCY TEST")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Basic dtype consistency
    if not test_dtype_consistency():
        all_passed = False
    
    # Test 2: Time encoding with various dtypes
    if not test_time_encoding_dtypes():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL DTYPE TESTS PASSED!")
        print("\nYou can now run training without dtype errors:")
        print("  python3 scripts/train.py --config configs/smoke.yaml")
    else:
        print("❌ SOME DTYPE TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
