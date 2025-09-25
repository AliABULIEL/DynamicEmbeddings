#!/usr/bin/env python
"""Quick test to verify imports are working."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    print("Testing imports...")
    
    # Test data imports
    from src.tide_lite.data.datasets import load_stsb, load_quora
    print("✓ Data imports OK")
    
    # Test model imports
    from src.tide_lite.models import (
        TIDELite, 
        TIDELiteConfig,
        BaselineEncoder,
        load_baseline,
        load_minilm_baseline
    )
    print("✓ Model imports OK")
    
    # Test creating models
    print("\nTesting model creation...")
    
    # Create baseline
    baseline = load_minilm_baseline()
    print(f"✓ MiniLM baseline created: {baseline.model_name}")
    
    # Create TIDE-Lite
    config = TIDELiteConfig(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        freeze_encoder=True
    )
    tide = TIDELite(config)
    print(f"✓ TIDE-Lite created with {tide.count_extra_parameters():,} extra params")
    
    # Test encoding (if models loaded)
    print("\nTesting encoding...")
    texts = ["Test sentence one.", "Test sentence two."]
    
    import torch
    with torch.no_grad():
        baseline_emb = baseline.encode_texts(texts, batch_size=2)
        print(f"✓ Baseline embeddings: {baseline_emb.shape}")
        
        tide_emb = tide.encode_texts(texts, batch_size=2)
        print(f"✓ TIDE-Lite embeddings: {tide_emb.shape}")
    
    print("\n✅ All imports and basic functionality working!")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
