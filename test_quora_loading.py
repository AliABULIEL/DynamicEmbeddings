#!/usr/bin/env python
"""Test Quora dataset loading with new fallback approach."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.tide_lite.data.datasets import load_quora

print("Testing improved Quora dataset loading...")
print("-" * 60)

try:
    # Try loading with small sample
    cfg = {
        "cache_dir": "./data",
        "max_samples": 10,
        "seed": 42
    }
    
    corpus, queries, qrels = load_quora(cfg)
    
    print(f"✅ SUCCESS! Loaded Quora dataset:")
    print(f"   - Corpus: {len(corpus)} documents")
    print(f"   - Queries: {len(queries)} queries")
    print(f"   - Relevance pairs: {len(qrels)} judgments")
    
    if len(corpus) > 0:
        print(f"\nSample document: {corpus['text'][0][:100]}...")
        
except Exception as e:
    print(f"❌ Failed to load Quora: {e}")
    print("\nThis might be due to network issues or dataset availability.")
    print("The system will continue with STS-B dataset only.")

print("-" * 60)
