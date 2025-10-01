#!/usr/bin/env python
"""Main training script for TIDE-Lite models.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --dry-run  # Test without training
    python scripts/train.py --num-epochs 1 --batch-size 16  # Quick test
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tide_lite.cli.train_cli import main

if __name__ == "__main__":
    sys.exit(main())
