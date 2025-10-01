#!/usr/bin/env python3
"""Clean all Python cache files to ensure fresh execution."""

import os
import shutil
from pathlib import Path

def clean_pycache(root_dir="."):
    """Remove all __pycache__ directories and .pyc files."""
    root = Path(root_dir)
    
    # Find and remove all __pycache__ directories
    pycache_dirs = list(root.glob("**/__pycache__"))
    for pycache_dir in pycache_dirs:
        print(f"Removing: {pycache_dir}")
        shutil.rmtree(pycache_dir, ignore_errors=True)
    
    # Find and remove all .pyc files (in case any are outside __pycache__)
    pyc_files = list(root.glob("**/*.pyc"))
    for pyc_file in pyc_files:
        print(f"Removing: {pyc_file}")
        pyc_file.unlink(missing_ok=True)
    
    # Also remove .pyo files (optimized Python bytecode)
    pyo_files = list(root.glob("**/*.pyo"))
    for pyo_file in pyo_files:
        print(f"Removing: {pyo_file}")
        pyo_file.unlink(missing_ok=True)
    
    print(f"\n✅ Cleaned {len(pycache_dirs)} __pycache__ directories")
    print(f"✅ Cleaned {len(pyc_files)} .pyc files")
    print(f"✅ Cleaned {len(pyo_files)} .pyo files")

if __name__ == "__main__":
    clean_pycache()
    print("\n✨ All Python cache files have been cleaned!")
    print("Your next run will use fresh source code.")
