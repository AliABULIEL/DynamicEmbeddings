#!/bin/bash
# Script to fix dependency issues

set -e

echo "🔧 Fixing dependency compatibility issues..."
echo ""

# Uninstall problematic packages
echo "📦 Uninstalling old packages..."
pip uninstall -y sentence-transformers transformers huggingface_hub 2>/dev/null || true

# Reinstall from pyproject.toml
echo ""
echo "📥 Reinstalling compatible versions..."
pip install -e ".[dev]"

echo ""
echo "✅ Dependencies fixed! Run 'pytest' to verify tests work."
