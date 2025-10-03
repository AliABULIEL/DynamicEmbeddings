#!/bin/bash
# Script to fix dependency issues

set -e

echo "🔧 Fixing dependency compatibility issues..."
echo ""
echo "⚠️  This will upgrade PyTorch to 2.2.0+ and related packages."
echo ""

# Uninstall problematic packages
echo "📦 Uninstalling old packages..."
pip uninstall -y torch sentence-transformers transformers huggingface_hub 2>/dev/null || true

# Reinstall from pyproject.toml
echo ""
echo "📥 Reinstalling compatible versions (this may take a few minutes)..."
pip install -e ".[dev]"

echo ""
echo "✅ Dependencies fixed! Run 'pytest' to verify tests work."
echo ""
echo "Installed versions:"
pip show torch sentence-transformers transformers | grep -E "Name:|Version:"
