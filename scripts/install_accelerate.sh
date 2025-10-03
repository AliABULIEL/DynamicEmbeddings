#!/bin/bash
# Quick fix: Install missing accelerate package

set -e

echo "📦 Installing missing accelerate package..."
echo ""

pip install "accelerate>=0.26.0"

echo ""
echo "✅ accelerate installed!"
echo ""
echo "Now run tests again:"
echo "  bash scripts/run_tests.sh"
