#!/bin/bash
# Clear Python cache and run evaluation

echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo "Running evaluation..."
python3 scripts/run_evaluation.py --checkpoint-dir outputs/smoke_test
