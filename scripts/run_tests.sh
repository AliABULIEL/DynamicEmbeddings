#!/bin/bash
# Safe test runner for macOS with FAISS

set -e

echo "🧪 Running tests with FAISS-safe environment..."
echo ""

# Set threading to single-threaded to prevent FAISS crashes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE

# For Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Disable pytest parallelization if installed
export PYTEST_XDIST_WORKER_COUNT=0

echo "Environment configured:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  Platform: $(uname -m)"
echo ""

# Run pytest with single process
if [ -z "$1" ]; then
    # Run all tests
    echo "Running all tests..."
    python3 -m pytest -v --tb=short -p no:xdist
else
    # Run specific test
    echo "Running: $@"
    python3 -m pytest -v --tb=short -p no:xdist "$@"
fi
