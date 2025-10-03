#!/bin/bash
# Script to diagnose and fix FAISS issues on macOS

set -e

echo "🔍 Diagnosing FAISS installation..."
echo ""

# Check architecture
ARCH=$(uname -m)
echo "System architecture: $ARCH"

# Check python3 architecture
PY_ARCH=$(python3 -c "import platform; print(platform.machine())")
echo "python3 architecture: $PY_ARCH"

# Check if architectures match
if [ "$ARCH" != "$PY_ARCH" ]; then
    echo "⚠️  WARNING: System and python3 architectures don't match!"
    echo "   This can cause FAISS crashes."
fi

echo ""

# Check FAISS installation
echo "Checking FAISS installation..."
if python3 -c "import faiss" 2>/dev/null; then
    FAISS_VERSION=$(python3 -c "import faiss; print(faiss.__version__)" 2>/dev/null || echo "unknown")
    echo "✅ FAISS installed: version $FAISS_VERSION"
else
    echo "❌ FAISS not installed"
    exit 1
fi

echo ""

# Test FAISS
echo "Testing FAISS basic operations..."
if python3 -c "
import faiss
import numpy as np
np.random.seed(42)
x = np.random.randn(10, 5).astype('float32')
x = np.ascontiguousarray(x)
idx = faiss.IndexFlatL2(5)
idx.add(x)
_, _ = idx.search(x[:3], 3)
print('Basic operations: OK')
" 2>/dev/null; then
    echo "✅ FAISS basic test passed"
else
    echo "❌ FAISS test failed with segmentation fault"
    echo ""
    echo "🔧 Attempting to fix..."
    echo ""

    # Try reinstalling
    echo "Reinstalling faiss-cpu..."
    pip uninstall -y faiss-cpu
    pip install faiss-cpu --no-cache-dir

    echo ""
    echo "Testing again..."
    if python3 -c "
import faiss
import numpy as np
x = np.random.randn(10, 5).astype('float32')
x = np.ascontiguousarray(x)
idx = faiss.IndexFlatL2(5)
idx.add(x)
_, _ = idx.search(x[:3], 3)
" 2>/dev/null; then
        echo "✅ Fix successful!"
    else
        echo "❌ Still failing. Please see docs/FAISS_TROUBLESHOOTING.md"
        echo "   Consider using conda: conda install -c conda-forge faiss-cpu"
        exit 1
    fi
fi

echo ""
echo "✅ FAISS is working correctly!"
echo ""
echo "You can now run tests:"
echo "  pytest tests/test_eval_smoke.py::TestFAISSRoundtrip -v"
