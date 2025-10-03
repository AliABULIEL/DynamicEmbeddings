# FAISS Segmentation Fault Fix for macOS

## Problem
Tests crash with `Segmentation fault` when using FAISS, especially on Apple Silicon (M1/M2/M3) Macs.

## Quick Fix (Try First)

The code has been updated to use contiguous arrays. Try running tests again:
```bash
pytest tests/test_eval_smoke.py::TestFAISSRoundtrip -v
```

If still crashing, continue to solutions below.

---

## Solution 1: Reinstall FAISS for Your Architecture

### For Apple Silicon (M1/M2/M3 Macs):

```bash
# Uninstall current FAISS
pip uninstall -y faiss-cpu

# Install ARM64-compatible version
pip install faiss-cpu --no-cache-dir
```

### For Intel Macs:

```bash
# Uninstall current FAISS
pip uninstall -y faiss-cpu

# Reinstall
pip install faiss-cpu --no-cache-dir
```

---

## Solution 2: Use Conda (Most Reliable)

If you have conda/miniconda:

```bash
# Create new environment
conda create -n temporal-lora python=3.10
conda activate temporal-lora

# Install FAISS from conda-forge (most stable)
conda install -c conda-forge faiss-cpu

# Install other dependencies
pip install -e ".[dev]"
```

---

## Solution 3: Skip FAISS Tests (Temporary)

If you need to continue without FAISS tests:

```bash
# Run tests excluding FAISS
pytest -v -k "not FAISS"
```

---

## Verify Your Architecture

Check if you're on Apple Silicon or Intel:
```bash
uname -m
# arm64 = Apple Silicon (M1/M2/M3)
# x86_64 = Intel
```

Check Python architecture:
```bash
python -c "import platform; print(platform.machine())"
```

Check FAISS installation:
```bash
python -c "import faiss; print(faiss.__version__)"
```

---

## Root Causes

1. **Architecture Mismatch**: Installing x86_64 FAISS on ARM Mac (or vice versa)
2. **BLAS Conflicts**: OpenBLAS vs Accelerate framework conflicts
3. **Memory Alignment**: Non-contiguous numpy arrays (now fixed in code)
4. **Old FAISS Version**: Older versions have more crashes

---

## Testing After Fix

Run these tests to verify FAISS works:

```bash
# Quick test
python -c "import faiss, numpy as np; x = np.random.randn(10, 5).astype('float32'); idx = faiss.IndexFlatL2(5); idx.add(x); print('FAISS OK!')"

# Full test suite
pytest tests/test_eval_smoke.py -v
```

Expected output: `FAISS OK!` and all tests passing.

---

## Still Having Issues?

1. Try the conda solution (Solution 2) - it's most reliable
2. Check system logs: `Console.app` → filter for "pytest" or "Python"
3. Try building FAISS from source (advanced):
   ```bash
   git clone https://github.com/facebookresearch/faiss.git
   cd faiss
   cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON .
   make -C build -j
   cd build/faiss/python && pip install .
   ```

---

## Alternative: Use Scikit-learn Instead

For small-scale testing, you can temporarily use sklearn instead of FAISS:

```python
from sklearn.metrics.pairwise import cosine_similarity

# Instead of FAISS
scores = cosine_similarity(query_embeddings, index_embeddings)
top_k_indices = np.argsort(-scores, axis=1)[:, :k]
```

Note: This is slower for large datasets but works reliably.
