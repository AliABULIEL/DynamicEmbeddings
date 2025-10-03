# Troubleshooting Guide

## Dependency Compatibility Issues

### Problem: `AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`

**Cause:** PyTorch version is too old (< 2.2.0). The newer transformers library requires PyTorch >= 2.2.0.

**Solution:** Run the fix script to upgrade all dependencies:

```bash
bash scripts/fix_dependencies.sh
```

**Or manually:**

```bash
# Uninstall old versions
pip uninstall -y torch sentence-transformers transformers huggingface_hub

# Reinstall with updated versions
pip install -e ".[dev]"

# Verify installation
pytest
```

---

## FAISS Segmentation Fault (macOS)

### Problem: `Fatal Python error: Segmentation fault` during FAISS tests

**Cause:** FAISS architecture mismatch or BLAS library conflicts on macOS.

**Quick Diagnosis:**
```bash
bash scripts/diagnose_faiss.sh
```

**Solutions:** See detailed guide at [docs/FAISS_TROUBLESHOOTING.md](docs/FAISS_TROUBLESHOOTING.md)

Quick fix:
```bash
pip uninstall -y faiss-cpu
pip install faiss-cpu --no-cache-dir
```

For Apple Silicon Macs, conda is recommended:
```bash
conda install -c conda-forge faiss-cpu
```

---

## Common Issues

### 1. Import Error: `cannot import name 'cached_download'`
- **Cause:** Old sentence-transformers version incompatible with new huggingface_hub
- **Fix:** Run `bash scripts/fix_dependencies.sh`

### 2. PyTorch CUDA Compatibility
- If you need CUDA support, install the appropriate torch version:
  ```bash
  pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu118
  ```

### 3. Tests Still Failing After Fix
- Clear pytest cache: `rm -rf .pytest_cache __pycache__ src/**/__pycache__`
- Reinstall in development mode: `pip install -e ".[dev]"`

---

## Minimum Requirements

| Package | Version |
|---------|---------|
| Python | >= 3.9 |
| PyTorch | >= 2.2.0 |
| sentence-transformers | >= 3.0.0 |
| transformers | >= 4.35.0 |
| huggingface_hub | >= 0.20.0 |

---

## Verifying Installation

Check installed versions:
```bash
pip show torch sentence-transformers transformers huggingface_hub | grep -E "Name:|Version:"
```

Run tests:
```bash
pytest -v
```

Expected output: All tests should pass (45+ tests collected).
