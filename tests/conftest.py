"""Pytest configuration and fixtures."""

import os
import sys

import pytest


def pytest_configure(config):
    """Configure pytest environment for macOS FAISS compatibility."""
    # Disable OpenMP threading which causes issues with FAISS on macOS
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # Prevent numpy threading issues
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    
    # For Apple Silicon Macs, ensure we're not using Rosetta
    if sys.platform == "darwin":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


@pytest.fixture(autouse=True, scope="session")
def setup_faiss():
    """Setup FAISS to work reliably in pytest on macOS."""
    try:
        import faiss
        # Force single-threaded mode
        faiss.omp_set_num_threads(1)
    except (ImportError, AttributeError):
        pass  # FAISS not installed or doesn't support threading control
    
    yield


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path
