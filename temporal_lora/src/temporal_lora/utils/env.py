"""
Environment capture utilities for reproducibility.

Captures CUDA info, package versions, git commit, and system details.
"""

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

from temporal_lora.utils.io import save_json
from temporal_lora.utils.paths import get_repro_dir


def get_git_commit() -> Optional[str]:
    """
    Get current git commit hash.

    Returns:
        Commit hash string or None if not in a git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_cuda_info() -> Dict[str, any]:
    """
    Get CUDA and GPU information.

    Returns:
        Dictionary with CUDA availability, version, device info
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["gpu_devices"] = [
            {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_gb": torch.cuda.get_device_properties(i).total_memory / 1e9,
            }
            for i in range(torch.cuda.device_count())
        ]

        # Try nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                capture_output=True,
                text=True,
                check=True,
            )
            info["nvidia_smi"] = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            info["nvidia_smi"] = "nvidia-smi not available"

    return info


def get_package_versions() -> Dict[str, str]:
    """
    Get versions of key packages.

    Returns:
        Dictionary mapping package names to versions
    """
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        packages = {}
        for line in result.stdout.strip().split("\n"):
            if "==" in line:
                pkg, ver = line.split("==")
                packages[pkg] = ver
        return packages
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def get_system_info() -> Dict[str, str]:
    """
    Get system platform information.

    Returns:
        Dictionary with OS, Python version, architecture
    """
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }


def dump_environment(output_path: Optional[Path] = None) -> Dict[str, any]:
    """
    Capture full environment information for reproducibility.

    Args:
        output_path: Path to save environment.json (defaults to repro dir)

    Returns:
        Dictionary with environment information
    """
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "system": get_system_info(),
        "cuda": get_cuda_info(),
        "pytorch_version": torch.__version__,
        "packages": get_package_versions(),
    }

    if output_path is None:
        output_path = get_repro_dir() / "environment.json"

    save_json(env_info, output_path, indent=2)

    return env_info
