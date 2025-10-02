"""Environment information dumping for reproducibility."""

import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

from temporal_lora.utils.io import save_json


def get_git_commit() -> Optional[str]:
    """Get current git commit SHA.
    
    Returns:
        str: Commit SHA or None if not in git repo
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


def get_pip_freeze() -> list[str]:
    """Get pip freeze output.
    
    Returns:
        list: List of installed packages with versions
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        return []


def get_cuda_info() -> dict[str, Any]:
    """Get CUDA information.
    
    Returns:
        dict: CUDA availability, version, and device info
    """
    info = {
        "available": torch.cuda.is_available(),
        "version": None,
        "device_count": 0,
        "devices": [],
    }
    
    if torch.cuda.is_available():
        info["version"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count()
        info["devices"] = [
            {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / 1e9,
            }
            for i in range(torch.cuda.device_count())
        ]
    
    return info


def dump_environment(output_dir: Path) -> Path:
    """Dump complete environment information to JSON.
    
    Args:
        output_dir: Directory to save environment dump
    
    Returns:
        Path: Path to saved environment file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"env_dump_{timestamp}.json"
    
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda": get_cuda_info(),
        },
        "git": {
            "commit": get_git_commit(),
        },
        "packages": get_pip_freeze(),
    }
    
    save_json(env_info, output_path, indent=2)
    
    return output_path
