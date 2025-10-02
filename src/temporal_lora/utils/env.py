"""Environment utilities for reproducibility."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def get_cuda_info() -> Dict[str, str]:
    """Get CUDA availability and version info.
    
    Returns:
        Dictionary with CUDA information.
    """
    info = {"cuda_available": "false", "cuda_version": "N/A"}
    
    try:
        import torch
        info["cuda_available"] = str(torch.cuda.is_available()).lower()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["device_count"] = str(torch.cuda.device_count())
            info["device_name"] = torch.cuda.get_device_name(0)
            info["torch_version"] = torch.__version__
    except ImportError:
        info["torch_version"] = "Not installed"
    
    return info


def get_nvidia_smi() -> str:
    """Get nvidia-smi output.
    
    Returns:
        nvidia-smi output as string.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nvidia-smi not available or no NVIDIA GPU detected"


def get_pip_freeze() -> str:
    """Get pip freeze output.
    
    Returns:
        Pip freeze output as string.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return "Failed to run pip freeze"


def get_git_sha(repo_path: Optional[Path] = None) -> str:
    """Get current git commit SHA.
    
    Args:
        repo_path: Path to git repository. If None, uses current directory.
        
    Returns:
        Git commit SHA or error message.
    """
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        if repo_path:
            cmd.extend(["-C", str(repo_path)])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "Not a git repository or git not available"


def dump_environment(output_dir: Path, repo_path: Optional[Path] = None) -> None:
    """Dump environment information to files.
    
    Args:
        output_dir: Directory to write environment files.
        repo_path: Path to git repository for SHA extraction.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    
    # Collect all info
    cuda_info = get_cuda_info()
    nvidia_smi = get_nvidia_smi()
    pip_freeze = get_pip_freeze()
    git_sha = get_git_sha(repo_path)
    
    # CUDA info file
    cuda_file = output_dir / "cuda_info.txt"
    with open(cuda_file, "w") as f:
        f.write(f"Generated at: {timestamp}\n\n")
        for key, value in cuda_info.items():
            f.write(f"{key}: {value}\n")
    
    # nvidia-smi output
    nvidia_file = output_dir / "nvidia_smi.txt"
    with open(nvidia_file, "w") as f:
        f.write(f"Generated at: {timestamp}\n\n")
        f.write(nvidia_smi)
    
    # Pip freeze
    pip_file = output_dir / "requirements_frozen.txt"
    with open(pip_file, "w") as f:
        f.write(pip_freeze)
    
    # Git SHA
    git_file = output_dir / "git_commit.txt"
    with open(git_file, "w") as f:
        f.write(f"Generated at: {timestamp}\n\n")
        f.write(f"Commit SHA: {git_sha}\n")
    
    # Consolidated JSON
    env_data = {
        "timestamp": timestamp,
        "git_commit": git_sha,
        "cuda": cuda_info,
        "python_version": sys.version,
        "platform": sys.platform,
    }
    
    json_file = output_dir / "environment.json"
    with open(json_file, "w") as f:
        json.dump(env_data, f, indent=2)
    
    print(f"Environment info dumped to: {output_dir}")
    print(f"  - {cuda_file.name}")
    print(f"  - {nvidia_file.name}")
    print(f"  - {pip_file.name}")
    print(f"  - {git_file.name}")
    print(f"  - {json_file.name}")
