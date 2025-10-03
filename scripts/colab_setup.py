"""
Colab setup script for Temporal LoRA project.
Run this in a Colab cell, then restart runtime before proceeding.
"""

import subprocess
import sys


def run_command(cmd: str, description: str):
    """Run a shell command and print status."""
    print(f"\n{description}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and "uninstall" not in cmd:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)


def setup_environment():
    """Setup Colab environment with exact package versions."""
    
    print("=" * 70)
    print("🚀 Setting up Temporal LoRA Environment")
    print("=" * 70)
    
    # Step 1: Clean uninstall
    print("\n🔄 Step 1/8: Removing conflicting packages...")
    run_command(
        "pip uninstall -y sentence-transformers transformers torch accelerate peft numpy typer click -q",
        "Cleaning previous installations..."
    )
    
    # Step 2: Install torch
    print("\n📦 Step 2/8: Installing PyTorch...")
    run_command(
        "pip install torch==2.2.1 --no-cache-dir -q",
        "Installing torch==2.2.1"
    )
    
    # Step 3: Install numpy
    print("\n📦 Step 3/8: Installing NumPy...")
    run_command(
        'pip install "numpy>=1.26.0,<2.0.0" --no-cache-dir -q',
        "Installing numpy"
    )
    
    # Step 4: Install transformers
    print("\n📦 Step 4/8: Installing Transformers...")
    run_command(
        "pip install transformers==4.40.0 --no-cache-dir -q",
        "Installing transformers==4.40.0"
    )
    
    # Step 5: Install sentence-transformers
    print("\n📦 Step 5/8: Installing Sentence Transformers...")
    run_command(
        "pip install sentence-transformers==3.0.1 --no-cache-dir -q",
        "Installing sentence-transformers==3.0.1"
    )
    
    # Step 6: Install PEFT
    print("\n📦 Step 6/8: Installing PEFT libraries...")
    run_command(
        "pip install accelerate==0.29.0 peft==0.10.0 --no-cache-dir -q",
        "Installing accelerate and peft"
    )
    
    # Step 7: Install CLI tools
    print("\n📦 Step 7/8: Installing CLI tools...")
    run_command(
        'pip install "typer[all]==0.9.0" "click>=8.0.0,<8.2.0" --no-cache-dir -q',
        "Installing typer and click"
    )
    
    # Step 8: Install other dependencies
    print("\n📦 Step 8/8: Installing remaining dependencies...")
    run_command(
        "pip install datasets faiss-cpu pyyaml umap-learn scikit-learn matplotlib seaborn pandas --no-cache-dir -q",
        "Installing data science packages"
    )
    
    print("\n" + "=" * 70)
    print("✅ Installation Complete!")
    print("=" * 70)
    print("\n⚠️  CRITICAL: You MUST restart the runtime now!")
    print("   📍 Go to: Runtime → Restart runtime")
    print("   📍 Or use: Ctrl+M . (dot)")
    print("\n   After restart, verify installation with:")
    print("   >>> import torch")
    print("   >>> print(torch.__version__)")
    print("=" * 70)


if __name__ == "__main__":
    setup_environment()
