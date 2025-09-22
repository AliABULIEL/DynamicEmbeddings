# How to Run TIDE-Lite Locally

This guide provides step-by-step instructions for setting up and running TIDE-Lite on your local machine.

## Prerequisites

### System Requirements
- **OS:** Linux, macOS, or Windows 10/11
- **Python:** 3.8 or higher
- **RAM:** Minimum 4GB (8GB recommended)
- **Storage:** 2GB free space
- **GPU (optional):** NVIDIA GPU with CUDA 11.0+ for faster training

### Check Prerequisites
```bash
# Check Python version
python --version  # Should show 3.8+

# Check pip
pip --version

# Check CUDA (optional, for GPU)
nvidia-smi  # Should show CUDA version if GPU available

# Check available memory
free -h  # Linux/macOS
# or
wmic OS get TotalVisibleMemorySize /Value  # Windows
```

## Installation Steps

### Step 1: Clone Repository
```bash
# Clone the repository
git clone https://github.com/yourusername/TIDE-Lite.git
cd TIDE-Lite

# Or download and extract ZIP
# wget https://github.com/yourusername/TIDE-Lite/archive/main.zip
# unzip main.zip && cd TIDE-Lite-main
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# On Windows (Git Bash):
source .venv/Scripts/activate

# Verify activation (should show .venv path)
which python  # Linux/macOS
where python  # Windows
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# OR Install PyTorch (GPU version - CUDA 11.8)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sentence_transformers; print('Sentence Transformers: OK')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 4: Download Pre-trained Models (First Run)
```bash
# The base encoder will be downloaded automatically on first use
# To pre-download:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

## Complete Training Pipeline

### End-to-End Commands (Copy & Paste)

```bash
#!/bin/bash
# File: run_complete_pipeline.sh
# Complete TIDE-Lite pipeline from training to reporting

# Set up environment
source .venv/bin/activate

# 1. TRAIN - Train TIDE-Lite model
echo "Step 1: Training TIDE-Lite..."
python -m tide_lite.cli.tide train \
    --config configs/defaults.yaml \
    --output-dir results/local_run \
    --num-epochs 3 \
    --batch-size 32 \
    --learning-rate 5e-5 \
    --temporal-weight 0.1

# 2. EVAL-STSB - Evaluate on STS-B benchmark  
echo "Step 2: Evaluating on STS-B..."
python -m tide_lite.cli.tide eval-stsb \
    --model-path results/local_run/checkpoints/final.pt \
    --output-dir results/local_run/eval_stsb \
    --split test \
    --compare-baseline

# 3. EVAL-QUORA - Evaluate on Quora retrieval
echo "Step 3: Evaluating on Quora..."
python -m tide_lite.cli.tide eval-quora \
    --model-path results/local_run/checkpoints/final.pt \
    --output-dir results/local_run/eval_quora \
    --max-corpus 10000 \
    --max-queries 1000

# 4. EVAL-TEMPORAL - Evaluate temporal capabilities
echo "Step 4: Evaluating temporal performance..."
python -m tide_lite.cli.tide eval-temporal \
    --model-path results/local_run/checkpoints/final.pt \
    --output-dir results/local_run/eval_temporal \
    --time-window-days 7 \
    --compare-baseline

# 5. BENCH-ALL - Run all benchmarks together
echo "Step 5: Running comprehensive benchmark..."
python -m tide_lite.cli.tide bench-all \
    --model-path results/local_run/checkpoints/final.pt \
    --output-dir results/local_run/bench_all

# 6. AGGREGATE - Collect all results
echo "Step 6: Aggregating results..."
python -m tide_lite.cli.tide aggregate \
    --results-dir results/local_run \
    --output results/local_run/summary.json

# 7. REPORT - Generate final report
echo "Step 7: Generating report..."
python -m tide_lite.cli.tide report \
    --input results/local_run/summary.json \
    --output-dir results/local_run/report

echo "Pipeline complete! Check results/local_run/report/ for final report."
```

### Make Script Executable and Run
```bash
# Make executable
chmod +x run_complete_pipeline.sh

# Run the complete pipeline
./run_complete_pipeline.sh
```

## Quick Tests

### Test Installation (Dry Run)
```bash
# Test without execution
python -m tide_lite.cli.tide train --dry-run --config configs/defaults.yaml
python -m tide_lite.cli.tide bench-all --dry-run --model-path dummy.pt
```

### Minimal Training Test
```bash
# Quick 1-epoch training for testing setup
python -m tide_lite.cli.tide train \
    --output-dir results/test \
    --num-epochs 1 \
    --batch-size 16 \
    --eval-every-n-steps 50
```

## Memory-Constrained Setup

### For Systems with <8GB RAM
```bash
# Reduce batch size and sequence length
python -m tide_lite.cli.tide train \
    --batch-size 8 \
    --eval-batch-size 16 \
    --max-seq-length 64 \
    --gradient-accumulation-steps 4 \
    --num-workers 0
```

### CPU-Only Training
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python -m tide_lite.cli.tide train \
    --device cpu \
    --batch-size 4 \
    --no-amp  # Disable mixed precision
```

## Monitoring Progress

### Real-time Monitoring
```bash
# In a separate terminal, monitor training
tail -f results/local_run/training.log

# Watch GPU usage (if using GPU)
watch -n 1 nvidia-smi

# Monitor system resources
htop  # Linux/macOS
# or
perfmon  # Windows
```

### Check Results
```bash
# View training metrics
cat results/local_run/metrics_train.json | python -m json.tool

# View evaluation results
cat results/local_run/eval_stsb/results.json | python -m json.tool

# List all checkpoints
ls -lh results/local_run/checkpoints/
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA/GPU Not Found
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CPU version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Out of Memory (OOM)
```bash
# Reduce batch size
python -m tide_lite.cli.tide train --batch-size 8

# Enable gradient checkpointing
python -m tide_lite.cli.tide train --gradient-checkpoint

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 3. Import Errors
```bash
# Reinstall dependencies
pip install --upgrade --force-reinstall -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"

# Add project to path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### 4. Slow Training
```bash
# Enable mixed precision
python -m tide_lite.cli.tide train --use-amp

# Increase workers for data loading
python -m tide_lite.cli.tide train --num-workers 4

# Use smaller model for testing
python -m tide_lite.cli.tide train --encoder-name google/bert_uncased_L-2_H-128_A-2
```

## Advanced Local Setup

### Using Custom Conda Environment
```bash
# Create conda environment
conda create -n tide-lite python=3.9
conda activate tide-lite

# Install PyTorch with conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Docker Setup
```bash
# Build Docker image
docker build -t tide-lite .

# Run training in container
docker run --gpus all -v $(pwd)/results:/app/results tide-lite \
    python -m tide_lite.cli.tide train --output-dir /app/results/docker_run
```

### Multi-GPU Setup (DataParallel)
```bash
# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
python -m tide_lite.cli.tide train \
    --batch-size 64 \
    --data-parallel
```

## Performance Optimization

### Faster Training
```bash
# Optimized settings for faster training
python -m tide_lite.cli.tide train \
    --batch-size 64 \
    --use-amp \
    --num-workers 4 \
    --pin-memory \
    --persistent-workers \
    --prefetch-factor 2
```

### Profile Performance
```bash
# Enable profiling
python -m tide_lite.cli.tide train \
    --profile \
    --profile-steps 10 \
    --output-dir results/profiled
```

## Next Steps

1. **Run Experiments:** Try different hyperparameters
2. **Custom Data:** See [docs/custom_data.md](../docs/custom_data.md)
3. **Deploy Model:** See [docs/deployment.md](../docs/deployment.md)
4. **Contribute:** See [CONTRIBUTING.md](../CONTRIBUTING.md)

## Support

If you encounter issues:
1. Check the [FAQ](../docs/FAQ.md)
2. Search [GitHub Issues](https://github.com/yourusername/TIDE-Lite/issues)
3. Join our [Discord](https://discord.gg/tide-lite)
4. Email: support@tide-lite.ai
