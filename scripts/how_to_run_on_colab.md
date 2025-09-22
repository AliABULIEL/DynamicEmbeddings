# How to Run TIDE-Lite on Google Colab

This guide provides complete instructions for running TIDE-Lite on Google Colab, perfect for users without local GPU access.

## Quick Start (2 Minutes)

### Open in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/TIDE-Lite/blob/main/notebooks/tide_lite_quickstart.ipynb)

### One-Click Setup Cell
```python
# Run this in the first cell to set up everything
!git clone https://github.com/yourusername/TIDE-Lite.git
%cd TIDE-Lite
!pip install -q -r requirements.txt
print("✅ TIDE-Lite ready! GPU:", !nvidia-smi --query-gpu=name --format=csv,noheader)
```

## Complete Pipeline on Colab

### Cell 1: Environment Setup
```python
# Check GPU availability and setup environment
import torch
import os

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU Available: {gpu_name}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detected. Training will be slower.")

# Clone repository
!git clone https://github.com/yourusername/TIDE-Lite.git
%cd TIDE-Lite

# Install dependencies silently
!pip install -q -r requirements.txt

# Import and verify
import sys
sys.path.append('/content/TIDE-Lite/src')
from tide_lite.models import TIDELite
print("✅ Setup complete!")
```

### Cell 2: Configuration
```python
# Set up paths and configuration
RESULTS_DIR = "/content/TIDE-Lite/results/colab_run"
CONFIG = {
    "batch_size": 32,
    "num_epochs": 3,
    "learning_rate": 5e-5,
    "temporal_weight": 0.1,
    "use_amp": True  # Mixed precision for faster training
}

# Create results directory
!mkdir -p {RESULTS_DIR}
print(f"Results will be saved to: {RESULTS_DIR}")
```

### Cell 3: TRAIN - Training TIDE-Lite
```python
# Train the model (takes ~10 minutes on T4 GPU)
!python -m tide_lite.cli.tide train \
    --config configs/defaults.yaml \
    --output-dir {RESULTS_DIR} \
    --batch-size {CONFIG['batch_size']} \
    --num-epochs {CONFIG['num_epochs']} \
    --learning-rate {CONFIG['learning_rate']} \
    --temporal-weight {CONFIG['temporal_weight']} \
    --use-amp

# Check if training completed
import os
checkpoint_path = f"{RESULTS_DIR}/checkpoints/final.pt"
if os.path.exists(checkpoint_path):
    print(f"✅ Training complete! Model saved to: {checkpoint_path}")
else:
    print("❌ Training failed. Check logs above.")
```

### Cell 4: EVAL-STSB - STS-B Evaluation
```python
# Evaluate on STS-B benchmark
!python -m tide_lite.cli.tide eval-stsb \
    --model-path {RESULTS_DIR}/checkpoints/final.pt \
    --output-dir {RESULTS_DIR}/eval_stsb \
    --split test \
    --compare-baseline

# Display results
import json
with open(f"{RESULTS_DIR}/eval_stsb/results.json", "r") as f:
    stsb_results = json.load(f)
print(f"STS-B Spearman: {stsb_results['spearman']:.3f}")
print(f"Baseline: {stsb_results.get('baseline_spearman', 'N/A')}")
```

### Cell 5: EVAL-QUORA - Quora Retrieval  
```python
# Evaluate on Quora duplicate pairs
!python -m tide_lite.cli.tide eval-quora \
    --model-path {RESULTS_DIR}/checkpoints/final.pt \
    --output-dir {RESULTS_DIR}/eval_quora \
    --max-corpus 5000 \
    --max-queries 500 \
    --batch-size 128

# Display results
with open(f"{RESULTS_DIR}/eval_quora/results.json", "r") as f:
    quora_results = json.load(f)
print(f"Quora nDCG@10: {quora_results['ndcg@10']:.3f}")
```

### Cell 6: EVAL-TEMPORAL - Temporal Evaluation
```python
# Evaluate temporal capabilities
!python -m tide_lite.cli.tide eval-temporal \
    --model-path {RESULTS_DIR}/checkpoints/final.pt \
    --output-dir {RESULTS_DIR}/eval_temporal \
    --time-window-days 7 \
    --max-samples 1000 \
    --compare-baseline

# Display results
with open(f"{RESULTS_DIR}/eval_temporal/results.json", "r") as f:
    temporal_results = json.load(f)
print(f"Temporal Accuracy: {temporal_results['accuracy']:.3f}")
print(f"Baseline: {temporal_results.get('baseline_accuracy', 'N/A')}")
```

### Cell 7: BENCH-ALL - Complete Benchmark Suite
```python
# Run all benchmarks at once
!python -m tide_lite.cli.tide bench-all \
    --model-path {RESULTS_DIR}/checkpoints/final.pt \
    --output-dir {RESULTS_DIR}/bench_all

print("✅ All benchmarks complete!")
```

### Cell 8: AGGREGATE - Collect Results
```python
# Aggregate all evaluation results
!python -m tide_lite.cli.tide aggregate \
    --results-dir {RESULTS_DIR} \
    --output {RESULTS_DIR}/summary.json

# Display summary
with open(f"{RESULTS_DIR}/summary.json", "r") as f:
    summary = json.load(f)

print("📊 Performance Summary:")
print("-" * 40)
for metric, value in summary.items():
    if isinstance(value, (int, float)):
        print(f"{metric:20s}: {value:.3f}")
```

### Cell 9: REPORT - Generate Final Report
```python
# Generate comprehensive report with visualizations
!python -m tide_lite.cli.tide report \
    --input {RESULTS_DIR}/summary.json \
    --output-dir {RESULTS_DIR}/report

# Display report location
print(f"📄 Report generated: {RESULTS_DIR}/report/")
!ls -la {RESULTS_DIR}/report/
```

### Cell 10: Visualize Results
```python
# Create visualizations
import matplotlib.pyplot as plt
import pandas as pd
import json

# Load results
with open(f"{RESULTS_DIR}/summary.json", "r") as f:
    results = json.load(f)

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# STS-B Comparison
axes[0].bar(['Baseline', 'TIDE-Lite'], 
           [0.82, results.get('stsb_spearman', 0)])
axes[0].set_title('STS-B Performance')
axes[0].set_ylabel('Spearman ρ')
axes[0].set_ylim([0.7, 0.9])

# Quora Retrieval
axes[1].bar(['Baseline', 'TIDE-Lite'],
           [0.68, results.get('quora_ndcg10', 0)])
axes[1].set_title('Quora Retrieval')
axes[1].set_ylabel('nDCG@10')
axes[1].set_ylim([0.6, 0.8])

# Temporal Accuracy
axes[2].bar(['Baseline', 'TIDE-Lite'],
           [0.50, results.get('temporal_accuracy', 0)])
axes[2].set_title('Temporal Reasoning')
axes[2].set_ylabel('Accuracy')
axes[2].set_ylim([0.4, 1.0])

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/performance_comparison.png", dpi=150)
plt.show()
print(f"📊 Plot saved to: {RESULTS_DIR}/performance_comparison.png")
```

### Cell 11: Download Results
```python
# Zip and download results
import shutil
from google.colab import files

# Create archive
archive_name = "tide_lite_results"
shutil.make_archive(archive_name, 'zip', RESULTS_DIR)

# Download
files.download(f"{archive_name}.zip")
print(f"📥 Results downloaded as {archive_name}.zip")
```

## Memory-Efficient Settings for Colab

### For Colab Free Tier (Limited RAM)
```python
# Reduced memory configuration
!python -m tide_lite.cli.tide train \
    --batch-size 16 \
    --eval-batch-size 32 \
    --max-seq-length 64 \
    --gradient-accumulation-steps 2 \
    --num-workers 1 \
    --no-pin-memory
```

### Handle Session Timeouts
```python
# Auto-save checkpoints to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set checkpoint directory to Drive
CHECKPOINT_DIR = "/content/drive/MyDrive/tide_lite_checkpoints"
!python -m tide_lite.cli.tide train \
    --output-dir {CHECKPOINT_DIR} \
    --save-every-n-steps 100
```

## Interactive Demo

### Cell: Try Your Own Sentences
```python
# Load trained model
from tide_lite.models import TIDELite
import torch
from datetime import datetime

# Load model
model = TIDELite.from_pretrained(f"{RESULTS_DIR}/checkpoints/final.pt")
model.eval()

# Interactive encoding function
def encode_with_time(text, date_str=None):
    """Encode text with optional temporal context"""
    if date_str:
        timestamp = datetime.strptime(date_str, "%Y-%m-%d").timestamp()
    else:
        timestamp = datetime.now().timestamp()
    
    with torch.no_grad():
        embedding = model.encode(text, timestamp)
    return embedding

# Example usage
text1 = "The president announced new policies"
text2 = "Stock markets react to news"

# Encode with different timestamps
emb_2020 = encode_with_time(text1, "2020-01-15")
emb_2024 = encode_with_time(text1, "2024-01-15")

# Compute similarity change
from torch.nn.functional import cosine_similarity
similarity = cosine_similarity(emb_2020.unsqueeze(0), emb_2024.unsqueeze(0))
print(f"Temporal drift: {(1 - similarity.item()) * 100:.1f}%")

# Interactive widget (if using Colab)
#@title Enter your text and date
input_text = "Climate change impacts" #@param {type:"string"}
input_date = "2024-01-15" #@param {type:"date"}

embedding = encode_with_time(input_text, input_date)
print(f"✅ Encoded '{input_text[:50]}...' for date {input_date}")
print(f"   Embedding shape: {embedding.shape}")
```

## Colab Pro Features

### Enable High-RAM Runtime
```python
# Check runtime type
import psutil
ram_gb = psutil.virtual_memory().total / (1024**3)
print(f"RAM: {ram_gb:.1f} GB")

if ram_gb > 25:
    print("✅ High-RAM runtime enabled")
    BATCH_SIZE = 128
else:
    print("⚠️ Standard runtime - using smaller batch size")
    BATCH_SIZE = 32
```

### Use TPU (Experimental)
```python
# Check for TPU availability
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print(f"✅ TPU available: {device}")
except:
    print("❌ TPU not available, using GPU/CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Troubleshooting Colab Issues

### Issue 1: Runtime Disconnected
```python
# Keep session alive
import time
from IPython.display import display, Javascript

def keep_alive():
    while True:
        display(Javascript('google.colab.output.clear()'))
        time.sleep(60 * 10)  # Every 10 minutes

# Run in background (careful - infinite loop)
# import threading
# threading.Thread(target=keep_alive, daemon=True).start()
```

### Issue 2: Out of Memory
```python
# Clear GPU memory
import torch
import gc

torch.cuda.empty_cache()
gc.collect()

# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Issue 3: Slow Data Loading
```python
# Pre-download data to Colab instance
!mkdir -p /content/data_cache
%env HF_DATASETS_CACHE=/content/data_cache
%env SENTENCE_TRANSFORMERS_HOME=/content/model_cache

# Prefetch models
from sentence_transformers import SentenceTransformer
_ = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ Models cached locally")
```

## Saving and Loading Results

### Save to Google Drive
```python
from google.colab import drive
import shutil

# Mount Drive
drive.mount('/content/drive')

# Copy results to Drive
drive_dir = "/content/drive/MyDrive/TIDE_Lite_Results"
!mkdir -p {drive_dir}
shutil.copytree(RESULTS_DIR, f"{drive_dir}/run_{datetime.now():%Y%m%d_%H%M%S}")
print(f"✅ Results saved to Google Drive: {drive_dir}")
```

### Load Previous Run
```python
# List previous runs
import os
drive_results = "/content/drive/MyDrive/TIDE_Lite_Results"
if os.path.exists(drive_results):
    runs = os.listdir(drive_results)
    print("Previous runs:")
    for run in sorted(runs)[-5:]:  # Show last 5
        print(f"  - {run}")
    
    # Load specific run
    # last_run = sorted(runs)[-1]
    # !cp -r {drive_results}/{last_run} /content/loaded_results
```

## Complete Copy-Paste Script

```python
# ============================================
# COMPLETE TIDE-LITE PIPELINE FOR COLAB
# Just copy and run this entire cell
# ============================================

import os
import json
import torch
from datetime import datetime

# Setup
print("🚀 Starting TIDE-Lite Pipeline...")
!git clone -q https://github.com/yourusername/TIDE-Lite.git
%cd TIDE-Lite
!pip install -q -r requirements.txt

# Configuration
RESULTS = "/content/results"
!mkdir -p {RESULTS}

# 1. TRAIN
print("\n📚 Training...")
!python -m tide_lite.cli.tide train \
    --output-dir {RESULTS} \
    --num-epochs 3 \
    --batch-size 32 \
    --temporal-weight 0.1 \
    --use-amp \
    > {RESULTS}/train.log 2>&1

# 2. EVAL-STSB
print("📊 Evaluating STS-B...")
!python -m tide_lite.cli.tide eval-stsb \
    --model-path {RESULTS}/checkpoints/final.pt \
    --output-dir {RESULTS}/eval_stsb \
    --compare-baseline \
    > {RESULTS}/stsb.log 2>&1

# 3. EVAL-QUORA
print("🔍 Evaluating Quora...")
!python -m tide_lite.cli.tide eval-quora \
    --model-path {RESULTS}/checkpoints/final.pt \
    --output-dir {RESULTS}/eval_quora \
    --max-corpus 5000 \
    > {RESULTS}/quora.log 2>&1

# 4. EVAL-TEMPORAL
print("⏰ Evaluating Temporal...")
!python -m tide_lite.cli.tide eval-temporal \
    --model-path {RESULTS}/checkpoints/final.pt \
    --output-dir {RESULTS}/eval_temporal \
    > {RESULTS}/temporal.log 2>&1

# 5. BENCH-ALL
print("🎯 Running all benchmarks...")
!python -m tide_lite.cli.tide bench-all \
    --model-path {RESULTS}/checkpoints/final.pt \
    --output-dir {RESULTS}/bench_all \
    > {RESULTS}/bench.log 2>&1

# 6. AGGREGATE
print("📈 Aggregating results...")
!python -m tide_lite.cli.tide aggregate \
    --results-dir {RESULTS} \
    --output {RESULTS}/summary.json \
    > {RESULTS}/aggregate.log 2>&1

# 7. REPORT
print("📄 Generating report...")
!python -m tide_lite.cli.tide report \
    --input {RESULTS}/summary.json \
    --output-dir {RESULTS}/report \
    > {RESULTS}/report.log 2>&1

# Display Summary
print("\n" + "="*50)
print("✅ PIPELINE COMPLETE!")
print("="*50)

with open(f"{RESULTS}/summary.json", "r") as f:
    summary = json.load(f)
    print("\n📊 Results:")
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}")

# Download results
from google.colab import files
!cd {RESULTS} && zip -q -r tide_results.zip *
files.download(f"{RESULTS}/tide_results.zip")
print("\n📥 Results downloaded as tide_results.zip")
```

## Next Steps

1. **Experiment with hyperparameters** in the configuration cells
2. **Try different base encoders** (see available models in configs)
3. **Upload your own data** for fine-tuning
4. **Share your notebook** with the community

## Resources

- [Example Notebooks](https://github.com/yourusername/TIDE-Lite/tree/main/notebooks)
- [Colab Tips & Tricks](https://colab.research.google.com/notebooks/pro.ipynb)
- [GPU Runtime Guide](https://colab.research.google.com/notebooks/gpu.ipynb)

## Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/TIDE-Lite/issues)
- **Discord:** [Join our server](https://discord.gg/tide-lite)
- **Email:** colab-support@tide-lite.ai
