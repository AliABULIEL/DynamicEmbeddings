# TIDE-Lite Colab Runbook

This runbook provides exact commands to train and evaluate TIDE-Lite in Google Colab.
Copy and paste these commands sequentially.

## Expected Runtime
- **Total time**: ~2-4 hours on Colab T4 GPU
- **Training**: ~1 hour for 3 epochs
- **Evaluation**: ~30 min per model
- **Full benchmark**: ~2-3 hours

## Step 1: Environment Setup

```bash
# Clone repository
!git clone https://github.com/yourusername/DynamicEmbeddings.git
%cd DynamicEmbeddings

# Install dependencies
!pip install -q -r requirements.txt
!pip install -q faiss-gpu  # For GPU, or faiss-cpu for CPU

# Verify installation
!python -c "import torch; print(f'PyTorch: {torch.__version__}')"
!python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
!python -c "import faiss; print('FAISS installed')"
```

## Step 2: Download Real Datasets

```bash
# STS-B will auto-download via Hugging Face datasets
# Quora will auto-download via Hugging Face datasets

# For TimeQA (if available) - otherwise uses TempLAMA
!mkdir -p data/timeqa

# TempLAMA auto-downloads as fallback
# If you have TimeQA access, download and extract:
# !wget -O data/timeqa/train.json <your-timeqa-url>
# !wget -O data/timeqa/dev.json <your-timeqa-url>
# !wget -O data/timeqa/test.json <your-timeqa-url>

# Set environment variable for TimeQA (optional)
import os
os.environ['TIMEQA_DATA_DIR'] = './data/timeqa'
```

## Step 3: Train TIDE-Lite

```bash
# Quick training (1 epoch for testing)
!python -m tide_lite.cli.tide train \
    --output-dir results/quick-test \
    --num-epochs 1 \
    --batch-size 32 \
    --run

# Full training (3 epochs)
!python -m tide_lite.cli.tide train \
    --output-dir results/full-training \
    --num-epochs 3 \
    --batch-size 32 \
    --learning-rate 2e-5 \
    --consistency-weight 0.1 \
    --run

# Monitor training
!tail -f results/full-training/metrics_train.json
```

### Recovery from Interruption
```bash
# Resume from checkpoint if training interrupted
!python -m tide_lite.cli.train \
    --output-dir results/full-training \
    --resume-from results/full-training/checkpoints/last_checkpoint.pt \
    --run
```

## Step 4: Evaluate TIDE-Lite

```bash
# Evaluate on STS-B
!python -m tide_lite.cli.tide eval-stsb \
    --model results/full-training/checkpoints/best_model.pt \
    --type tide_lite \
    --output-dir results/evaluation \
    --run

# Evaluate on Quora (with limited data for speed)
!python -m tide_lite.cli.tide eval-quora \
    --model results/full-training/checkpoints/best_model.pt \
    --type tide_lite \
    --output-dir results/evaluation \
    --max-corpus 5000 \
    --max-queries 500 \
    --run

# Evaluate temporal understanding
!python -m tide_lite.cli.tide eval-temporal \
    --model results/full-training/checkpoints/best_model.pt \
    --type tide_lite \
    --output-dir results/evaluation \
    --max-samples 1000 \
    --run
```

## Step 5: Benchmark Baselines

```bash
# MiniLM baseline (fastest)
!python -m tide_lite.cli.tide bench-all \
    --model minilm \
    --type baseline \
    --output-dir results/baselines \
    --run

# E5-Base baseline (optional, slower)
!python -m tide_lite.cli.tide bench-all \
    --model e5-base \
    --type baseline \
    --output-dir results/baselines \
    --run

# BGE-Base baseline (optional, slower)
!python -m tide_lite.cli.tide bench-all \
    --model bge-base \
    --type baseline \
    --output-dir results/baselines \
    --run
```

## Step 6: Small Ablation Study

```bash
# Quick ablation (2x2x2 grid = 8 configs)
!python -m tide_lite.cli.tide ablation \
    --time-mlp-hidden 64,128 \
    --consistency-weight 0.05,0.1 \
    --time-encoding sinusoidal,learnable \
    --output-dir results/ablation \
    --num-epochs 1 \
    --run
```

### Reduce if Memory Limited
```bash
# Minimal ablation (2 configs only)
!python -m tide_lite.cli.tide ablation \
    --time-mlp-hidden 128 \
    --consistency-weight 0.05,0.1 \
    --time-encoding sinusoidal \
    --output-dir results/ablation-mini \
    --num-epochs 1 \
    --run
```

## Step 7: Aggregate Results

```bash
# Aggregate all metrics
!python -m tide_lite.cli.tide aggregate \
    --results-dir results/ \
    --output results/summary.json \
    --run

# View summary
!cat results/summary.json | python -m json.tool | head -50
```

## Step 8: Generate Report

```bash
# Generate plots
!python -m tide_lite.cli.plots \
    --input results/summary.json \
    --output-dir results/figures \
    --run

# Generate markdown report
!python -m tide_lite.cli.tide report \
    --input results/summary.json \
    --output-dir reports/ \
    --run

# View report
!head -100 reports/report.md
```

## Step 9: Download Results

```python
# Zip results for download
import shutil
shutil.make_archive('tide_lite_results', 'zip', 'results')

# Download link
from google.colab import files
files.download('tide_lite_results.zip')
```

## Optimization Tips

### If Running Out of Memory
```bash
# Reduce batch size
--batch-size 16

# Limit dataset size
--max-corpus 1000
--max-queries 100

# Use CPU for small experiments
export CUDA_VISIBLE_DEVICES=""
```

### Speed Up Training
```bash
# Reduce epochs
--num-epochs 1

# Use smaller validation set
--val-check-interval 0.5

# Skip some evaluations
# Only run STS-B, skip Quora/Temporal
```

### Debug Mode
```bash
# Tiny data for debugging
!python -m tide_lite.cli.tide train \
    --output-dir results/debug \
    --num-epochs 1 \
    --batch-size 8 \
    --max-train-samples 100 \
    --run
```

## Expected Outputs

After completing all steps, you should have:
```
results/
├── full-training/
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── final_model.pt
│   └── metrics_train.json
├── evaluation/
│   ├── metrics_stsb_*.json
│   ├── metrics_quora_*.json
│   └── metrics_temporal_*.json
├── baselines/
│   └── metrics_*_minilm.json
├── summary.json
├── summary.csv
└── figures/
    ├── model_comparison.png
    ├── ablation_heatmap.png
    └── latency_vs_quality.png

reports/
└── report.md
```

## Verification

```python
# Verify key metrics
import json

with open('results/summary.json') as f:
    summary = json.load(f)

print("Models evaluated:", list(summary['models'].keys()))
print("\nBest performers:")
for metric, model in summary['best_models'].items():
    print(f"  {metric}: {model}")
```

## Common Issues & Solutions

### Issue: FAISS not found
```bash
!pip install faiss-cpu  # If GPU not available
```

### Issue: Out of memory
```bash
# Restart runtime and use smaller batch
Runtime > Restart runtime
# Then use batch_size=8
```

### Issue: Slow training
```bash
# Check GPU is enabled
Runtime > Change runtime type > GPU
```

### Issue: Import errors
```bash
# Reinstall package in editable mode
!pip install -e .
```

## Notes

- All datasets are real (STS-B from GLUE, Quora from HF, TimeQA/TempLAMA)
- No synthetic data is used
- Results are saved incrementally (can resume if interrupted)
- Dry-run mode available for all commands (omit --run flag)

**Total expected runtime: 2-4 hours on Colab T4 GPU**
