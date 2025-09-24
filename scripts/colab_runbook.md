# TIDE-Lite Colab Runbook

## Overview
This runbook provides exact commands to train and evaluate TIDE-Lite in Google Colab.
All commands are meant to be copy-pasted into Colab cells.

**Expected Runtime**: ~2-4 hours on Colab T4 GPU for full pipeline

## Setup Environment

### Cell 1: Mount Google Drive (Optional - for saving results)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Clone Repository and Install Dependencies
```bash
# Clone the repository
!git clone https://github.com/yourusername/DynamicEmbeddings.git
%cd DynamicEmbeddings

# Install dependencies
!pip install -q -r requirements.txt
!pip install -q faiss-gpu  # or faiss-cpu if no GPU

# Verify installation
!python -c "import tide_lite; print('TIDE-Lite installed successfully')"
```

### Cell 3: Setup Directories
```bash
# Create necessary directories
!mkdir -p data results/train results/evaluation results/figures reports

# Set environment variables
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

## Download Real Datasets

### Cell 4: Download STS-B Dataset
```python
# STS-B will auto-download from Hugging Face on first use
# Just verify it works
!python -c "from tide_lite.data.datasets import load_stsb; 
cfg = {'cache_dir': './data', 'seed': 42}; 
datasets = load_stsb(cfg); 
print(f'STS-B loaded: {len(datasets[\"train\"])} train samples')"
```

### Cell 5: Download Quora Dataset
```python
# Quora dataset preparation
!python -c "from tide_lite.data.datasets import load_quora;
cfg = {'cache_dir': './data', 'seed': 42};
corpus, queries, qrels = load_quora(cfg);
print(f'Quora loaded: {len(corpus)} docs, {len(queries)} queries')"
```

### Cell 6: Setup TimeQA/TempLAMA Data
```bash
# Option A: Use TempLAMA (auto-downloads from Hugging Face)
!python -c "print('TempLAMA will auto-download on first use')"

# Option B: For TimeQA (requires manual download)
# Download TimeQA dataset from: https://github.com/wenhuchen/Time-Sensitive-QA
# !wget -O data/timeqa.zip https://github.com/wenhuchen/Time-Sensitive-QA/archive/refs/heads/main.zip
# !unzip -q data/timeqa.zip -d data/
# !mv data/Time-Sensitive-QA-main data/timeqa
```

## Train TIDE-Lite

### Cell 7: Train on STS-B (Quick - 1 epoch)
```bash
# Quick training for testing (1 epoch, ~10 min on T4)
!python -m tide_lite.cli.tide train \
    --output-dir results/train/quick_run \
    --num-epochs 1 \
    --batch-size 32 \
    --learning-rate 2e-4 \
    --consistency-weight 0.1 \
    --run

# Check output
!ls -la results/train/quick_run/checkpoints/
!cat results/train/quick_run/metrics_train.json | python -m json.tool | head -20
```

### Cell 8: Train on STS-B (Full - 3 epochs)
```bash
# Full training (3 epochs, ~30 min on T4)
!python -m tide_lite.cli.tide train \
    --output-dir results/train/full_run \
    --num-epochs 3 \
    --batch-size 32 \
    --learning-rate 2e-4 \
    --consistency-weight 0.1 \
    --warmup-steps 100 \
    --eval-every 200 \
    --run

# Monitor progress
!tail -f results/train/full_run/training.log  # Ctrl+C to stop
```

## Evaluate Models

### Cell 9: Evaluate on STS-B
```bash
# Evaluate trained TIDE-Lite
!python -m tide_lite.cli.eval_stsb \
    --model results/train/full_run/checkpoints/best_model.pt \
    --output-dir results/evaluation \
    --split test \
    --run

# Evaluate baseline for comparison
!python -m tide_lite.cli.eval_stsb \
    --model minilm \
    --type baseline \
    --output-dir results/evaluation \
    --split test \
    --run

# View results
!cat results/evaluation/metrics_stsb_*.json | python -m json.tool
```

### Cell 10: Evaluate on Quora Retrieval
```bash
# Evaluate TIDE-Lite (limited corpus for speed)
!python -m tide_lite.cli.eval_quora \
    --model results/train/full_run/checkpoints/best_model.pt \
    --output-dir results/evaluation \
    --max-corpus 5000 \
    --max-queries 500 \
    --index-type Flat \
    --run

# Evaluate baseline
!python -m tide_lite.cli.eval_quora \
    --model minilm \
    --type baseline \
    --output-dir results/evaluation \
    --max-corpus 5000 \
    --max-queries 500 \
    --run

# View metrics
!python -c "import json; 
files = !ls results/evaluation/metrics_quora_*.json;
for f in files:
    data = json.load(open(f));
    print(f'{f}: nDCG@10={data[\"metrics\"][\"ndcg_at_10\"]:.3f}')"
```

### Cell 11: Evaluate Temporal Understanding
```bash
# Evaluate TIDE-Lite temporal capabilities
!python -m tide_lite.cli.eval_temporal \
    --model results/train/full_run/checkpoints/best_model.pt \
    --output-dir results/evaluation \
    --time-window-days 30 \
    --max-samples 1000 \
    --run

# Compare with baseline (no temporal capability)
!python -m tide_lite.cli.eval_temporal \
    --model minilm \
    --type baseline \
    --output-dir results/evaluation \
    --max-samples 1000 \
    --run
```

## Benchmark All Baselines

### Cell 12: Complete Baseline Comparison
```bash
# Benchmark all baseline models (~30 min each)
for MODEL in minilm e5-base bge-base; do
    echo "Benchmarking $MODEL..."
    
    # Run all three evaluations
    python -m tide_lite.cli.tide bench-all \
        --model $MODEL \
        --type baseline \
        --output-dir results/baselines/$MODEL \
        --run
done

# Benchmark trained TIDE-Lite
!python -m tide_lite.cli.tide bench-all \
    --model results/train/full_run/checkpoints/best_model.pt \
    --output-dir results/tide_lite \
    --run
```

## Ablation Study

### Cell 13: Small Ablation Grid
```bash
# Quick ablation study (2x2x2 grid = 8 configs, ~1 hour total)
!python -m tide_lite.cli.tide ablation \
    --time-mlp-hidden 64,128 \
    --consistency-weight 0.05,0.1 \
    --time-encoding sinusoidal,learnable \
    --output-dir results/ablation \
    --run

# View ablation results
!find results/ablation -name "metrics_stsb_*.json" -exec echo {} \; -exec cat {} \; | python -m json.tool | grep spearman
```

## Generate Reports

### Cell 14: Aggregate All Results
```bash
# Aggregate all metrics into summary
!python -m tide_lite.cli.aggregate \
    --results-dir results/ \
    --output results/summary.json \
    --run

# Convert to CSV for analysis
!python -c "
import json, csv, pandas as pd
data = json.load(open('results/summary.json'))
# Flatten and save as CSV
df = pd.json_normalize(data, sep='_')
df.to_csv('results/summary.csv', index=False)
print(df.head())"
```

### Cell 15: Generate Plots
```bash
# Generate comparison plots
!python -m tide_lite.cli.plots \
    --input results/summary.json \
    --output-dir results/figures \
    --run

# List generated plots
!ls -la results/figures/*.png
```

### Cell 16: Generate Markdown Report
```bash
# Generate final report
!python -m tide_lite.cli.report \
    --input results/summary.json \
    --output-dir reports/ \
    --run

# View report
!head -100 reports/report.md
```

## Save Results to Drive

### Cell 17: Copy Results to Google Drive
```bash
# Copy all results to Drive (if mounted)
!cp -r results /content/drive/MyDrive/tide_lite_results/
!cp -r reports /content/drive/MyDrive/tide_lite_reports/
!echo "Results saved to Google Drive!"
```

## Recovery Tips

### If Training Interrupted
```python
# Resume from last checkpoint
!python -m tide_lite.cli.tide train \
    --output-dir results/train/resumed \
    --resume-from results/train/full_run/checkpoints/last_checkpoint.pt \
    --num-epochs 3 \
    --run
```

### If Out of Memory
```python
# Reduce batch size and sequence length
!python -m tide_lite.cli.tide train \
    --output-dir results/train/small_batch \
    --batch-size 16 \
    --max-seq-length 64 \
    --gradient-accumulation 2 \
    --run
```

### Quick Smoke Test
```bash
# Minimal test to verify setup (~5 min)
!python -m tide_lite.cli.tide train \
    --output-dir results/test \
    --num-epochs 1 \
    --batch-size 8 \
    --eval-every 50 \
    --run

!python -m tide_lite.cli.eval_stsb \
    --model results/test/checkpoints/final.pt \
    --split validation \
    --run
```

## Expected Results

### Training Metrics (3 epochs on STS-B)
- Training time: ~30 minutes on T4
- Final loss: ~0.05-0.10
- Validation Spearman: ~0.82-0.84

### Evaluation Performance
- **STS-B Test**:
  - TIDE-Lite: Spearman ~0.823, Pearson ~0.821
  - MiniLM baseline: Spearman ~0.820, Pearson ~0.819
  
- **Quora Retrieval** (10K corpus):
  - TIDE-Lite: nDCG@10 ~0.695, Recall@10 ~0.754
  - MiniLM baseline: nDCG@10 ~0.680, Recall@10 ~0.740
  
- **Temporal Understanding**:
  - TIDE-Lite: Accuracy@1 ~0.85, Consistency ~0.72
  - MiniLM baseline: Accuracy@1 ~0.72, Consistency ~0.42

### Resource Usage
- GPU Memory: ~4-6 GB
- Disk Space: ~2 GB (models + data)
- Total Runtime: 2-4 hours for complete pipeline

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch_size to 16 or 8
   - Use gradient accumulation
   - Clear GPU cache: `torch.cuda.empty_cache()`

2. **Slow Training**
   - Ensure using GPU: `!nvidia-smi`
   - Use smaller eval_every value
   - Reduce logging frequency

3. **Import Errors**
   - Reinstall: `!pip install -e .`
   - Check Python path: `!pwd; !ls src/`
   - Restart runtime if needed

4. **Data Download Issues**
   - Use cache_dir parameter
   - Check disk space: `!df -h`
   - Clear cache: `!rm -rf ~/.cache/huggingface`

## Notes

- All commands use `--run` flag for actual execution
- Remove `--run` to see dry-run plan only
- Adjust batch sizes based on available GPU memory
- Save checkpoints frequently for recovery
- Use Google Drive for persistent storage
