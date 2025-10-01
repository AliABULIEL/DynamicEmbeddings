# TIDE-Lite Training & Evaluation Flows

This document outlines the two main workflows for TIDE-Lite experiments:
1. **Smoke Test Flow** - Quick validation (1-2 minutes)
2. **Full Training Flow** - Complete training and evaluation (2-6 hours)

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify GPU (optional but recommended for full flow)
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🚀 Smoke Test Flow (Quick Validation)

**Purpose:** Quickly validate that all components work without full training.  
**Duration:** 1-2 minutes  
**GPU Required:** No

### Automatic Execution

```bash
chmod +x run_smoke_test.sh
./run_smoke_test.sh
```

### Manual Stage-by-Stage

```bash
# Stage 1: Run integration smoke test
python tests/test_integration_smoke.py

# Stage 2: Run unit tests
pytest tests/ -v

# Stage 3: Test training pipeline (dry-run, no actual training)
python -m src.tide_lite.cli.train \
    --config configs/tide_lite.yaml \
    --dry-run \
    --num-epochs 1 \
    --batch-size 4 \
    --max-samples 16

# Stage 4: Test model APIs
python -c "
from src.tide_lite.models import MiniLMBaseline, TIDELite, TIDELiteConfig
baseline = MiniLMBaseline()
texts = ['Test one', 'Test two']
emb = baseline.encode_texts(texts, batch_size=2)
print(f'Success! Shape: {emb.shape}')
"

# Stage 5: Test data loading (downloads if needed)
python -c "
from src.tide_lite.data.datasets import load_stsb
cfg = {'cache_dir': './data', 'max_samples': 10, 'seed': 42}
data = load_stsb(cfg)
print(f'Loaded {len(data[\"train\"])} samples')
"
```

### Expected Output
✅ All tests pass  
✅ Models initialize correctly  
✅ Data loads successfully  
✅ CLI parsers work  

---

## 🎯 Full Training Flow (Complete Pipeline)

**Purpose:** Train TIDE-Lite and evaluate on all benchmarks.  
**Duration:** 2-6 hours (depending on GPU)  
**GPU Required:** Strongly recommended

### Automatic Execution

```bash
chmod +x run_full_flow.sh
./run_full_flow.sh
```

### Manual Stage-by-Stage

#### Stage 1: Environment Setup
```bash
# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

# Set experiment name
export EXPERIMENT_NAME="tide_lite_$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"
```

#### Stage 2: Data Preparation
```bash
# Download and cache datasets
python -c "
from src.tide_lite.data.datasets import load_stsb, load_quora

# This downloads ~500MB on first run
stsb = load_stsb({'cache_dir': './data'})
print(f'STS-B: {len(stsb[\"train\"])} train samples')

corpus, queries, qrels = load_quora({'cache_dir': './data'})
print(f'Quora: {len(corpus)} documents')
"

# Optional: Setup temporal datasets
# Download TimeQA to ./data/timeqa/ or TempLAMA to ./data/templama/
```

#### Stage 3: Baseline Evaluation (Optional but Recommended)
```bash
# Evaluate MiniLM baseline for comparison
python -m src.tide_lite.cli.eval_stsb \
    --model-name "sentence-transformers/all-MiniLM-L6-v2" \
    --model-type baseline \
    --output-dir "${OUTPUT_DIR}/baselines/minilm" \
    --batch-size 128

python -m src.tide_lite.cli.eval_quora \
    --model-name "sentence-transformers/all-MiniLM-L6-v2" \
    --model-type baseline \
    --output-dir "${OUTPUT_DIR}/baselines/minilm" \
    --batch-size 128 \
    --top-k 1 5 10
```

#### Stage 4: TIDE-Lite Training (Main)
```bash
# Train TIDE-Lite model
python -m src.tide_lite.cli.train \
    --config configs/tide_lite.yaml \
    --output-dir "${OUTPUT_DIR}/tide_lite" \
    --num-epochs 5 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --warmup-steps 1000 \
    --eval-every 2000 \
    --save-every 2000 \
    --gradient-accumulation 2 \
    --mixed-precision \
    --seed 42 \
    --wandb \
    --wandb-project tide-lite

# Training outputs:
# - ${OUTPUT_DIR}/tide_lite/checkpoints/  # Intermediate checkpoints
# - ${OUTPUT_DIR}/tide_lite/final/        # Final model
# - ${OUTPUT_DIR}/tide_lite/logs/         # Training logs
```

#### Stage 5: TIDE-Lite Evaluation
```bash
MODEL_PATH="${OUTPUT_DIR}/tide_lite/final"

# Evaluate on STS-B (without temporal)
python -m src.tide_lite.cli.eval_stsb \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}/eval/stsb_baseline" \
    --batch-size 128

# Evaluate on STS-B (with temporal)
python -m src.tide_lite.cli.eval_stsb \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}/eval/stsb_temporal" \
    --use-temporal \
    --batch-size 128

# Evaluate on Quora (without temporal)
python -m src.tide_lite.cli.eval_quora \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}/eval/quora_baseline" \
    --batch-size 128 \
    --top-k 1 5 10 20

# Evaluate on Quora (with temporal)
python -m src.tide_lite.cli.eval_quora \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}/eval/quora_temporal" \
    --use-temporal \
    --batch-size 128 \
    --top-k 1 5 10 20

# Optional: Evaluate on temporal datasets
python -m src.tide_lite.cli.eval_temporal \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}/eval/temporal" \
    --dataset timeqa \
    --batch-size 128
```

#### Stage 6: Results Aggregation
```bash
# Aggregate all results
python -m src.tide_lite.cli.aggregate \
    --experiment-dir "${OUTPUT_DIR}" \
    --output-file "${OUTPUT_DIR}/results_summary.json"

# Generate plots
python -m src.tide_lite.cli.plots \
    --results-file "${OUTPUT_DIR}/results_summary.json" \
    --output-dir "${OUTPUT_DIR}/plots"

# Generate final report
python -m src.tide_lite.cli.report \
    --experiment-dir "${OUTPUT_DIR}" \
    --output-file "${OUTPUT_DIR}/final_report.md"
```

#### Stage 7: Compare Results
```bash
# View summary
cat "${OUTPUT_DIR}/final_report.md"

# Compare models
python -c "
import json
with open('${OUTPUT_DIR}/results_summary.json') as f:
    results = json.load(f)
    
print('STS-B Spearman Correlation:')
for model, score in results.get('stsb', {}).items():
    print(f'  {model}: {score:.4f}')
    
print('\nQuora Recall@10:')
for model, metrics in results.get('quora', {}).items():
    print(f'  {model}: {metrics.get(\"recall@10\", \"N/A\"):.4f}')
"
```

---

## 📊 Expected Results

### Smoke Test Flow
- ✅ All tests pass in ~1-2 minutes
- ✅ No training performed (dry-run only)
- ✅ Validates setup is correct

### Full Training Flow

#### Baseline (MiniLM-L6-v2)
- STS-B: ~0.82-0.84 Spearman correlation
- Quora: ~0.65-0.70 Recall@10

#### TIDE-Lite (After Training)
- STS-B (baseline): ~0.83-0.85 Spearman correlation
- STS-B (temporal): ~0.84-0.86 Spearman correlation
- Quora (baseline): ~0.66-0.71 Recall@10
- Quora (temporal): ~0.68-0.73 Recall@10
- Extra parameters: ~50K (vs 0 for baseline)

---

## 🔧 Customization Options

### Training Hyperparameters
```bash
--num-epochs 10          # More epochs for better convergence
--batch-size 64          # Larger batch if GPU memory allows
--learning-rate 5e-5     # Lower LR for stability
--warmup-steps 2000      # More warmup for larger datasets
--gradient-accumulation 4  # Simulate larger batches
```

### Evaluation Options
```bash
--max-corpus-size 10000  # Limit Quora corpus for speed
--use-temporal           # Enable temporal embeddings
--top-k 1 3 5 10 20 50  # Multiple k values for retrieval
```

### Data Options
```bash
--max-samples 1000       # Limit samples for testing
--cache-dir ./data       # Dataset cache location
--timeqa-data-dir ./data/timeqa  # TimeQA location
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--batch-size 16
--eval-batch-size 64

# Enable gradient accumulation
--gradient-accumulation 8

# Use mixed precision
--mixed-precision
```

### No GPU Available
```bash
# Use CPU (much slower)
--device cpu
--batch-size 8  # Smaller batches for CPU
```

### Missing Temporal Datasets
```bash
# Skip temporal evaluation
--skip-temporal

# Or download TimeQA/TempLAMA to ./data/
```

### Slow Data Loading
```bash
# Pre-download datasets
python -c "
from datasets import load_dataset
load_dataset('glue', 'stsb', cache_dir='./data')
load_dataset('quora', cache_dir='./data')
"
```

---

## 📈 Monitoring Training

### With Weights & Biases
```bash
# Login first
wandb login

# Training will log to WandB automatically with --wandb flag
# View at: https://wandb.ai/<your-username>/tide-lite
```

### Local Monitoring
```bash
# Watch training logs
tail -f ${OUTPUT_DIR}/tide_lite/logs/training.log

# Check GPU usage
watch nvidia-smi

# Monitor checkpoints
ls -la ${OUTPUT_DIR}/tide_lite/checkpoints/
```

---

## 🎓 Next Steps

1. **Run smoke test** to verify setup
2. **Run full training** with default settings
3. **Compare results** between baseline and TIDE-Lite
4. **Experiment with hyperparameters** for better performance
5. **Add temporal datasets** for temporal evaluation
6. **Try different base encoders** (E5, BGE, etc.)

---

## 📝 Notes

- First run downloads ~500MB of data
- Full training needs ~8GB GPU memory
- Results are saved to `outputs/` directory
- Use `--dry-run` flag to test without training
- Enable `--wandb` for experiment tracking
