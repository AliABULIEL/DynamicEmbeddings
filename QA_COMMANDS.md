# TIDE-Lite Final QA Commands

## 🚀 Quick Setup & Run Commands

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/yourusername/DynamicEmbeddings.git
cd DynamicEmbeddings

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. CPU Smoke Test (2-5 min)
```bash
# Quick smoke test with minimal data
python -m src.tide_lite.cli.train_cli \
    --batch-size 8 \
    --num-epochs 1 \
    --dry-run \
    --output-dir results/smoke

# Evaluate
python -m src.tide_lite.cli.eval_stsb_cli \
    --model-path results/smoke \
    --dry-run

# Generate plots
python scripts/plot.py --dry-run --output-dir outputs
```

### 3. GPU Short Run (15 min)
```bash
# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Train with mixed precision
python -m src.tide_lite.cli.train_cli \
    --batch-size 32 \
    --num-epochs 3 \
    --use-amp \
    --output-dir results/gpu_short

# Full evaluation
python -m src.tide_lite.cli.eval_stsb_cli \
    --model-path results/gpu_short \
    --batch-size 64

# Generate plots and report
python scripts/plot.py \
    --input results/gpu_short/metrics_all.json \
    --output-dir outputs
```

### 4. Google Colab Instructions
```python
# Cell 1: Setup
!git clone https://github.com/yourusername/DynamicEmbeddings.git
%cd DynamicEmbeddings
!pip install -q -r requirements.txt

# Cell 2: Verify GPU
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Cell 3: Train
!python -m src.tide_lite.cli.train_cli \
    --batch-size 48 \
    --num-epochs 5 \
    --use-amp \
    --output-dir results/colab

# Cell 4: Evaluate
!python -m src.tide_lite.cli.eval_stsb_cli --model-path results/colab
!python scripts/plot.py --output-dir outputs

# Cell 5: Display Results
from IPython.display import Image, display
import os

if os.path.exists('outputs/fig_score_vs_dim.png'):
    display(Image('outputs/fig_score_vs_dim.png'))
if os.path.exists('outputs/fig_latency_vs_dim.png'):
    display(Image('outputs/fig_latency_vs_dim.png'))
```

## 📊 Expected Artifacts

After running the pipeline, you should see:

### Directory Structure
```
outputs/
├── metrics_all.csv          # Tabular results (all experiments)
├── metrics_all.json         # Complete metrics dump
├── fig_score_vs_dim.png    # Performance vs MLP dimension
├── fig_latency_vs_dim.png  # Latency comparison
├── fig_temporal_ablation.png # Temporal weight analysis
└── REPORT.md               # Auto-generated report

results/[run_name]/
├── config_used.json        # Training configuration
├── metrics_train.json      # Training metrics
├── checkpoints/
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_final.pt
│   └── tide_lite.pt
└── training.log            # Detailed logs
```

### Expected Metrics (Approximate)

| Metric | Baseline | TIDE-Lite | Notes |
|--------|----------|-----------|-------|
| **STS-B Spearman** | 0.77-0.78 | 0.78-0.79 | +1-2% improvement |
| **Extra Parameters** | 0 | 53,200 | Only 0.23% of base model |
| **Latency (GPU)** | 6-7ms | 8-9ms | <2ms overhead |
| **Memory Usage** | ~450MB | ~452MB | Negligible increase |

## 🔧 Troubleshooting

### Common Issues & Solutions

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train.py --batch-size 16
   ```

2. **Import Errors**
   ```bash
   # Ensure correct path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **Slow Training**
   ```bash
   # Enable AMP and increase workers
   python train.py --use-amp --num-workers 4
   ```

4. **Missing Dependencies**
   ```bash
   pip install scipy  # If Spearman correlation fails
   pip install matplotlib seaborn  # If plotting fails
   ```

## ✅ Validation Checklist

Run these commands to verify everything works:

```bash
# 1. Test imports
python -c "from src.tide_lite.models import TIDELite; print('✓ Models import')"
python -c "from src.tide_lite.train import TIDETrainer; print('✓ Training import')"
python -c "from src.tide_lite.eval import evaluate_stsb; print('✓ Eval import')"

# 2. Run minimal tests
python -m pytest tests/test_utils.py -v
python -m pytest tests/test_train_smoke.py::test_model_initialization -v

# 3. Check CLI tools
python -m src.tide_lite.cli.train_cli --help
python -m src.tide_lite.cli.eval_stsb_cli --help
python scripts/plot.py --help

# 4. Dry run pipeline
bash scripts/run_quick.sh cpu
```

## 📈 Performance Benchmarks

### Training Time (Estimated)
- **CPU (4 cores)**: 45-60 min for 3 epochs
- **T4 GPU**: 8-10 min for 3 epochs  
- **V100 GPU**: 5-6 min for 3 epochs
- **A100 GPU**: 3-4 min for 3 epochs

### Inference Throughput
- **CPU**: ~100-150 samples/sec
- **T4 GPU**: ~800-1000 samples/sec
- **V100 GPU**: ~1500-2000 samples/sec

## 🎯 Next Steps

1. **Run Full Experiments**
   ```bash
   # Temporal weight ablation
   for lambda in 0.0 0.1 0.2; do
       python train.py --temporal-weight $lambda
   done
   ```

2. **Try Different Encoders**
   - `sentence-transformers/all-mpnet-base-v2`
   - `sentence-transformers/all-MiniLM-L12-v2`
   - `BAAI/bge-small-en-v1.5`

3. **Evaluate on More Tasks**
   - Quora duplicate detection
   - Temporal QA datasets
   - Custom temporal benchmarks

## 📝 Citation

If you use TIDE-Lite, please cite:
```bibtex
@article{tide-lite-2024,
  title={TIDE-Lite: Efficient Temporal Adaptation for Sentence Embeddings},
  author={Your Name},
  year={2024}
}
```

---
*Last updated: [Current Date]*
*Repository: https://github.com/yourusername/DynamicEmbeddings*
