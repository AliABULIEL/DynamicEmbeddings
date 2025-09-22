# TIDE-Lite: Temporally-Indexed Dynamic Embeddings (Lightweight)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab Ready](https://img.shields.io/badge/Colab-Ready-green.svg)](notebooks/tide_lite_colab.ipynb)

**One-line pitch:** *"We're taking a frozen MiniLM encoder and bolting on a tiny timestamp-aware adapter that learns to shift embeddings over time — giving us dynamic, temporally-consistent sentence embeddings at almost no extra cost."*

## 🚀 Quickstart

### 1️⃣ CPU Smoke Test (2-5 min)
```bash
# Clone and setup
git clone https://github.com/yourusername/DynamicEmbeddings.git
cd DynamicEmbeddings

# Create environment and install
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run smoke test (tiny dataset, 1 epoch)
python -m src.tide_lite.cli.train_cli \
    --batch-size 16 \
    --num-epochs 1 \
    --dry-run \
    --output-dir results/smoke_test

# Quick eval
python -m src.tide_lite.cli.eval_stsb_cli \
    --model-path results/smoke_test \
    --dry-run

# Generate plots
python scripts/plot.py --dry-run
```

### 2️⃣ Local GPU Short Run (15 min)
```bash
# Ensure CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Train with mixed precision
python -m src.tide_lite.cli.train_cli \
    --batch-size 32 \
    --num-epochs 3 \
    --use-amp \
    --output-dir results/gpu_run

# Full evaluation suite
python -m src.tide_lite.cli.eval_stsb_cli \
    --model-path results/gpu_run \
    --batch-size 64

# Generate all plots
python scripts/plot.py \
    --input results/gpu_run/summary.json \
    --output-dir outputs
```

### 3️⃣ Google Colab Full Run (≤45 min)
```python
# In Colab notebook:
!git clone https://github.com/yourusername/DynamicEmbeddings.git
%cd DynamicEmbeddings

# Install dependencies with pinned versions
!pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
!pip install transformers==4.44.0 sentence-transformers==3.0.0
!pip install -r requirements.txt

# Verify CUDA
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Run full training
!python -m src.tide_lite.cli.train_cli \
    --batch-size 48 \
    --num-epochs 5 \
    --use-amp \
    --output-dir results/colab_full

# Evaluate on all tasks
!python -m src.tide_lite.cli.eval_stsb_cli --model-path results/colab_full
!python -m src.tide_lite.cli.eval_quora_cli --model-path results/colab_full
!python -m src.tide_lite.cli.eval_temporal_cli --model-path results/colab_full

# Generate plots and display
!python scripts/plot.py --output-dir outputs

from IPython.display import Image, display
display(Image('outputs/fig_score_vs_dim.png'))
display(Image('outputs/fig_latency_vs_dim.png'))
display(Image('outputs/fig_temporal_ablation.png'))
```

---

## 🏗️ Architecture Overview

TIDE-Lite adds a lightweight temporal modulation layer (~53K params) to any frozen sentence encoder:

```
Text + Timestamp → Frozen Encoder → Base Embedding → Temporal Gating → Time-Aware Embedding
                                           ↑
                                    Temporal MLP (53K params)
```

### Key Components:
- **Frozen Base Encoder**: all-MiniLM-L6-v2 (22.7M params, unchanged)
- **Sinusoidal Time Encoding**: 32-dim representation of timestamps
- **Temporal MLP**: 32→128→384 transformation (~53K trainable params)
- **Gating Mechanism**: Element-wise modulation via sigmoid activation

---

## 📊 Expected Results

| Metric | Baseline | TIDE-Lite | Improvement |
|--------|----------|-----------|-------------|
| STS-B Spearman | 0.771 | 0.785 | +1.8% |
| Quora nDCG@10 | 0.682 | 0.694 | +1.8% |
| Temporal Consistency | 0.45 | 0.71 | +58% |
| Extra Parameters | 0 | 53K | +0.23% |
| Latency Overhead | 0ms | <2ms | minimal |

---

## 📁 Project Structure

```
DynamicEmbeddings/
├── src/tide_lite/
│   ├── models/         # TIDE-Lite architecture
│   ├── train/          # Training logic & losses
│   ├── eval/           # Evaluation metrics
│   ├── data/           # Dataset loaders
│   └── cli/            # Command-line tools
├── configs/            # Training configurations
├── scripts/
│   ├── run_quick.sh    # Quick experiment runner
│   └── plot.py         # Visualization generator
├── outputs/            # Results & plots
├── tests/              # Unit tests
└── notebooks/          # Jupyter/Colab demos
```

---

## 🎯 Training Your Own TIDE-Lite

### Basic Training
```python
from src.tide_lite.models import TIDELite, TIDELiteConfig
from src.tide_lite.train import TIDETrainer, TrainingConfig

# Configure model
config = TIDELiteConfig(
    encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    time_encoding_dim=32,
    mlp_hidden_dim=128,
    freeze_encoder=True
)

# Initialize model
model = TIDELite(config)

# Setup training
train_config = TrainingConfig(
    num_epochs=3,
    batch_size=32,
    learning_rate=5e-5,
    temporal_weight=0.1,  # λ for temporal consistency loss
    output_dir="results/my_run"
)

# Train
trainer = TIDETrainer(model, train_config)
trainer.train()
```

### Custom Dataset with Timestamps
```python
from torch.utils.data import Dataset

class TemporalTextDataset(Dataset):
    def __init__(self, texts, timestamps):
        self.texts = texts
        self.timestamps = timestamps  # Unix timestamps
    
    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "timestamp": self.timestamps[idx]
        }
```

---

## 🔬 Key Ablations

### Temporal Weight (λ)
```bash
for lambda in 0.0 0.05 0.1 0.2 0.5; do
    python -m src.tide_lite.cli.train_cli \
        --temporal-weight $lambda \
        --output-dir results/lambda_$lambda
done
```

### MLP Hidden Dimension
```bash
for hidden in 64 128 256; do
    python -m src.tide_lite.cli.train_cli \
        --mlp-hidden-dim $hidden \
        --output-dir results/mlp_$hidden
done
```

---

## 📈 Visualization

After training, generate comprehensive plots:

```bash
# Generate all plots
python scripts/plot.py \
    --input results/summary.json \
    --output-dir outputs \
    --format png

# Expected outputs:
# outputs/fig_score_vs_dim.png      - Performance vs MLP dimension
# outputs/fig_latency_vs_dim.png    - Efficiency comparison
# outputs/fig_temporal_ablation.png - Temporal weight impact
# outputs/REPORT.md                 - Auto-generated report
```

---

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Smoke test (minimal data)
python -m pytest tests/test_train_smoke.py -v

# Format check
black src/ tests/ --check

# Linting
ruff check src/
```

---

## 📚 Citation

If you use TIDE-Lite in your research, please cite:

```bibtex
@article{tide-lite-2024,
  title={TIDE-Lite: Temporally-Indexed Dynamic Embeddings for Efficient Temporal Adaptation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## ⚡ Performance Tips

1. **Use Mixed Precision**: Add `--use-amp` flag for 2x speedup
2. **Batch Size**: Increase to GPU memory limit (typically 48-64)
3. **Gradient Accumulation**: For larger effective batch sizes
4. **Pin Memory**: Enabled by default in dataloaders
5. **Num Workers**: Set to CPU cores / 2

---

## 🐛 Common Issues

### CUDA Out of Memory
```bash
# Reduce batch size or enable gradient checkpointing
python train.py --batch-size 16 --gradient-checkpointing
```

### Slow Training
```bash
# Enable mixed precision and increase workers
python train.py --use-amp --num-workers 4
```

### Import Errors
```bash
# Ensure you're in the repo root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

**Questions?** Open an issue or reach out!
