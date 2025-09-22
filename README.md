# TIDE-Lite: Temporally-Indexed Dynamic Embeddings (Lite Edition)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab Ready](https://img.shields.io/badge/Colab-Ready-green.svg)](notebooks/tide_lite_colab.ipynb)

## What is TIDE-Lite?

TIDE-Lite is a **lightweight temporal adaptation layer** for sentence embeddings that adds time-awareness with minimal overhead. Instead of retraining massive language models, TIDE-Lite uses a tiny MLP (≤1M parameters) to modulate frozen encoder embeddings based on temporal context.

**Key Innovation:** Temporal gating mechanism that preserves baseline performance while adding time-awareness through learned modulation patterns.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Input Text                          │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│            Frozen Encoder (e.g., MiniLM-L6)                 │
│                     [22M params, fixed]                      │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
                 Base Embedding (384d)
                       ↓
         ┌─────────────┴─────────────┐
         ↓                           ↓
┌──────────────────┐        ┌──────────────────┐
│   Timestamp      │        │  Preservation    │
│   (Unix time)    │        │     Branch       │
└────────┬─────────┘        └────────┬─────────┘
         ↓                           ↓
┌──────────────────┐                │
│ Time Encoding    │                │
│ (Sinusoidal 32d) │                │
└────────┬─────────┘                │
         ↓                           │
┌──────────────────┐                │
│  Temporal MLP    │                │
│ 32→128→384 dims  │                │
│  [53K params]    │                │
└────────┬─────────┘                │
         ↓                           │
┌──────────────────┐                │
│ Sigmoid Gating   │                │
└────────┬─────────┘                │
         ↓                           ↓
         └─────────────┬─────────────┘
                       ↓
                Hadamard Product
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                 Temporally-Modulated Embedding              │
│                          (384d output)                       │
└─────────────────────────────────────────────────────────────┘
```

## Constraints & Design Principles

- **≤1M Extra Parameters:** Entire temporal module must stay under 1M params
- **Frozen Base Encoder:** No fine-tuning of the base model (22M+ params stay frozen)
- **Single GPU Training:** Must run on 1x T4/V100 GPU or Colab
- **Preservation by Default:** Maintain baseline performance on non-temporal tasks
- **Fast Inference:** <2ms additional latency over base encoder

## Benchmarks

| Benchmark | Metric | Description |
|-----------|---------|-------------|
| **STS-B** | Spearman ρ | Semantic textual similarity (standard) |
| **Quora Duplicate Pairs** | nDCG@10 | Question retrieval effectiveness |
| **TimeQA-Lite** | Accuracy | Temporal reasoning (synthetic) |
| **Latency** | ms/query | Inference speed overhead |
| **Memory** | MB | Additional memory footprint |

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (optional, for GPU)
- 4GB RAM minimum

### Local Installation
```bash
# Clone repository
git clone https://github.com/yourusername/TIDE-Lite.git
cd TIDE-Lite

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab
```python
# Run in first cell
!git clone https://github.com/yourusername/TIDE-Lite.git
%cd TIDE-Lite
!pip install -r requirements.txt
```

## Quickstart

### Complete Pipeline (Copy-Paste Commands)

```bash
# 1. Train TIDE-Lite model (3 epochs, ~10 min on GPU)
python -m tide_lite.cli.tide train \
    --config configs/defaults.yaml \
    --output-dir results/quickstart \
    --num-epochs 3

# 2. Evaluate on STS-B benchmark
python -m tide_lite.cli.tide eval-stsb \
    --model-path results/quickstart/checkpoints/final.pt \
    --output-dir results/quickstart/eval_stsb \
    --compare-baseline

# 3. Evaluate on Quora retrieval
python -m tide_lite.cli.tide eval-quora \
    --model-path results/quickstart/checkpoints/final.pt \
    --output-dir results/quickstart/eval_quora \
    --max-corpus 10000

# 4. Evaluate temporal capabilities
python -m tide_lite.cli.tide eval-temporal \
    --model-path results/quickstart/checkpoints/final.pt \
    --output-dir results/quickstart/eval_temporal \
    --compare-baseline

# 5. Run all benchmarks at once
python -m tide_lite.cli.tide bench-all \
    --model-path results/quickstart/checkpoints/final.pt \
    --output-dir results/quickstart/bench_all

# 6. Aggregate results across all evaluations
python -m tide_lite.cli.tide aggregate \
    --results-dir results/quickstart \
    --output results/quickstart/summary.json

# 7. Generate final report with plots
python -m tide_lite.cli.tide report \
    --input results/quickstart/summary.json \
    --output-dir results/quickstart/report
```

### Quick Test (Dry Run - No Execution)
```bash
# Test the entire pipeline without actual execution
python -m tide_lite.cli.tide train --dry-run
python -m tide_lite.cli.tide bench-all --dry-run --model-path dummy.pt
python -m tide_lite.cli.tide report --dry-run --input dummy.json
```

## Expected Results

| Model | STS-B (ρ) | Quora (nDCG@10) | Temporal Acc | Latency | Extra Params |
|-------|-----------|-----------------|--------------|---------|--------------|
| MiniLM-L6 (baseline) | 0.82 | 0.68 | 0.50 | 8ms | 0 |
| TIDE-Lite (no temporal) | 0.81 | 0.67 | 0.72 | 10ms | 53K |
| **TIDE-Lite (full)** | **0.82** | **0.69** | **0.85** | 10ms | 53K |

## Advanced Usage

### Custom Training Configuration
```bash
python -m tide_lite.cli.tide train \
    --batch-size 64 \
    --learning-rate 3e-5 \
    --temporal-weight 0.15 \
    --mlp-hidden-dim 256 \
    --time-encoding-dim 64
```

### Ablation Studies
```bash
python -m tide_lite.cli.tide ablation \
    --mlp-hidden-dims 64,128,256 \
    --temporal-weights 0.0,0.1,0.2 \
    --output-dir results/ablation
```

### Using Pre-trained Models
```python
from tide_lite.models import TIDELite

# Load pre-trained model
model = TIDELite.from_pretrained("path/to/checkpoint.pt")

# Encode with timestamp
embedding = model.encode(
    text="The stock market crashed", 
    timestamp="2008-09-15"
)
```

## Project Structure

```
TIDE-Lite/
├── src/tide_lite/
│   ├── cli/           # Command-line interfaces
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # TIDE-Lite architecture
│   ├── train/         # Training logic and losses
│   ├── eval/          # Evaluation scripts
│   ├── plots/         # Visualization utilities
│   └── utils/         # Configuration and helpers
├── configs/           # YAML configuration files
├── scripts/           # Helper scripts and guides
├── notebooks/         # Jupyter/Colab notebooks
├── tests/            # Unit and integration tests
└── results/          # Training outputs (gitignored)
```

## Documentation

- **[Local Setup Guide](scripts/how_to_run_locally.md)** - Detailed local installation
- **[Colab Guide](scripts/how_to_run_on_colab.md)** - Running on Google Colab
- **[Examples](scripts/examples.md)** - Comprehensive usage examples
- **[API Reference](docs/api.md)** - Python API documentation
- **[Paper](paper/tide_lite.pdf)** - Technical details and experiments

## Citing TIDE-Lite

If you use TIDE-Lite in your research, please cite:
```bibtex
@article{tide-lite-2024,
  title={TIDE-Lite: Temporally-Indexed Dynamic Embeddings with Minimal Overhead},
  author={Your Name},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/TIDE-Lite/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/TIDE-Lite/discussions)
- **Email:** tide-lite@example.com
