# TIDE-Lite CLI Examples

This document provides example commands for benchmarking TIDE-Lite against baselines using the unified orchestrator CLI.

## Important Note

All commands default to **dry-run mode** (show plan without executing). Add `--run` flag to actually execute.

## Baseline Benchmarks

### MiniLM Baseline
```bash
# Benchmark MiniLM on all three tasks (dry-run)
python -m tide_lite.cli.tide bench-all --model minilm --type baseline

# Actually execute
python -m tide_lite.cli.tide bench-all --model minilm --type baseline --run

# Individual evaluations
python -m tide_lite.cli.tide eval-stsb --model minilm --type baseline
python -m tide_lite.cli.tide eval-quora --model minilm --type baseline --max-corpus 10000
python -m tide_lite.cli.tide eval-temporal --model minilm --type baseline
```

### E5-Base Baseline
```bash
# Benchmark E5-Base on all three tasks
python -m tide_lite.cli.tide bench-all --model e5-base --type baseline

# With custom output directory
python -m tide_lite.cli.tide bench-all --model e5-base --type baseline \
    --output-dir results/baselines/e5-base
```

### BGE-Base Baseline
```bash
# Benchmark BGE-Base on all three tasks
python -m tide_lite.cli.tide bench-all --model bge-base --type baseline

# With limited data for quick testing
python -m tide_lite.cli.tide bench-all --model bge-base --type baseline \
    --max-corpus 1000 --max-queries 100
```

## TIDE-Lite Benchmark

### Trained Model
```bash
# Benchmark trained TIDE-Lite model
python -m tide_lite.cli.tide bench-all \
    --model results/run-20240315/checkpoints/best_model.pt \
    --type tide_lite

# Or use path to checkpoint directory
python -m tide_lite.cli.tide bench-all \
    --model path/to/tide_checkpoint \
    --type tide_lite
```

## Comparative Analysis

### Run All Baselines + TIDE-Lite
```bash
#!/bin/bash
# Script to benchmark all models

# Output directory
OUTPUT_DIR="results/benchmark-$(date +%Y%m%d)"

# Run baselines
for model in minilm e5-base bge-base; do
    echo "Benchmarking $model..."
    python -m tide_lite.cli.tide bench-all \
        --model $model \
        --type baseline \
        --output-dir $OUTPUT_DIR \
        --run
done

# Run TIDE-Lite
echo "Benchmarking TIDE-Lite..."
python -m tide_lite.cli.tide bench-all \
    --model results/latest/checkpoints/best_model.pt \
    --type tide_lite \
    --output-dir $OUTPUT_DIR \
    --run

# Aggregate results
python -m tide_lite.cli.tide aggregate \
    --results-dir $OUTPUT_DIR \
    --output $OUTPUT_DIR/summary.json \
    --run

# Generate report
python -m tide_lite.cli.tide report \
    --input $OUTPUT_DIR/summary.json \
    --output-dir $OUTPUT_DIR/report \
    --run
```

## Configuration Consistency

All benchmarks use these consistent parameters:

- **Tokenizer**: Model-specific (but same for each model across tasks)
- **Max Sequence Length**: 128 tokens
- **FAISS Index**: Flat (exact search)
- **FAISS Params**: Same across all models
- **Batch Size**: 128 for retrieval, 64 for others
- **Pooling**: Mean pooling

## Quick Test Commands

```bash
# Quick smoke test with minimal data
python -m tide_lite.cli.tide bench-all \
    --model minilm \
    --type baseline \
    --max-corpus 100 \
    --max-queries 10 \
    --run

# Dry-run to see full execution plan
python -m tide_lite.cli.tide bench-all \
    --model minilm \
    --type baseline
```

## Expected Outputs

Each benchmark creates:
- `results/metrics_stsb_<model>.json` - STS-B correlation metrics
- `results/metrics_quora_<model>.json` - Retrieval metrics (nDCG, Recall, MRR)
- `results/metrics_temporal_<model>.json` - Temporal understanding metrics

## Runtime Estimates

On a typical GPU (e.g., V100, T4):
- MiniLM baseline: ~5-10 minutes
- E5-Base baseline: ~10-15 minutes
- BGE-Base baseline: ~10-15 minutes
- TIDE-Lite: ~5-10 minutes (similar to MiniLM)

On CPU:
- Multiply times by ~5-10x

## Troubleshooting

If you encounter memory issues:
```bash
# Reduce batch size
export CUDA_VISIBLE_DEVICES=""  # Force CPU
python -m tide_lite.cli.tide bench-all \
    --model minilm \
    --type baseline \
    --max-corpus 1000 \
    --run
```

If FAISS is not installed:
```bash
# CPU version
pip install faiss-cpu

# GPU version (requires CUDA)
pip install faiss-gpu
```
