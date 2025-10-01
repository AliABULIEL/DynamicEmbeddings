# TIDE-Lite Benchmarking Examples

This document provides example commands for benchmarking TIDE-Lite against baseline models using the unified orchestrator CLI.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m tide_lite.cli.tide --help
```

## Benchmarking Baselines

All benchmark commands use identical configuration:
- **Tokenizer**: Model-specific (auto-detected)
- **Max sequence length**: 128 tokens
- **FAISS index**: Flat (exact search)
- **Batch size**: 128 for retrieval, 64 for STS-B

### 1. Benchmark MiniLM (Baseline)

```bash
# Dry-run (default) - shows execution plan
python -m tide_lite.cli.tide bench-all --model minilm --type baseline

# Actually execute benchmarks
python -m tide_lite.cli.tide bench-all --model minilm --type baseline --run

# Output files:
# - results/metrics_stsb_minilm.json
# - results/metrics_quora_minilm.json
# - results/metrics_temporal_minilm.json
```

### 2. Benchmark E5-Base (Baseline)

```bash
# Dry-run (default) - shows execution plan
python -m tide_lite.cli.tide bench-all --model e5-base --type baseline

# Actually execute benchmarks
python -m tide_lite.cli.tide bench-all --model e5-base --type baseline --run

# Output files:
# - results/metrics_stsb_e5-base.json
# - results/metrics_quora_e5-base.json
# - results/metrics_temporal_e5-base.json
```

### 3. Benchmark BGE-Base (Baseline)

```bash
# Dry-run (default) - shows execution plan
python -m tide_lite.cli.tide bench-all --model bge-base --type baseline

# Actually execute benchmarks
python -m tide_lite.cli.tide bench-all --model bge-base --type baseline --run

# Output files:
# - results/metrics_stsb_bge-base.json
# - results/metrics_quora_bge-base.json
# - results/metrics_temporal_bge-base.json
```

### 4. Benchmark TIDE-Lite (Trained Model)

```bash
# After training TIDE-Lite
python -m tide_lite.cli.tide train --output-dir results/tide_run1 --run

# Benchmark the trained model
python -m tide_lite.cli.tide bench-all \
    --model results/tide_run1/checkpoints/best_model.pt \
    --type tide_lite

# Actually execute benchmarks
python -m tide_lite.cli.tide bench-all \
    --model results/tide_run1/checkpoints/best_model.pt \
    --type tide_lite \
    --run

# Output files:
# - results/metrics_stsb_best_model.json
# - results/metrics_quora_best_model.json
# - results/metrics_temporal_best_model.json
```

## Individual Evaluations

You can also run individual evaluations:

```bash
# STS-B only
python -m tide_lite.cli.tide eval-stsb --model minilm --type baseline --run

# Quora retrieval only
python -m tide_lite.cli.tide eval-quora --model minilm --type baseline --run

# Temporal understanding only
python -m tide_lite.cli.tide eval-temporal --model minilm --type baseline --run
```

## Custom Configuration

Override default parameters:

```bash
# Use IVF index for faster retrieval (less accurate)
python -m tide_lite.cli.tide eval-quora \
    --model minilm \
    --type baseline \
    --index-type IVF \
    --max-corpus 50000 \
    --max-queries 5000 \
    --run

# Custom time window for temporal evaluation
python -m tide_lite.cli.tide eval-temporal \
    --model minilm \
    --type baseline \
    --time-window-days 7 \
    --run
```

## Aggregate Results

After running benchmarks, aggregate all results:

```bash
# Aggregate all metrics
python -m tide_lite.cli.tide aggregate --results-dir results/ --run

# Generate report with plots
python -m tide_lite.cli.tide report --input results/summary.json --run
```

## Ablation Study

Run hyperparameter grid search:

```bash
# Dry-run to see all configurations
python -m tide_lite.cli.tide ablation \
    --time-mlp-hidden 64,128,256 \
    --consistency-weight 0.05,0.1,0.2 \
    --time-encoding sinusoidal,learnable

# Execute ablation study (warning: time-consuming)
python -m tide_lite.cli.tide ablation \
    --time-mlp-hidden 64,128,256 \
    --consistency-weight 0.05,0.1,0.2 \
    --time-encoding sinusoidal,learnable \
    --run
```

## Expected Runtimes

On a typical GPU (e.g., NVIDIA T4):
- **Single baseline benchmark**: ~10-15 minutes
- **Full bench-all (3 evaluations)**: ~30-45 minutes
- **TIDE-Lite training**: ~1-2 hours (3 epochs)
- **Ablation study (9 configs)**: ~4-6 hours

On CPU:
- Multiply all times by approximately 5-10x

## Unified Configuration

All evaluations use consistent parameters to ensure fair comparison:

| Parameter | Value | Note |
|-----------|-------|------|
| Max sequence length | 128 | Same for all models |
| Pooling strategy | mean | Average over tokens |
| FAISS index | Flat | Exact search (default) |
| Quora max corpus | 10000 | For efficiency |
| Quora max queries | 1000 | For efficiency |
| Temporal window | 30 days | Default time window |
| STS-B split | test | Final evaluation |

## Tips

1. **Always dry-run first**: Check the execution plan before running
2. **Use smaller datasets for testing**: Add `--max-corpus 1000 --max-queries 100`
3. **Monitor GPU memory**: Reduce batch size if OOM occurs
4. **Save intermediate results**: Results are saved after each evaluation
5. **Resume from checkpoints**: Training can be resumed if interrupted

## Troubleshooting

If you encounter issues:

```bash
# Check available models
python -c "from tide_lite.models.baselines import MODEL_MAP; print(MODEL_MAP)"

# Verify FAISS installation
python -c "import faiss; print('FAISS available')"

# Test with minimal data
python -m tide_lite.cli.tide eval-stsb \
    --model minilm \
    --type baseline \
    --batch-size 8 \
    --run
```
