# TIDE-Lite CLI Examples

This document provides example commands for benchmarking and evaluation using the TIDE-Lite unified orchestrator CLI.

## Quick Reference

All commands default to **dry-run mode** (show plan only). Add `--run` flag to actually execute.

## Benchmarking Baselines

### MiniLM Baseline
```bash
# Dry-run (default) - shows execution plan
python -m tide_lite.cli.tide bench-all --model minilm --type baseline

# Actually execute with --run
python -m tide_lite.cli.tide bench-all --model minilm --type baseline --run

# With custom output directory
python -m tide_lite.cli.tide bench-all --model minilm --type baseline \
    --output-dir results/baselines/minilm --run
```

### E5-Base Baseline
```bash
# Dry-run (default) - shows execution plan
python -m tide_lite.cli.tide bench-all --model e5-base --type baseline

# Actually execute
python -m tide_lite.cli.tide bench-all --model e5-base --type baseline --run

# With custom output directory
python -m tide_lite.cli.tide bench-all --model e5-base --type baseline \
    --output-dir results/baselines/e5-base --run
```

### BGE-Base Baseline
```bash
# Dry-run (default) - shows execution plan
python -m tide_lite.cli.tide bench-all --model bge-base --type baseline

# Actually execute
python -m tide_lite.cli.tide bench-all --model bge-base --type baseline --run

# With custom output directory
python -m tide_lite.cli.tide bench-all --model bge-base --type baseline \
    --output-dir results/baselines/bge-base --run
```

## Benchmarking TIDE-Lite

### Trained TIDE-Lite Checkpoint
```bash
# Dry-run (default) - shows execution plan
python -m tide_lite.cli.tide bench-all --model path/to/tide_checkpoint.pt

# Actually execute
python -m tide_lite.cli.tide bench-all --model path/to/tide_checkpoint.pt --run

# Example with specific checkpoint
python -m tide_lite.cli.tide bench-all \
    --model results/run-20240315-142530/checkpoints/best_model.pt \
    --output-dir results/tide_lite_eval --run
```

## Complete Baseline Comparison

```bash
# Run all baselines in sequence (dry-run)
python -m tide_lite.cli.tide bench-all --model minilm --type baseline
python -m tide_lite.cli.tide bench-all --model e5-base --type baseline
python -m tide_lite.cli.tide bench-all --model bge-base --type baseline

# Run all baselines in sequence (actual execution)
python -m tide_lite.cli.tide bench-all --model minilm --type baseline --run
python -m tide_lite.cli.tide bench-all --model e5-base --type baseline --run
python -m tide_lite.cli.tide bench-all --model bge-base --type baseline --run

# Then aggregate results
python -m tide_lite.cli.tide aggregate --results-dir results/ --run

# Generate comparison report
python -m tide_lite.cli.tide report --input results/summary.json --run
```

## Individual Evaluations

### STS-B Only
```bash
# Baseline
python -m tide_lite.cli.tide eval-stsb --model minilm --type baseline --run

# TIDE-Lite
python -m tide_lite.cli.tide eval-stsb --model path/to/checkpoint.pt --run
```

### Quora Retrieval Only
```bash
# Baseline with specific FAISS config
python -m tide_lite.cli.tide eval-quora --model minilm --type baseline \
    --index-type Flat --max-corpus 10000 --max-queries 1000 --run

# TIDE-Lite
python -m tide_lite.cli.tide eval-quora --model path/to/checkpoint.pt \
    --index-type Flat --max-corpus 10000 --max-queries 1000 --run
```

### Temporal Evaluation Only
```bash
# Baseline
python -m tide_lite.cli.tide eval-temporal --model minilm --type baseline \
    --time-window-days 30 --run

# TIDE-Lite
python -m tide_lite.cli.tide eval-temporal --model path/to/checkpoint.pt \
    --time-window-days 30 --run
```

## Consistent Parameters

All evaluations use consistent parameters across models:
- **Max sequence length**: 128 tokens
- **FAISS index type**: Flat (exact search)
- **Corpus size**: 10,000 documents (Quora)
- **Query size**: 1,000 queries (Quora)
- **Time window**: 30 days (Temporal)
- **Batch size**: 128 (evaluation)

## Output Structure

Results are saved in JSON format:
```
results/
├── metrics_stsb_<model>.json
├── metrics_quora_<model>.json
├── metrics_temporal_<model>.json
├── summary.json            # Aggregated results
├── summary.csv             # CSV format
└── report.md              # Generated report
```

## Troubleshooting

### Out of Memory
Reduce batch size in evaluation:
```bash
python -m tide_lite.cli.tide bench-all --model minilm --type baseline \
    --batch-size 32 --run
```

### FAISS Not Installed
Install FAISS for retrieval evaluation:
```bash
# CPU version
pip install faiss-cpu

# GPU version (if CUDA available)
pip install faiss-gpu
```

### Missing Datasets
Datasets are automatically downloaded on first use. Ensure internet connection and sufficient disk space (~2GB).

## Notes

- All commands show execution plan by default (dry-run mode)
- Add `--run` flag to actually execute commands
- Results are saved to `results/` by default (override with `--output-dir`)
- Use `--help` on any command for detailed options
