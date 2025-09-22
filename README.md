# TIDE-Lite: Temporally-Indexed Dynamic Embeddings

Frozen encoder + tiny timestamp-aware MLP that modulates sentence embeddings.
Runs on Colab/1xGPU. Includes STS-B (Spearman), Quora retrieval (nDCG@10), and temporal evaluation (TimeQA-lite).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training

### Basic Training

```bash
# Train with default configuration
python -m tide_lite.cli.train_cli

# Dry run to see training plan (no execution)
python -m tide_lite.cli.train_cli --dry-run

# Custom configuration file
python -m tide_lite.cli.train_cli --config configs/custom.yaml
```

### Override Parameters

```bash
# Adjust hyperparameters
python -m tide_lite.cli.train_cli \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --num-epochs 5 \
    --temporal-weight 0.2

# Custom output directory
python -m tide_lite.cli.train_cli \
    --output-dir results/experiment1 \
    --checkpoint-dir checkpoints/experiment1

# Disable mixed precision (for debugging)
python -m tide_lite.cli.train_cli --no-amp

# Fine-tune encoder (not frozen)
python -m tide_lite.cli.train_cli --no-freeze-encoder
```

### Configuration Priority

1. Command-line arguments (highest priority)
2. Custom config file (`--config`)
3. Default config (`configs/defaults.yaml`)

### Output Structure

```
results/run_20240315_142530/
├── config_used.json        # Actual configuration used
├── metrics_train.json      # Training metrics per epoch
├── training.log           # Detailed training logs
├── dry_run_summary.json   # Dry run plan (if --dry-run)
└── checkpoints/
    ├── checkpoint_epoch_1.pt
    ├── checkpoint_step_500.pt
    └── checkpoint_final.pt
```

## Model Architecture

- **Frozen Encoder**: sentence-transformers/all-MiniLM-L6-v2 (22M params)
- **Temporal MLP**: 32 → 128 → 384 dims (53K extra params)
- **Time Encoding**: Sinusoidal positional encoding adapted for timestamps
- **Gating**: Sigmoid activation for element-wise modulation

## Expected Results

| Model | STS-B (Spearman) | nDCG@10 (Quora) | Temporal Acc | Latency (ms) | Extra Params |
|-------|------------------|-----------------|--------------|--------------|--------------|
| all-MiniLM-L6-v2 (baseline) | 0.82 | 0.68 | N/A | 8 | 0 |
| TIDE-Lite (no temporal loss) | 0.81 | 0.67 | 0.72 | 10 | 53K |
| TIDE-Lite (full) | 0.82 | 0.69 | 0.85 | 10 | 53K |

## Project Structure

```
DynamicEmbeddings/
├── src/tide_lite/
│   ├── data/          # Data loading and collation
│   ├── models/        # TIDE-Lite and baselines
│   ├── train/         # Training logic and losses
│   ├── eval/          # Evaluation metrics (TBD)
│   ├── cli/           # Command-line interfaces
│   └── utils/         # Configuration and utilities
├── configs/           # YAML configuration files
├── scripts/           # Helper scripts
├── notebooks/         # Colab notebooks
└── results/           # Training outputs
```

See `configs/defaults.yaml` for all configuration options.
