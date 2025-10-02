# Temporal LoRA for Dynamic Sentence Embeddings

**Dynamic embeddings** `z = f(x, t)` via time-bucket LoRA adapters on a frozen sentence encoder.

## Overview

This project implements temporal adaptations to sentence embeddings using Low-Rank Adaptation (LoRA) on time-stratified buckets. The base model (`sentence-transformers/all-MiniLM-L6-v2`) remains frozen while lightweight LoRA adapters learn period-specific representations.

**Key Features:**
- **Time-bucketed LoRA**: Separate adapters for `≤2018` and `2019-2024` (configurable)
- **Cross-period negative sampling**: Harder negatives from different time periods
- **Multi-index retrieval**: FAISS indexes per bucket with configurable merge strategies
- **Comprehensive evaluation**: NDCG@10, Recall@10/100, MRR with bootstrap CIs and permutation tests
- **Visualizations**: Period heatmaps, UMAP projections, term drift trajectories

## Quickstart

```bash
pip install -r requirements.txt
python -m temporal_lora.cli env-dump
python -m temporal_lora.cli prepare-data --max_per_bucket 6000
python -m temporal_lora.cli train-adapters --epochs 2 --lora_r 16 --cross_period_negatives true
python -m temporal_lora.cli build-indexes
python -m temporal_lora.cli evaluate --scenarios within cross all --mode multi-index --merge softmax
python -m temporal_lora.cli visualize
python -m temporal_lora.cli export-deliverables
```

## Project Structure

```
DynamicEmbeddings/
├─ src/temporal_lora/        # Main package
│  ├─ config/                # YAML configs
│  ├─ utils/                 # Shared utilities
│  ├─ data/                  # Data processing
│  ├─ models/                # LoRA adapters & encoding
│  ├─ retrieval/             # FAISS indexing
│  ├─ evaluation/            # Metrics & statistical tests
│  └─ visualization/         # Plots
├─ tests/                    # Unit tests
├─ notebooks/                # Exploratory analysis
├─ scripts/                  # Standalone utilities
├─ deliverables/             # Output artifacts
└─ data/                     # Processed data only
```

## Configuration

Default configs in `src/temporal_lora/config/`:
- `data.yaml`: Buckets, max samples per bucket
- `model.yaml`: Base model, LoRA hyperparameters
- `train.yaml`: Training settings (epochs, LR, batch size)
- `eval.yaml`: Evaluation modes and merge strategies

## Requirements

- Python 3.9+
- Single GPU (T4 or better) or CPU
- ~8GB RAM for default dataset size

## License

MIT
