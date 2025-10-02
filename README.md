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

## Results

### Quick Interpretation Guide

After running the full pipeline, check the deliverables for key insights:

1. **Off-Diagonal Gains (Cross-Period Performance)**
   - The Δ heatmaps show where LoRA improves most over baseline
   - Look for **green cells in cross-period scenarios** (e.g., 2019-2024 queries retrieving ≤2018 docs)
   - Positive deltas indicate LoRA's temporal adaptation reduces the semantic gap between periods
   - Rule of thumb: Δ NDCG@10 > +0.02 suggests meaningful improvement

2. **Year-Gap vs Performance Curves** (if generated)
   - Plots NDCG@10 decline as query-document year gap increases
   - LoRA should show **flatter curves** = more robust to temporal drift
   - Baseline typically shows steeper degradation beyond 3-4 year gaps

3. **Memory & Runtime Considerations**
   - **Training**: ~5-10 min per bucket on T4 (2 epochs, 6k samples, r=16)
   - **Inference**: Minimal overhead (~2-5% vs baseline) due to LoRA's low-rank structure
   - **Storage**: LoRA adapters are ~2-5MB each vs ~80MB for full fine-tuned models
   - **Scalability**: Multi-index retrieval adds latency (~1.5-2x single-index) but enables cross-period queries

### Generated Artifacts

After running `python -m temporal_lora.cli export-deliverables`, find:

- **Results**:
  - `deliverables/results/baseline_results.csv` - Baseline metrics per scenario
  - `deliverables/results/lora_results.csv` - LoRA metrics per scenario
  - `deliverables/results/quick_ablation.csv` - Ablation study results
  - `deliverables/results/quick_ablation.md` - Ablation summary

- **Visualizations**:
  - `deliverables/figures/comparison_heatmaps_ndcg@10.png` - Baseline | LoRA | Δ for NDCG@10
  - `deliverables/figures/comparison_heatmaps_recall@10.png` - Recall@10 comparison
  - `deliverables/figures/comparison_heatmaps_recall@100.png` - Recall@100 comparison
  - `deliverables/figures/comparison_heatmaps_mrr.png` - MRR comparison
  - `deliverables/figures/umap_embeddings.png` - UMAP projection by time bucket

- **Reproducibility**:
  - `deliverables/repro/environment.json` - System info, CUDA, packages, git SHA
  - `deliverables/repro/requirements_frozen.txt` - Exact package versions
  - `deliverables/README_results.md` - Top-line numbers and methodology

### Running Ablations

To test hyperparameter sensitivity:

```bash
python -m temporal_lora.ablate.quick --lora-ranks 8 16 --merge-strategies softmax mean --max-queries 100
```

This runs a quick ablation over LoRA rank and merge strategy, saving results to `deliverables/results/quick_ablation.*`.

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
