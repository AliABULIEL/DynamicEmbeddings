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

### Key Outputs

After running the pipeline, key results are found in `deliverables/`:

#### Performance Matrices
- **`results/{mode}_ndcg_at_10.csv`** - Query bucket × doc bucket matrices for NDCG@10
- **`results/{mode}_recall_at_10.csv`** - Recall@10 matrices  
- **`results/{mode}_recall_at_100.csv`** - Recall@100 matrices
- **`results/{mode}_mrr.csv`** - Mean reciprocal rank matrices
- **`results/comparisons/delta_*.csv`** - Improvement matrices (LoRA - baseline)

Where `{mode}` ∈ {`baseline_frozen`, `lora`, `full_ft`, `seq_ft`}

#### Visualizations
- **`figures/heatmap_panel_ndcg_at_10.png`** - Three-panel heatmap: Baseline | LoRA | Δ
- **`figures/heatmap_panel_recall_at_10.png`** - Recall@10 comparison
- **`figures/heatmap_panel_recall_at_100.png`** - Recall@100 comparison
- **`figures/heatmap_panel_mrr.png`** - MRR comparison
- **`figures/umap_embeddings.png`** - UMAP projection colored by time bucket (≤10k points)
- **`figures/drift_trajectories.png`** - Term drift visualization with arrowed polylines

#### Efficiency & Ablation
- **`results/efficiency_summary.csv`** - Parameter count, size (MB), training time
- **`results/quick_ablation.csv`** - LoRA rank × target modules ablation
- **`results/quick_ablation_summary.md`** - Best configurations and insights
- **`results/{mode}_temperature_sweep.csv`** - Multi-index merge temperature optimization

#### Reproducibility
- **`repro/system_info.txt`** - CUDA, CPU, OS details
- **`repro/pip_freeze.txt`** - Exact package versions
- **`repro/git_sha.txt`** - Git commit for reproducibility

### Interpretation Guide

**1. Cross-Period Performance (Off-Diagonal Elements)**
- Δ heatmaps show improvements where LoRA matters most
- **Green cells** in cross-period positions (e.g., 2018 queries → 2024 docs) indicate successful temporal adaptation
- **Δ NDCG@10 > +0.05** = strong improvement; **< +0.02** = marginal

**2. Term Drift Trajectories**
- Arrowed polylines show semantic evolution across time buckets
- Longer arrows = more semantic drift for that term
- Terms like "transformer", "BERT", "LLM" should show clear temporal progression

**3. Efficiency Gains**
- LoRA: **<2% trainable params**, ~1-2 MB per bucket, ~5-10 min training (T4, 2 epochs)
- Full FT: **100% trainable params**, ~86 MB per bucket, ~15-20 min training
- Storage: **40x smaller** for LoRA (2 buckets) vs full FT

**4. Temperature Sweep**
- Optimal temperature typically **2.0-3.0** for softmax merge
- Higher temps = smoother blending, lower temps = sharper selection
- Check `temperature_sweep.csv` for best config per metric

### Expected Results

**Within-Period (Diagonal):**
- NDCG@10: 0.70-0.85 (baseline), 0.75-0.88 (LoRA)
- Recall@100: 0.85-0.95 (baseline), 0.88-0.97 (LoRA)

**Cross-Period (Off-Diagonal):**
- NDCG@10: 0.35-0.55 (baseline), 0.45-0.65 (LoRA)  
- **Δ NDCG@10**: +0.05 to +0.15 (typical LoRA improvement)

**Efficiency:**
- LoRA trainable %: 1.0-1.5%
- LoRA size: 1-2 MB per bucket
- Training speedup: 2-3x faster than full FT

### Running the Full Pipeline

See `notebooks/chronoembed_demo.ipynb` for a **one-click end-to-end demo** (runs in ~30-45 min on Colab T4).

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
