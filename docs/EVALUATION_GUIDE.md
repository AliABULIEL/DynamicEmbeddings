# Evaluation System Guide

This guide explains how to use the comprehensive evaluation system for comparing training modes, optimizing merge strategies, and tracking efficiency.

## Overview

The evaluation system provides:

1. **Cross-Mode Evaluation**: Compare baseline, LoRA, full FT, and sequential FT
2. **Query×Doc Bucket Matrices**: Detailed metrics for all time period combinations
3. **Temperature Sweep**: Optimize multi-index merge parameters
4. **Efficiency Tracking**: Parameter counts, model sizes, training times
5. **Delta Analysis**: Quantify LoRA improvements over baseline

## Quick Start

### 1. Evaluate All Modes

Run comprehensive evaluation across all training modes:

```bash
python -m temporal_lora.cli evaluate-all-modes \
  --modes "baseline_frozen,lora,full_ft,seq_ft" \
  --temperature-sweep \
  --temperatures "1.5,2.0,3.0"
```

**Output:**
- `deliverables/results/{mode}_ndcg_at_10.csv` - NDCG@10 matrices
- `deliverables/results/{mode}_recall_at_10.csv` - Recall@10 matrices
- `deliverables/results/{mode}_recall_at_100.csv` - Recall@100 matrices
- `deliverables/results/{mode}_mrr.csv` - MRR matrices
- `deliverables/results/{mode}_temperature_sweep.csv` - Temperature optimization
- `deliverables/results/comparisons/delta_*.csv` - LoRA vs Baseline deltas

### 2. Generate Efficiency Summary

Track parameter counts and training times:

```bash
python -m temporal_lora.cli efficiency-summary \
  --modes "baseline_frozen,lora,full_ft,seq_ft"
```

**Output:**
- `deliverables/results/efficiency_summary.csv`

**Metrics included:**
- Total parameters
- Trainable parameters (%)
- Model/adapter size (MB)
- Training time (seconds)
- Peak GPU memory (MB)

### 3. Single Mode Evaluation

Evaluate a specific mode:

```bash
python -m temporal_lora.cli evaluate \
  --lora \
  --mode multi-index \
  --merge softmax \
  --temperature 2.0
```

## Understanding Results

### Query×Doc Bucket Matrices

Each metric CSV contains a matrix where:
- **Rows**: Query time buckets (where queries come from)
- **Columns**: Document time buckets (where documents are retrieved from)
- **Diagonal**: Within-period retrieval (queries and docs from same period)
- **Off-diagonal**: Cross-period retrieval

Example `lora_ndcg_at_10.csv`:

```
           bucket_2018  bucket_2019_2024
bucket_2018      0.85           0.42
bucket_2019_2024 0.38           0.91
```

**Interpretation:**
- `0.85`: Within-period NDCG for 2018 (queries from 2018, docs from 2018)
- `0.42`: Cross-period NDCG (2018 queries, 2019-2024 docs)
- Diagonal values should be higher than off-diagonal

### Temperature Sweep Results

Format: `{mode}_temperature_sweep.csv`

```
temperature,query_bucket,merge_strategy,ndcg@10,recall@10,recall@100,mrr
1.5,bucket_2018,softmax,0.82,0.73,0.95,0.78
2.0,bucket_2018,softmax,0.85,0.76,0.96,0.81
3.0,bucket_2018,softmax,0.83,0.74,0.95,0.79
```

**Find optimal temperature:**
- Look for temperature with highest average NDCG@10
- Typically around 2.0-3.0 for softmax merge
- Lower temps = sharper (more confident)
- Higher temps = smoother (more exploration)

### Delta Matrices

Format: `delta_ndcg_at_10.csv` (LoRA - Baseline)

```
           bucket_2018  bucket_2019_2024
bucket_2018     +0.05          +0.12
bucket_2019_2024 +0.15         +0.08
```

**Positive values** = LoRA improvement  
**Negative values** = Baseline better (investigate!)

**Key insights:**
- Larger improvements on off-diagonal = better temporal adaptation
- Consistent positive diagonal = general improvement

### Efficiency Summary

Example output:

```
mode,bucket,total_params,trainable_params,trainable_percent,size_mb,wall_clock_seconds
baseline_frozen,all,22653952,0,0.0,0.0,0
lora,bucket_2018,22653952,294912,1.3,1.2,127.3
lora,bucket_2019_2024,22653952,294912,1.3,1.2,145.8
full_ft,bucket_2018,22653952,22653952,100.0,86.5,412.1
```

**Compare:**
- LoRA: <2% trainable, ~1MB per bucket, fast training
- Full FT: 100% trainable, ~86MB per bucket, slower training
- Sequential FT: Shows accumulation over steps

## Testing

Run evaluation smoke tests:

```bash
pytest tests/test_eval_smoke.py -v
```

**Tests cover:**
- FAISS index roundtrip
- Metric calculations
- Multi-index search with all merge strategies
- Cross-bucket evaluation matrices
- Temperature sweep
- Mode comparison and delta computation
