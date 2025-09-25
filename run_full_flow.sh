#!/bin/bash
# Full training and evaluation flow
# Takes 2-6 hours depending on GPU and dataset sizes

set -e  # Exit on error

echo "================================================"
echo "TIDE-Lite Full Training & Evaluation Flow"
echo "================================================"

# Configuration
EXPERIMENT_NAME="tide_lite_full_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"
GPU_ID=0  # Change if using different GPU

# ================================================
# STAGE 1: Environment Setup
# ================================================
echo -e "\n[Stage 1] Environment Check"
echo "--------------------------------"

# Check GPU availability
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠ No GPU available - training will be slow!')
"

# Check dependencies
python3 -c "
import transformers
import datasets
import scipy
import pandas
print('✓ All dependencies available')
"

# ================================================
# STAGE 2: Data Preparation
# ================================================
echo -e "\n[Stage 2] Data Preparation"
echo "--------------------------------"

# Download and cache datasets
python3 -c "
from src.tide_lite.data.datasets import load_stsb, load_quora

print('Downloading/caching STS-B...')
cfg = {'cache_dir': './data'}
stsb = load_stsb(cfg)
print(f'  Train: {len(stsb[\"train\"])} samples')
print(f'  Val: {len(stsb[\"validation\"])} samples')
print(f'  Test: {len(stsb[\"test\"])} samples')

print('Downloading/caching Quora...')
corpus, queries, qrels = load_quora(cfg)
print(f'  Corpus: {len(corpus)} documents')
print(f'  Queries: {len(queries)} queries')
"

# Optional: Check for temporal datasets
echo -e "\nChecking for temporal datasets..."
if [ -d "./data/timeqa" ] || [ -d "./data/templama" ]; then
    echo "✓ Temporal datasets found"
    EVAL_TEMPORAL=true
else
    echo "⚠ No temporal datasets - skipping temporal evaluation"
    EVAL_TEMPORAL=false
fi

# ================================================
# STAGE 3: Baseline Evaluation (Optional)
# ================================================
echo -e "\n[Stage 3] Baseline Evaluation"
echo "--------------------------------"

# Evaluate baseline models for comparison
echo "Evaluating MiniLM baseline on STS-B..."
python3 -m src.tide_lite.cli.eval_stsb \
    --model-name "sentence-transformers/all-MiniLM-L6-v2" \
    --model-type baseline \
    --output-dir "${OUTPUT_DIR}/baselines/minilm" \
    --batch-size 128 \
    --device cuda:${GPU_ID}

echo "Evaluating MiniLM baseline on Quora..."
python3 -m src.tide_lite.cli.eval_quora \
    --model-name "sentence-transformers/all-MiniLM-L6-v2" \
    --model-type baseline \
    --output-dir "${OUTPUT_DIR}/baselines/minilm" \
    --batch-size 128 \
    --top-k 1 5 10 20 \
    --device cuda:${GPU_ID}

# ================================================
# STAGE 4: TIDE-Lite Training
# ================================================
echo -e "\n[Stage 4] TIDE-Lite Training"
echo "--------------------------------"

# Main training run
python3 -m src.tide_lite.cli.train \
    --config configs/tide_lite.yaml \
    --output-dir "${OUTPUT_DIR}/tide_lite" \
    --num-epochs 5 \
    --batch-size 32 \
    --eval-batch-size 128 \
    --learning-rate 1e-4 \
    --warmup-steps 1000 \
    --eval-every 2000 \
    --save-every 2000 \
    --gradient-accumulation 2 \
    --mixed-precision \
    --device cuda:${GPU_ID} \
    --seed 42 \
    --wandb \
    --wandb-project tide-lite-experiments \
    --wandb-name "${EXPERIMENT_NAME}"

# Get the final model path
MODEL_PATH="${OUTPUT_DIR}/tide_lite/final"

# ================================================
# STAGE 5: TIDE-Lite Evaluation
# ================================================
echo -e "\n[Stage 5] TIDE-Lite Evaluation"
echo "--------------------------------"

# Evaluate on STS-B (without temporal)
echo "Evaluating TIDE-Lite on STS-B (baseline mode)..."
python3 -m src.tide_lite.cli.eval_stsb \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}/eval/stsb_baseline" \
    --batch-size 128 \
    --device cuda:${GPU_ID}

# Evaluate on STS-B (with temporal if available)
echo "Evaluating TIDE-Lite on STS-B (temporal mode)..."
python3 -m src.tide_lite.cli.eval_stsb \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}/eval/stsb_temporal" \
    --use-temporal \
    --batch-size 128 \
    --device cuda:${GPU_ID}

# Evaluate on Quora (without temporal)
echo "Evaluating TIDE-Lite on Quora (baseline mode)..."
python3 -m src.tide_lite.cli.eval_quora \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}/eval/quora_baseline" \
    --batch-size 128 \
    --top-k 1 5 10 20 50 \
    --device cuda:${GPU_ID}

# Evaluate on Quora (with temporal)
echo "Evaluating TIDE-Lite on Quora (temporal mode)..."
python3 -m src.tide_lite.cli.eval_quora \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}/eval/quora_temporal" \
    --use-temporal \
    --batch-size 128 \
    --top-k 1 5 10 20 50 \
    --device cuda:${GPU_ID}

# Temporal dataset evaluation (if available)
if [ "$EVAL_TEMPORAL" = true ]; then
    echo "Evaluating on temporal datasets..."
    python3 -m src.tide_lite.cli.eval_temporal \
        --model-path "${MODEL_PATH}" \
        --output-dir "${OUTPUT_DIR}/eval/temporal" \
        --dataset timeqa \
        --batch-size 128 \
        --device cuda:${GPU_ID}
fi

# ================================================
# STAGE 6: Results Aggregation & Reporting
# ================================================
echo -e "\n[Stage 6] Results Aggregation"
echo "--------------------------------"

# Aggregate all results
python3 -m src.tide_lite.cli.aggregate \
    --experiment-dir "${OUTPUT_DIR}" \
    --output-file "${OUTPUT_DIR}/results_summary.json"

# Generate plots
python3 -m src.tide_lite.cli.plots \
    --results-file "${OUTPUT_DIR}/results_summary.json" \
    --output-dir "${OUTPUT_DIR}/plots"

# Generate final report
python3 -m src.tide_lite.cli.report \
    --experiment-dir "${OUTPUT_DIR}" \
    --output-file "${OUTPUT_DIR}/final_report.md"

# ================================================
# STAGE 7: Model Comparison (Optional)
# ================================================
echo -e "\n[Stage 7] Model Comparison"
echo "--------------------------------"

python3 -c "
import json
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')

# Load results
with open(output_dir / 'results_summary.json') as f:
    results = json.load(f)

print('Performance Summary:')
print('-' * 50)

# STS-B Results
if 'stsb' in results:
    print('STS-B Spearman Correlation:')
    for model, score in results['stsb'].items():
        print(f'  {model}: {score:.4f}')

# Quora Results
if 'quora' in results:
    print('\nQuora Retrieval Recall@10:')
    for model, metrics in results['quora'].items():
        if 'recall@10' in metrics:
            print(f'  {model}: {metrics[\"recall@10\"]:.4f}')

print('-' * 50)
print(f'Full results saved to: {output_dir}/final_report.md')
"

# ================================================
# COMPLETION
# ================================================
echo -e "\n================================================"
echo "✅ Full training and evaluation complete!"
echo "================================================"
echo "Results directory: ${OUTPUT_DIR}"
echo ""
echo "Key files:"
echo "  - Model: ${MODEL_PATH}"
echo "  - Summary: ${OUTPUT_DIR}/results_summary.json"
echo "  - Report: ${OUTPUT_DIR}/final_report.md"
echo "  - Plots: ${OUTPUT_DIR}/plots/"
echo ""
echo "To view in WandB: https://wandb.ai/your-username/tide-lite-experiments"
