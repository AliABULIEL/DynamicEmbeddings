# TIDE-Lite Pipeline Examples

This document provides comprehensive examples of using the TIDE-Lite unified orchestrator CLI (`tide`) for the complete pipeline from training through evaluation to reporting.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Basic training
tide train --config configs/defaults.yaml

# Quick evaluation on all tasks
tide bench-all --model-path results/run_*/checkpoints/final.pt

# Generate report
tide aggregate --results-dir results/
tide report --input results/summary.json
```

## Complete Pipeline Examples

### 1. Standard Training and Evaluation Pipeline

```bash
# Step 1: Train TIDE-Lite model
tide train \
  --config configs/defaults.yaml \
  --output-dir results/experiment1 \
  --num-epochs 3 \
  --batch-size 32 \
  --learning-rate 5e-5 \
  --temporal-weight 0.1

# Step 2: Evaluate on all tasks
tide bench-all \
  --model-path results/experiment1/checkpoints/final.pt \
  --output-dir results/experiment1/eval

# Step 3: Compare with baseline
tide bench-all \
  --baseline minilm \
  --output-dir results/experiment1/eval_baseline

# Step 4: Aggregate results
tide aggregate \
  --results-dir results/experiment1 \
  --output results/experiment1/summary.json

# Step 5: Generate report
tide report \
  --input results/experiment1/summary.json \
  --output-dir results/experiment1/report
```

### 2. Individual Task Evaluation

```bash
# STS-B evaluation with comparison
tide eval-stsb \
  --model-path results/experiment1/checkpoints/final.pt \
  --output-dir results/eval_stsb \
  --split test \
  --compare-baseline

# Quora retrieval with limited corpus
tide eval-quora \
  --model-path results/experiment1/checkpoints/final.pt \
  --output-dir results/eval_quora \
  --index-type IVFFlat \
  --max-corpus 50000 \
  --max-queries 5000

# Temporal evaluation with custom window
tide eval-temporal \
  --model-path results/experiment1/checkpoints/final.pt \
  --output-dir results/eval_temporal \
  --time-window-days 7 \
  --max-samples 10000 \
  --compare-baseline
```

### 3. Ablation Study

```bash
# Run ablation study on key hyperparameters
tide ablation \
  --mlp-hidden-dims 32,64,128,256 \
  --temporal-weights 0.0,0.05,0.1,0.2,0.5 \
  --time-encoding-dims 16,32,64 \
  --config configs/defaults.yaml \
  --output-dir results/ablation \
  --num-epochs 2

# Aggregate ablation results
tide aggregate \
  --results-dir results/ablation \
  --output results/ablation/summary.json

# Generate ablation report
tide report \
  --input results/ablation/summary.json \
  --output-dir results/ablation/report
```

### 4. Quick Testing Pipeline (Dry Run)

```bash
# Test the entire pipeline without execution
tide train --dry-run --config configs/defaults.yaml

tide bench-all --dry-run \
  --model-path results/test_model.pt

tide ablation --dry-run \
  --mlp-hidden-dims 64,128 \
  --temporal-weights 0.1,0.2

tide aggregate --dry-run --results-dir results/

tide report --dry-run --input results/summary.json
```

### 5. Production Pipeline with Custom Settings

```bash
# Production training with optimized settings
tide train \
  --config configs/production.yaml \
  --output-dir results/prod_v1 \
  --batch-size 64 \
  --learning-rate 3e-5 \
  --num-epochs 5 \
  --temporal-weight 0.15 \
  --no-amp  # Disable mixed precision for stability

# Comprehensive evaluation
for split in validation test; do
  tide eval-stsb \
    --model-path results/prod_v1/checkpoints/final.pt \
    --output-dir results/prod_v1/eval_${split} \
    --split ${split} \
    --batch-size 128
done

# Large-scale retrieval evaluation
tide eval-quora \
  --model-path results/prod_v1/checkpoints/final.pt \
  --output-dir results/prod_v1/eval_retrieval \
  --index-type Flat \
  --batch-size 256

# Temporal evaluation with fine-grained window
tide eval-temporal \
  --model-path results/prod_v1/checkpoints/final.pt \
  --output-dir results/prod_v1/eval_temporal \
  --time-window-days 1 \
  --batch-size 64
```

### 6. Baseline Comparison Study

```bash
# Evaluate all baseline models
for baseline in minilm e5-base bge-base; do
  tide bench-all \
    --baseline ${baseline} \
    --output-dir results/baselines/${baseline}
done

# Evaluate TIDE-Lite
tide bench-all \
  --model-path results/best_model/checkpoints/final.pt \
  --output-dir results/tide_lite

# Aggregate all results
tide aggregate \
  --results-dir results/ \
  --output results/comparison_summary.json

# Generate comparison report
tide report \
  --input results/comparison_summary.json \
  --output-dir results/comparison_report
```

### 7. Continuous Integration Pipeline

```bash
#!/bin/bash
# ci_pipeline.sh - Run in CI/CD environment

set -e  # Exit on error

# Quick smoke test
tide train \
  --config configs/ci_test.yaml \
  --output-dir /tmp/ci_test \
  --num-epochs 1 \
  --batch-size 16 \
  --dry-run

# Evaluation on small subset
tide eval-stsb \
  --baseline minilm \
  --output-dir /tmp/ci_test/eval \
  --batch-size 32 \
  --dry-run

echo "CI pipeline passed!"
```

### 8. Hyperparameter Search

```bash
# Grid search over learning rates and batch sizes
for lr in 1e-5 3e-5 5e-5 1e-4; do
  for bs in 16 32 64; do
    output_dir="results/hp_search/lr${lr}_bs${bs}"
    
    tide train \
      --config configs/defaults.yaml \
      --output-dir ${output_dir} \
      --learning-rate ${lr} \
      --batch-size ${bs} \
      --num-epochs 2
    
    tide eval-stsb \
      --model-path ${output_dir}/checkpoints/final.pt \
      --output-dir ${output_dir}/eval \
      --split validation
  done
done

# Aggregate hyperparameter search results
tide aggregate \
  --results-dir results/hp_search \
  --output results/hp_search/summary.json
```

## Advanced Usage

### Custom Configuration Override

```bash
# Override multiple configuration parameters
tide train \
  --config configs/defaults.yaml \
  --output-dir results/custom \
  --encoder-name microsoft/deberta-v3-base \
  --hidden-dim 768 \
  --mlp-hidden-dim 256 \
  --time-encoding-dim 64 \
  --mlp-dropout 0.2 \
  --no-freeze-encoder \
  --gradient-clip 0.5 \
  --warmup-steps 500
```

### Distributed Evaluation

```bash
# Evaluate on different GPUs in parallel
CUDA_VISIBLE_DEVICES=0 tide eval-stsb \
  --model-path model.pt \
  --output-dir results/gpu0 &

CUDA_VISIBLE_DEVICES=1 tide eval-quora \
  --model-path model.pt \
  --output-dir results/gpu1 &

CUDA_VISIBLE_DEVICES=2 tide eval-temporal \
  --model-path model.pt \
  --output-dir results/gpu2 &

wait  # Wait for all evaluations to complete
```

### Batch Processing Multiple Models

```bash
# Evaluate all checkpoints in a directory
for checkpoint in results/*/checkpoints/*.pt; do
  model_name=$(basename ${checkpoint} .pt)
  
  tide bench-all \
    --model-path ${checkpoint} \
    --output-dir results/batch_eval/${model_name}
done
```

## Environment Variables

```bash
# Set environment variables for common paths
export TIDE_DATA_DIR=/path/to/data
export TIDE_RESULTS_DIR=/path/to/results
export TIDE_CONFIG=configs/production.yaml

# Use in commands
tide train --config ${TIDE_CONFIG} --output-dir ${TIDE_RESULTS_DIR}/run1
```

## Troubleshooting

```bash
# Debug mode with verbose output
tide train --verbose --dry-run --config configs/defaults.yaml

# Check GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Validate configuration
python -m tide_lite.utils.validate_config --config configs/defaults.yaml

# Clean up incomplete runs
rm -rf results/*/checkpoints/checkpoint_step_*.pt
```

## Performance Tips

1. **For faster training:**
   ```bash
   tide train --batch-size 64 --gradient-accumulation-steps 2 --use-amp
   ```

2. **For memory-limited GPUs:**
   ```bash
   tide train --batch-size 16 --max-seq-length 64 --gradient-checkpoint
   ```

3. **For CPU evaluation:**
   ```bash
   tide eval-stsb --device cpu --batch-size 8 --num-workers 4
   ```

4. **For large-scale retrieval:**
   ```bash
   tide eval-quora --index-type IVFFlat --batch-size 512 --fp16
   ```

## Monitoring

```bash
# Watch training progress
tail -f results/experiment1/training.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Track metrics
python -c "import json; print(json.load(open('results/experiment1/metrics_train.json')))"
```

## Export and Deployment

```bash
# Export model for serving
python -m tide_lite.export \
  --model-path results/best_model/checkpoints/final.pt \
  --output-path models/tide_lite_v1.onnx \
  --format onnx

# Test exported model
python -m tide_lite.test_export \
  --model-path models/tide_lite_v1.onnx \
  --test-text "Example sentence" \
  --timestamp "2024-01-15"
```

## Notes

- All commands support `--dry-run` flag for testing without execution
- Use `--verbose` for detailed logging
- Results are saved in JSON format for easy parsing
- The orchestrator ensures commands run in the correct sequence
- Ctrl+C gracefully stops execution at the current step
