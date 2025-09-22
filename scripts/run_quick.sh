#!/bin/bash
# Quick smoke test script for TIDE-Lite
# Runs minimal training + eval on CPU (2-5 minutes)

set -e  # Exit on error

echo "=================================================="
echo "TIDE-LITE CPU SMOKE TEST"
echo "=================================================="

# Check Python
echo "Python version:"
python --version

# Install dependencies if needed
echo -e "\n📦 Checking dependencies..."
pip install -q torch transformers datasets sentence-transformers faiss-cpu scikit-learn numpy pandas tqdm matplotlib peft scipy PyYAML

# Create output directory
OUTPUT_DIR="results/smoke_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo -e "\n🚀 Running quick training (1 epoch, small batch)..."
python scripts/train.py \
    --config configs/default.yaml \
    --output-dir $OUTPUT_DIR \
    --num-epochs 1 \
    --batch-size 8 \
    --eval-batch-size 16 \
    --save-every-n-steps 100 \
    --eval-every-n-steps 100 \
    --no-amp

echo -e "\n📊 Running evaluation on trained model..."
python scripts/evaluate.py \
    --model tide-lite \
    --checkpoint $OUTPUT_DIR/checkpoints \
    --task all \
    --output-dir $OUTPUT_DIR \
    --max-samples 100

echo -e "\n📈 Generating plots..."
python scripts/plot.py \
    --input-dir $OUTPUT_DIR \
    --output-dir $OUTPUT_DIR/plots

echo -e "\n✅ Smoke test complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Expected artifacts:"
echo "  • $OUTPUT_DIR/metrics_all.json"
echo "  • $OUTPUT_DIR/metrics_stsb_tide-lite.json"  
echo "  • $OUTPUT_DIR/metrics_quora_tide-lite.json"
echo "  • $OUTPUT_DIR/plots/fig_score_vs_dim.png"
echo "  • $OUTPUT_DIR/plots/fig_latency_vs_dim.png"
echo "=================================================="
