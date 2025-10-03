#!/bin/bash
# Complete Pipeline - Copy and paste these commands in Colab

# ============================================================================
# FULL PIPELINE - RUN THESE COMMANDS IN ORDER
# ============================================================================

# Step 1: Download Data (5-10 min)
echo "📥 Step 1/7: Downloading arXiv data..."
python3 scripts/download_arxiv_data.py --max-papers 10000

# Step 2: Prepare Data into Time Buckets (5-10 min)
echo "🔧 Step 2/7: Preparing data..."
python3 -m temporal_lora.cli prepare-data \
  --max-per-bucket 2000 \
  --balance-per-bin

# Step 3: Train LoRA Adapters (20-30 min)
echo "🎓 Step 3/7: Training LoRA adapters..."
python3 -m temporal_lora.cli train-adapters \
  --mode lora \
  --hard-temporal-negatives \
  --neg-k 4 \
  --lora-r 16 \
  --epochs 2 \
  --batch-size 16

# Step 4: Build FAISS Indexes (2-5 min)
echo "🗂️  Step 4/7: Building FAISS indexes..."
python3 -m temporal_lora.cli build-indexes --lora

# Step 5: Run Comprehensive Benchmark (5-10 min) 🔥 NEW!
echo "📊 Step 5/7: Running benchmark comparison..."
python3 -m temporal_lora.cli benchmark --report

# Step 6: Generate Additional Visualizations (2-3 min) [OPTIONAL]
echo "📈 Step 6/7: Creating visualizations..."
python3 -m temporal_lora.cli visualize

# Step 7: Export All Results (1 min) [OPTIONAL]
echo "📦 Step 7/7: Exporting deliverables..."
python3 -m temporal_lora.cli export-deliverables

# ============================================================================
# DONE!
# ============================================================================

echo ""
echo "======================================================================"
echo "✅ PIPELINE COMPLETE!"
echo "======================================================================"
echo ""
echo "📁 Check your results:"
echo "   deliverables/results/benchmark/BENCHMARK_REPORT.md"
echo "   deliverables/results/benchmark/figures/benchmark_comparison.png"
echo "   deliverables/results/benchmark/figures/improvement_heatmap.png"
echo ""
echo "🔥 Look for improvement highlights (🔥 icons = >5% gains)"
echo "======================================================================"
