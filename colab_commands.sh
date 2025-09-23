#!/bin/bash
# EXACT COMMANDS TO RUN IN GOOGLE COLAB
# Just copy and paste these into Colab cells

# ============================================
# CELL 1: Setup (1 minute)
# ============================================
echo "Setting up environment..."
git clone https://github.com/YOUR_USERNAME/DynamicEmbeddings.git
cd DynamicEmbeddings
pip install -q -r requirements.txt
echo "✅ Setup complete!"

# ============================================
# CELL 2: Quick Test (5 minutes)
# ============================================
echo "Running quick test..."
python scripts/train.py \
    --config configs/smoke.yaml \
    --device cuda \
    --batch-size 64

# ============================================
# CELL 3: Full Training (30 minutes)
# ============================================
echo "Starting full training..."
python scripts/train.py \
    --encoder-name "sentence-transformers/all-MiniLM-L6-v2" \
    --time-encoding-dim 64 \
    --mlp-hidden-dim 256 \
    --batch-size 256 \
    --num-epochs 20 \
    --learning-rate 2e-5 \
    --temporal-weight 0.15 \
    --use-amp \
    --output-dir results/production

# ============================================
# CELL 4: Evaluate (2 minutes)
# ============================================
echo "Running evaluation..."
python scripts/run_evaluation.py --checkpoint-dir results/production

# ============================================
# CELL 5: Show Results
# ============================================
echo "Results:"
cat results/production/eval/eval_results.json | python -m json.tool

# ============================================
# CELL 6: Download Results
# ============================================
echo "Packaging results..."
zip -r tide_results.zip results/
# Then use: files.download('tide_results.zip')
