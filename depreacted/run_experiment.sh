#!/bin/bash
# TIDE-Lite Training & Evaluation Command Reference
# Complete guide for running experiments

echo "================================================"
echo "TIDE-LITE EXPERIMENT RUNNER"
echo "================================================"

# Function to print section headers
print_section() {
    echo ""
    echo "### $1 ###"
    echo ""
}

# Parse command line argument
if [ $# -eq 0 ]; then
    echo "Usage: ./run_experiment.sh [smoke|quick|full|custom|eval|ablation]"
    echo ""
    echo "Options:"
    echo "  smoke    - Quick CPU test (5 min)"
    echo "  quick    - Fast GPU training (30 min)"  
    echo "  full     - Complete training (2 hours)"
    echo "  custom   - Interactive configuration"
    echo "  eval     - Evaluate existing checkpoint"
    echo "  ablation - Run ablation studies"
    exit 1
fi

MODE=$1

case $MODE in
    smoke)
        print_section "SMOKE TEST - Quick Validation"
        echo "Running minimal training for testing setup..."
        echo "Expected time: ~5 minutes on CPU"
        echo ""
        
        python3 scripts/train.py \
            --config configs/smoke.yaml \
            --output-dir outputs/smoke_$(date +%Y%m%d_%H%M%S)
        
        echo ""
        echo "✅ Smoke test complete! Check outputs/ for results"
        ;;
        
    quick)
        print_section "QUICK TRAINING - Fast Results"
        echo "Running accelerated training with reduced epochs..."
        echo "Expected time: ~30 minutes on GPU"
        echo ""
        
        python3 scripts/train.py \
            --encoder-name "sentence-transformers/all-MiniLM-L6-v2" \
            --batch-size 64 \
            --num-epochs 3 \
            --learning-rate 5e-5 \
            --temporal-weight 0.1 \
            --use-amp \
            --eval-every-n-steps 200 \
            --output-dir results/quick_$(date +%Y%m%d_%H%M%S)
        
        echo ""
        echo "✅ Quick training complete!"
        ;;
        
    full)
        print_section "FULL TRAINING - Best Performance"
        echo "Running complete training pipeline..."
        echo "Expected time: ~2 hours on GPU"
        echo ""
        
        # Create experiment directory
        EXP_DIR="results/full_$(date +%Y%m%d_%H%M%S)"
        
        # Run training
        python3 scripts/train.py \
            --encoder-name "sentence-transformers/all-MiniLM-L6-v2" \
            --batch-size 128 \
            --num-epochs 10 \
            --learning-rate 2e-5 \
            --warmup-steps 500 \
            --temporal-weight 0.15 \
            --preservation-weight 0.05 \
            --mlp-hidden-dim 128 \
            --use-amp \
            --save-every-n-steps 500 \
            --eval-every-n-steps 250 \
            --output-dir $EXP_DIR
        
        echo ""
        echo "Training complete! Running evaluation..."
        
        # Evaluate final checkpoint
        python3 scripts/run_evaluation.py \
            --checkpoint-dir $EXP_DIR
        
        echo ""
        echo "✅ Full pipeline complete!"
        echo "📊 Results: $EXP_DIR/report.html"
        ;;
        
    custom)
        print_section "CUSTOM CONFIGURATION"
        echo "Interactive experiment setup..."
        echo ""
        
        # Get user inputs
        read -p "Encoder name [sentence-transformers/all-MiniLM-L6-v2]: " ENCODER
        ENCODER=${ENCODER:-sentence-transformers/all-MiniLM-L6-v2}
        
        read -p "Batch size [32]: " BATCH
        BATCH=${BATCH:-32}
        
        read -p "Number of epochs [3]: " EPOCHS
        EPOCHS=${EPOCHS:-3}
        
        read -p "Learning rate [5e-5]: " LR
        LR=${LR:-5e-5}
        
        read -p "Temporal weight [0.1]: " TEMP_W
        TEMP_W=${TEMP_W:-0.1}
        
        read -p "MLP hidden dimension [128]: " MLP_DIM
        MLP_DIM=${MLP_DIM:-128}
        
        read -p "Output directory [results/custom]: " OUT_DIR
        OUT_DIR=${OUT_DIR:-results/custom}
        
        echo ""
        echo "Starting training with custom configuration..."
        
        python3 scripts/train.py \
            --encoder-name "$ENCODER" \
            --batch-size $BATCH \
            --num-epochs $EPOCHS \
            --learning-rate $LR \
            --temporal-weight $TEMP_W \
            --mlp-hidden-dim $MLP_DIM \
            --output-dir "$OUT_DIR"
        ;;
        
    eval)
        print_section "EVALUATION MODE"
        echo "Evaluate existing checkpoints..."
        echo ""
        
        read -p "Enter checkpoint directory path: " CKPT_DIR
        
        if [ ! -d "$CKPT_DIR" ]; then
            echo "Error: Directory $CKPT_DIR not found!"
            exit 1
        fi
        
        python3 scripts/run_evaluation.py \
            --checkpoint-dir "$CKPT_DIR"
        
        echo ""
        echo "✅ Evaluation complete!"
        echo "📊 Report: $CKPT_DIR/report.html"
        ;;
        
    ablation)
        print_section "ABLATION STUDIES"
        echo "Running systematic ablation experiments..."
        echo ""
        
        BASE_DIR="results/ablations_$(date +%Y%m%d_%H%M%S)"
        mkdir -p $BASE_DIR
        
        # 1. No temporal component (baseline)
        echo "1/5: Training without temporal module..."
        python3 scripts/train.py \
            --config configs/smoke.yaml \
            --temporal-weight 0.0 \
            --output-dir $BASE_DIR/no_temporal &
        
        # 2. Different temporal weights
        echo "2/5: Testing temporal weight = 0.05..."
        python3 scripts/train.py \
            --config configs/smoke.yaml \
            --temporal-weight 0.05 \
            --output-dir $BASE_DIR/temp_0.05 &
        
        echo "3/5: Testing temporal weight = 0.2..."
        python3 scripts/train.py \
            --config configs/smoke.yaml \
            --temporal-weight 0.2 \
            --output-dir $BASE_DIR/temp_0.2 &
        
        # 4. Different MLP sizes
        echo "4/5: Testing MLP dimension = 64..."
        python3 scripts/train.py \
            --config configs/smoke.yaml \
            --mlp-hidden-dim 64 \
            --output-dir $BASE_DIR/mlp_64 &
        
        echo "5/5: Testing MLP dimension = 256..."
        python3 scripts/train.py \
            --config configs/smoke.yaml \
            --mlp-hidden-dim 256 \
            --output-dir $BASE_DIR/mlp_256 &
        
        # Wait for all jobs
        wait
        
        echo ""
        echo "Ablations complete! Generating comparison report..."
        
        # Run evaluation on all ablations
        for dir in $BASE_DIR/*/; do
            echo "Evaluating $dir..."
            python3 scripts/run_evaluation.py --checkpoint-dir "$dir"
        done
        
        # Create summary
        python3 -c "
import json
import glob
from pathlib import Path

base = '$BASE_DIR'
results = {}

for exp_dir in glob.glob(f'{base}/*/'):
    name = Path(exp_dir).name
    eval_file = Path(exp_dir) / 'eval' / 'eval_results.json'
    if eval_file.exists():
        with open(eval_file) as f:
            results[name] = json.load(f)

print('\n=== ABLATION RESULTS ===')
print('Experiment            | Spearman | Pearson  | MSE')
print('-'*50)
for name, metrics in sorted(results.items()):
    if 'stsb' in metrics:
        m = metrics['stsb']
        print(f'{name:20} | {m.get(\"spearman\", 0):.4f}  | {m.get(\"pearson\", 0):.4f}  | {m.get(\"mse\", 0):.4f}')

# Save summary
with open(f'{base}/ablation_summary.json', 'w') as f:
    json.dump(results, f, indent=2)
        "
        
        echo ""
        echo "✅ Ablation study complete!"
        echo "📊 Results: $BASE_DIR/ablation_summary.json"
        ;;
        
    *)
        echo "Unknown mode: $MODE"
        echo "Use: smoke, quick, full, custom, eval, or ablation"
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "EXPERIMENT COMPLETE"
echo "================================================"
