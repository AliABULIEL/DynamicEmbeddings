#!/bin/bash
# Quick test script for hard negatives functionality

cd /Users/aliab/Desktop/DynamicEmbeddings

echo "🧪 Running Hard Negatives Tests..."
echo ""

python test_hard_negatives.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✅ SUCCESS! You can now retrain:"
    echo "============================================"
    echo ""
    echo "python -m temporal_lora.cli train-adapters \\"
    echo "    --mode lora \\"
    echo "    --epochs 5 \\"
    echo "    --lora-r 32 \\"
    echo "    --hard-temporal-negatives"
    echo ""
else
    echo ""
    echo "============================================"
    echo "❌ FAILED - Check errors above"
    echo "============================================"
fi

exit $exit_code
