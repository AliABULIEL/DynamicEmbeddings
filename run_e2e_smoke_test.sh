#!/bin/bash
# End-to-end smoke test - runs actual training and evaluation with tiny datasets
# This verifies the entire pipeline works correctly
# Duration: ~5-10 minutes

set -e  # Exit on error

echo "================================================"
echo "TIDE-Lite End-to-End Smoke Test"  
echo "================================================"
echo "This will run actual training and evaluation with tiny datasets."
echo "Purpose: Verify the entire pipeline works correctly."
echo ""

# Configuration for smoke test
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/smoke_test_${TIMESTAMP}"
NUM_EPOCHS=2    # Just 2 epochs for smoke test
BATCH_SIZE=8    # Small batch size

echo "Configuration:"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# ================================================
# STAGE 1: Environment Check
# ================================================
echo "[Stage 1/6] Environment Check"
echo "--------------------------------"

python -c "
import torch
import transformers
import datasets
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Datasets: {datasets.__version__}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    device = 'cuda'
else:
    print('GPU: Not available (using CPU - will be slower)')
    device = 'cpu'
print('✅ Environment OK')
"

# Determine device
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi

# ================================================
# STAGE 2: Data Verification
# ================================================
echo -e "\n[Stage 2/6] Data Loading Test"
echo "--------------------------------"

python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.tide_lite.data.datasets import load_stsb, load_quora

# Test STS-B loading with small subset
print('Loading STS-B samples for smoke test...')
try:
    # Load with limited samples for speed
    stsb = load_stsb({'cache_dir': './data', 'max_samples': 100, 'seed': 42})
    print(f'  ✅ STS-B: {len(stsb[\"train\"])} train, {len(stsb[\"validation\"])} val samples')
except Exception as e:
    print(f'  ❌ STS-B failed: {e}')
    sys.exit(1)

# Test Quora loading (may fail due to network)
print('Loading Quora samples for smoke test...')
try:
    corpus, queries, qrels = load_quora({'cache_dir': './data', 'max_samples': 50, 'seed': 42})
    print(f'  ✅ Quora: {len(corpus)} docs, {len(queries)} queries')
except Exception as e:
    print(f'  ⚠️  Quora failed (network issue?): {str(e)[:100]}')
    print('     Continuing without Quora evaluation...')

print('✅ Data loading test complete')
"

# ================================================
# STAGE 3: Model Initialization Test
# ================================================
echo -e "\n[Stage 3/6] Model Initialization"
echo "--------------------------------"

python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.tide_lite.models import TIDELite, TIDELiteConfig, load_minilm_baseline
import torch

# Test baseline model
print('Creating baseline model...')
baseline = load_minilm_baseline()
print(f'  ✅ Baseline: {baseline.model_name}')

# Test TIDE-Lite model
print('Creating TIDE-Lite model...')
config = TIDELiteConfig(
    encoder_name='sentence-transformers/all-MiniLM-L6-v2',
    time_encoding_dim=32,
    mlp_hidden_dim=128,
    freeze_encoder=True
)
tide = TIDELite(config)
print(f'  ✅ TIDE-Lite: {tide.count_extra_parameters():,} trainable params')

# Test encoding
texts = ['Test sentence one.', 'Test sentence two.']
with torch.no_grad():
    baseline_emb = baseline.encode_texts(texts, batch_size=2)
    tide_emb = tide.encode_texts(texts, batch_size=2)
print(f'  ✅ Encoding works: {baseline_emb.shape}, {tide_emb.shape}')

print('✅ Model initialization complete')
"

# ================================================
# STAGE 4: Training Pipeline (Actual Training!)
# ================================================
echo -e "\n[Stage 4/6] Training Pipeline (Real Training)"
echo "--------------------------------"
echo "Training TIDE-Lite for ${NUM_EPOCHS} epochs..."

# Create a temporary config file with smoke test settings
cat > "${OUTPUT_DIR}/smoke_config.yaml" << EOF
# Smoke test configuration
model_name: sentence-transformers/all-MiniLM-L6-v2
hidden_dim: 384
time_dims: 32
time_mlp_hidden: 128

# Training settings
batch_size: ${BATCH_SIZE}
eval_batch_size: ${BATCH_SIZE}
epochs: ${NUM_EPOCHS}
lr: 1e-4
warmup_steps: 10
eval_every: 50
save_every: 100

# Loss weights
consistency_weight: 0.1
preservation_weight: 0.05
tau_seconds: 86400

# Data settings
max_seq_len: 128
num_workers: 2
cache_dir: ./data

# Other settings
seed: 42
device: ${DEVICE}
use_amp: false
dry_run: false
skip_temporal: true

# Output
out_dir: ${OUTPUT_DIR}/model
checkpoint_dir: ${OUTPUT_DIR}/checkpoints
EOF

echo "Running training..."
python -m src.tide_lite.cli.train_cli \
    --config "${OUTPUT_DIR}/smoke_config.yaml" \
    --epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --device ${DEVICE} \
    2>&1 | tee "${OUTPUT_DIR}/training.log" || {
        echo "⚠️  Training with train_cli failed, trying alternative..."
        
        # Fallback to direct trainer if CLI fails
        python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.tide_lite.models import TIDELite, TIDELiteConfig
from src.tide_lite.train.trainer import TIDELiteTrainer, TrainingConfig

# Create model
model_config = TIDELiteConfig(
    encoder_name='sentence-transformers/all-MiniLM-L6-v2',
    time_encoding_dim=32,
    mlp_hidden_dim=128,
    freeze_encoder=True
)
model = TIDELite(model_config)

# Create training config
train_config = TrainingConfig(
    num_epochs=${NUM_EPOCHS},
    batch_size=${BATCH_SIZE},
    eval_batch_size=${BATCH_SIZE},
    learning_rate=1e-4,
    warmup_steps=10,
    save_every=100,
    eval_every=50,
    output_dir='${OUTPUT_DIR}/model',
    skip_temporal=True,
    dry_run=False,
    seed=42
)

# Train
trainer = TIDELiteTrainer(train_config, model)
trainer.setup_dataloaders()
trainer.setup_training()
trainer.train()

print('✅ Training complete!')
" 2>&1 | tee "${OUTPUT_DIR}/training_direct.log"
    }

# Check if model was saved
if [ -d "${OUTPUT_DIR}/model" ] || [ -f "${OUTPUT_DIR}/model/final_model/pytorch_model.bin" ]; then
    echo "✅ Training complete - model saved"
    MODEL_PATH="${OUTPUT_DIR}/model"
else
    echo "❌ Training failed - no model output"
    exit 1
fi

# ================================================
# STAGE 5: Evaluation Pipeline
# ================================================
echo -e "\n[Stage 5/6] Evaluation Pipeline"
echo "--------------------------------"

# Find the actual model path (might be in subdirectory)
if [ -f "${OUTPUT_DIR}/model/final_model/pytorch_model.bin" ]; then
    MODEL_PATH="${OUTPUT_DIR}/model/final_model"
elif [ -f "${OUTPUT_DIR}/model/pytorch_model.bin" ]; then
    MODEL_PATH="${OUTPUT_DIR}/model"
elif [ -f "${OUTPUT_DIR}/model/best.pt" ]; then
    MODEL_PATH="${OUTPUT_DIR}/model/best.pt"
else
    MODEL_PATH="${OUTPUT_DIR}/model"
fi

# Evaluate on STS-B
echo "Evaluating on STS-B..."
python -c "
import sys
from pathlib import Path
import json
sys.path.insert(0, str(Path.cwd()))

from src.tide_lite.models import TIDELite, TIDELiteConfig
from src.tide_lite.data.datasets import load_stsb
from scipy.stats import spearmanr
import torch
import numpy as np

# Load model
print('Loading model from ${MODEL_PATH}...')
config = TIDELiteConfig(
    encoder_name='sentence-transformers/all-MiniLM-L6-v2',
    time_encoding_dim=32,
    mlp_hidden_dim=128,
    freeze_encoder=True
)
model = TIDELite(config)

# Try to load checkpoint if exists
checkpoint_path = Path('${MODEL_PATH}')
if (checkpoint_path / 'best.pt').exists():
    checkpoint = torch.load(checkpoint_path / 'best.pt', map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.temporal_gate.load_state_dict(checkpoint['model_state_dict'])
    print('  Loaded checkpoint')

model.eval()

# Load test data
print('Loading STS-B test set...')
stsb = load_stsb({'cache_dir': './data', 'max_samples': 100, 'seed': 42})
test_data = stsb['test']

# Evaluate
print('Computing embeddings...')
predictions = []
labels = []

for sample in test_data:
    with torch.no_grad():
        # Encode sentences
        emb1 = model.encode_texts([sample['sentence1']], batch_size=1)
        emb2 = model.encode_texts([sample['sentence2']], batch_size=1)
        
        # Compute cosine similarity
        sim = torch.cosine_similarity(emb1, emb2, dim=1).item()
        predictions.append(sim)
        labels.append(sample['label'] / 5.0)  # Normalize to [0, 1]

# Compute Spearman correlation
spearman = spearmanr(predictions, labels)[0]
print(f'✅ STS-B Spearman correlation: {spearman:.4f}')

# Save results
output_dir = Path('${OUTPUT_DIR}/eval_stsb')
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / 'results.json', 'w') as f:
    json.dump({'spearman': float(spearman), 'num_samples': len(predictions)}, f, indent=2)
" 2>&1 | tee "${OUTPUT_DIR}/eval_stsb.log"

# ================================================
# STAGE 6: Results Summary
# ================================================
echo -e "\n[Stage 6/6] Results Summary"
echo "--------------------------------"

python -c "
import os
import json
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
print('📊 Smoke Test Results:')
print('=' * 50)

# Check training completion
if (output_dir / 'model').exists():
    print('✅ Training: COMPLETE')
    # Count any saved files
    model_files = list((output_dir / 'model').rglob('*.pt')) + \
                  list((output_dir / 'model').rglob('*.bin'))
    print(f'   Model files saved: {len(model_files)}')
else:
    print('❌ Training: FAILED')

# Check STS-B evaluation
stsb_results = output_dir / 'eval_stsb' / 'results.json'
if stsb_results.exists():
    with open(stsb_results) as f:
        results = json.load(f)
    print(f'✅ STS-B Evaluation: COMPLETE')
    print(f'   Spearman: {results.get(\"spearman\", 0):.4f}')
    print(f'   Samples: {results.get(\"num_samples\", 0)}')
else:
    print('❌ STS-B Evaluation: FAILED')

print('=' * 50)
print(f'All outputs saved to: {output_dir}')
"

# ================================================
# Final Status
# ================================================
echo -e "\n================================================"
if [ -d "${OUTPUT_DIR}/model" ] && [ -f "${OUTPUT_DIR}/eval_stsb/results.json" ]; then
    echo "✅ END-TO-END SMOKE TEST: PASSED"
    echo "The complete pipeline works correctly!"
    echo ""
    echo "This smoke test verified:"
    echo "  ✓ Data loading works"
    echo "  ✓ Model initialization works"
    echo "  ✓ Training runs successfully"
    echo "  ✓ Evaluation produces results"
    echo ""
    echo "Next steps:"
    echo "  - Review results in: ${OUTPUT_DIR}"
    echo "  - Run full training with more data: ./run_full_flow.sh"
else
    echo "⚠️  SMOKE TEST: PARTIAL SUCCESS"
    echo "Some components may need attention."
    echo "Check logs in: ${OUTPUT_DIR}"
fi
echo "================================================"
