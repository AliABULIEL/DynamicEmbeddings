#!/bin/bash
# Quick smoke test flow - validates everything works without full training
# Takes ~1-2 minutes to run

set -e  # Exit on error

echo "================================================"
echo "TIDE-Lite Smoke Test Flow"
echo "================================================"

# 1. Setup environment
echo "1. Checking environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 2. Run integration smoke test
echo -e "\n2. Running integration smoke test..."
python tests/test_integration_smoke.py

# 3. Run unit tests
echo -e "\n3. Running unit tests..."
pytest tests/test_models.py -v
pytest tests/test_data.py -v
pytest tests/test_config.py -v
pytest tests/test_time_encoding.py -v

# 4. Dry-run training (no actual training)
echo -e "\n4. Testing training pipeline (dry-run)..."
python -m src.tide_lite.cli.train \
    --config configs/tide_lite.yaml \
    --dry-run \
    --output-dir outputs/smoke_test \
    --num-epochs 1 \
    --batch-size 4 \
    --max-samples 16

# 5. Test model creation and encoding
echo -e "\n5. Testing model APIs..."
python -c "
from src.tide_lite.models import load_minilm_baseline, TIDELite, TIDELiteConfig
import torch

# Test baseline
baseline = load_minilm_baseline()
texts = ['Test sentence one.', 'Test sentence two.']
emb = baseline.encode_texts(texts, batch_size=2)
print(f'Baseline embeddings shape: {emb.shape}')

# Test TIDE-Lite
config = TIDELiteConfig(
    encoder_name='sentence-transformers/all-MiniLM-L6-v2',
    freeze_encoder=True
)
tide = TIDELite(config)
emb = tide.encode_texts(texts, batch_size=2)
print(f'TIDE-Lite embeddings shape: {emb.shape}')
print(f'Extra parameters: {tide.count_extra_parameters():,}')
"

# 6. Verify data loading
echo -e "\n6. Testing data loaders..."
python -c "
from src.tide_lite.data.datasets import load_stsb, load_quora

# Test STS-B
cfg = {'cache_dir': './data', 'max_samples': 10, 'seed': 42}
stsb = load_stsb(cfg)
print(f'STS-B loaded: {len(stsb[\"train\"])} train samples')

# Test Quora
corpus, queries, qrels = load_quora(cfg)
print(f'Quora loaded: {len(corpus)} docs, {len(queries)} queries')
"

echo -e "\n================================================"
echo "✅ Smoke test passed! System ready for full training."
echo "================================================"
