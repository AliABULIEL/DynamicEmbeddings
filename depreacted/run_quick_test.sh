#!/bin/bash
# Quick test without downloading external datasets
# Tests core components only - runs in ~30 seconds

set -e  # Exit on error

echo "================================================"
echo "TIDE-Lite Quick Component Test"
echo "================================================"

# 1. Test imports and basic functionality
echo "1. Testing imports and model creation..."
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

print('  Importing modules...')
from src.tide_lite.models import TIDELite, TIDELiteConfig, load_minilm_baseline
from src.tide_lite.train.trainer import TIDELiteTrainer, TrainingConfig

print('  Creating baseline model...')
baseline = load_minilm_baseline()
print(f'  ✓ Baseline created: {baseline.model_name}')

print('  Creating TIDE-Lite model...')
config = TIDELiteConfig(
    encoder_name='sentence-transformers/all-MiniLM-L6-v2',
    freeze_encoder=True
)
tide = TIDELite(config)
print(f'  ✓ TIDE-Lite created: {tide.count_extra_parameters():,} extra params')

print('  Testing encoding...')
import torch
texts = ['Test sentence one.', 'Test sentence two.']
with torch.no_grad():
    emb = baseline.encode_texts(texts, batch_size=2)
    print(f'  ✓ Baseline embeddings: {emb.shape}')
    emb = tide.encode_texts(texts, batch_size=2)
    print(f'  ✓ TIDE-Lite embeddings: {emb.shape}')

print('✅ All core components working!')
"

# 2. Test training components (dry-run only)
echo -e "\n2. Testing training pipeline (dry-run)..."
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.tide_lite.models import TIDELite, TIDELiteConfig
from src.tide_lite.train.trainer import TIDELiteTrainer, TrainingConfig

config = TrainingConfig(
    num_epochs=1,
    batch_size=4,
    dry_run=True,  # Dry run only
    output_dir='./outputs/test_dry_run'
)

model_config = TIDELiteConfig(
    encoder_name='sentence-transformers/all-MiniLM-L6-v2',
    freeze_encoder=True
)
model = TIDELite(model_config)

trainer = TIDELiteTrainer(config, model)
trainer.train()

print('✅ Training pipeline works (dry-run tested)!')
"

# 3. Test individual model components
echo -e "\n3. Testing model components..."
python3 -c "
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path.cwd()))

from src.tide_lite.models.tide_lite import SinusoidalTimeEncoding, TemporalGatingMLP

# Test time encoding
encoder = SinusoidalTimeEncoding(encoding_dim=32)
timestamps = torch.tensor([1609459200.0, 1640995200.0, 1672531200.0])
encoding = encoder(timestamps)
print(f'  ✓ Time encoding: {encoding.shape}')

# Test temporal gating
gate = TemporalGatingMLP(
    input_dim=32,
    hidden_dim=128,
    output_dim=384,
    activation='sigmoid'
)
time_encoding = torch.randn(4, 32)
gates = gate(time_encoding)
print(f'  ✓ Temporal gating: {gates.shape}')

print('✅ All model components working!')
"

# 4. Optional: Test data loading with small samples
echo -e "\n4. Testing data loading (optional, may require downloads)..."
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

try:
    from src.tide_lite.data.datasets import load_stsb
    cfg = {'cache_dir': './data', 'max_samples': 4, 'seed': 42}
    dataset = load_stsb(cfg)
    print(f'  ✓ STS-B loaded: {len(dataset[\"train\"])} samples')
except Exception as e:
    print(f'  ⚠ STS-B loading skipped: {e}')

print('Note: Full dataset tests available in run_smoke_test.sh')
"

echo -e "\n================================================"
echo "✅ Quick component test complete!"
echo "All core TIDE-Lite components are working."
echo ""
echo "Next steps:"
echo "  - Run full smoke test: ./run_smoke_test.sh"
echo "  - Run full training: ./run_full_flow.sh"
echo "================================================"
