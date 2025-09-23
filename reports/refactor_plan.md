# TIDE-Lite Refactor Plan

## Executive Summary
Major refactoring to achieve production-quality codebase with real temporal datasets, unified interfaces, and single-responsibility modules.

## Current State Inventory

### `/src/tide_lite/` Structure
```
cli/           # Multiple scattered CLI scripts (9 files)
├── train_cli.py, eval_*.py  # Separate CLIs per task
├── tide.py                   # Orchestrator (underutilized)
└── aggregate_cli.py, plots_cli.py, report_cli.py

data/          # Mixed responsibilities
├── datasets.py              # SYNTHETIC timestamps (must fix)
├── temporal_datasets.py    # Placeholder, no real temporal data
└── dataloaders.py, collate.py

models/        # Clean structure
├── tide_lite.py            # Main model
└── baselines.py            # Baseline models

train/         # Duplicated loss functions
├── losses.py               # Main losses
└── advanced_losses.py      # Duplicate functionality

eval/          # Inconsistent naming
├── eval_stsb.py
├── eval_temporal.py
└── retrieval_quora.py      # Different naming pattern

evaluation/    # EMPTY duplicate directory
utils/         # Duplicate config handling
config/        # Another config module
```

### `/configs/` - Too Many Configs (7 files)
- `default.yaml`, `defaults.yaml` - DUPLICATES
- `balanced.yaml`, `bench_small.yaml`, `colab.yaml`, `realistic_production.yaml`, `smoke.yaml`
- Violates "one default + one Colab" rule

### `/scripts/` - Chaos (25 files!)
- **Duplicates**: `generate_report.py`, `generate_report_fixed.py`
- **Redundant**: `evaluate.py`, `evaluate_checkpoint.py`, `run_evaluation.py`
- **Scattered**: Multiple train scripts (`train.py`, `train_simple.py`, `train_temporal.py`)
- **Dead code**: `test_dtype.py`, `verify_fixes.py`

## Critical Issues

### 1. Synthetic Data Usage
- `datasets.py`: `_generate_synthetic_timestamps()` creates fake timestamps
- No real temporal datasets (TimeQA/TempLAMA not implemented)
- Temporal evaluation meaningless with synthetic data

### 2. Duplicate/Dead Code
- Two evaluation directories (`eval/` and `evaluation/`)
- Multiple report generators
- Scattered CLI scripts instead of unified orchestrator
- Config handling in both `utils/` and `config/`

### 3. Inconsistent Interfaces
- Models don't share unified embedding API
- Different evaluation scripts with different interfaces
- No consistent logging setup

## Refactor Decisions

### KEEP (with modifications)
- `src/tide_lite/models/` - Clean, just needs unified interface
- `src/tide_lite/cli/tide.py` - Expand as main orchestrator
- `configs/default.yaml` - As single default config
- `notebooks/Colab_TIDE_Lite.ipynb` - Update for new structure

### CHANGE
1. **Unify CLI** → Single `tide` orchestrator with subcommands
2. **Real temporal data** → Implement TimeQA/TempLAMA loaders
3. **Merge losses** → Single `losses.py` module
4. **Consolidate eval** → Single `evaluation/` module with consistent API
5. **Single config system** → Use `config/` module only

### REMOVE
- `src/tide_lite/evaluation/` - Empty duplicate
- `src/tide_lite/train/advanced_losses.py` - Merge into losses.py
- `src/tide_lite/utils/config.py` - Use config/ module
- `configs/`: Keep only `default.yaml` and `colab.yaml`
- `scripts/`: Remove 20+ redundant files, keep only essential
- All synthetic timestamp generation code

## File Moves & Renames
```bash
# Consolidate evaluation
mv src/tide_lite/eval/* src/tide_lite/evaluation/
rm -rf src/tide_lite/eval/

# Clean configs
rm configs/defaults.yaml configs/balanced.yaml configs/bench_small.yaml
rm configs/realistic_production.yaml configs/smoke.yaml

# Clean scripts - keep only essentials
rm scripts/generate_report_fixed.py scripts/evaluate_checkpoint.py
rm scripts/run_evaluation.py scripts/train_simple.py scripts/train_temporal.py
rm scripts/test_*.py scripts/verify_fixes.py
```

## Import Updates Required
1. Change all `from ..eval.` to `from ..evaluation.`
2. Update config imports to use `config.schema`
3. Remove references to synthetic timestamp functions
4. Update CLI scripts to use unified orchestrator

## Implementation Phases

### Phase 1: Clean Structure (Immediate)
- Remove duplicates and dead code
- Consolidate directories
- Fix import paths

### Phase 2: Real Data (Priority)
- Implement TimeQA/TempLAMA loaders
- Remove synthetic timestamp generation
- Add real temporal evaluation

### Phase 3: Unified Interface
- Create base embedding interface
- Ensure all models implement it
- Standardize evaluation API

### Phase 4: Production CLI
- Expand tide.py orchestrator
- Add all subcommands with --dry-run default
- Remove individual CLI scripts

## Risk Assessment

### Low Risk
- Removing empty/duplicate directories
- Consolidating config files
- Cleaning dead scripts

### Medium Risk
- Merging loss modules (need careful testing)
- Updating all import paths
- Consolidating evaluation code

### High Risk
- Replacing synthetic with real temporal data (core functionality change)
- Unified model interface (affects all models)
- CLI consolidation (user-facing change)

## Success Metrics
- Zero duplicate files
- All models share same embedding API
- Real temporal dataset (TimeQA/TempLAMA) working
- Single `tide` CLI handles all operations
- All modules have proper logging (no print statements)
- Type hints and docstrings complete

## Next Steps
1. Execute Phase 1 cleanup immediately
2. Implement TimeQA/TempLAMA loaders
3. Create unified model interface
4. Consolidate CLI into single orchestrator
5. Update documentation and tests

---
*Generated: 2024-12-19*
*Target: Production-ready codebase with real temporal evaluation*
