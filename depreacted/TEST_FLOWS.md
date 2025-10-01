# TIDE-Lite Test Flows

This repository provides multiple test flows to validate the TIDE-Lite implementation at different levels. Each test serves a specific purpose and takes different amounts of time.

## 📋 Test Hierarchy

| Test Level | Script | Duration | Purpose | Requires GPU |
|------------|--------|----------|---------|--------------|
| **Quick Test** | `run_quick_test.sh` | 30 seconds | Verify imports and basic functionality | No |
| **Smoke Test** | `run_smoke_test.sh` | 1-2 minutes | Test components with mock data | No |
| **E2E Smoke** | `run_e2e_smoke_test.sh` | 5-10 minutes | **Run actual training/eval with small data** | Optional |
| **Full Flow** | `run_full_flow.sh` | 2-6 hours | Complete training on full datasets | Recommended |

## 🚀 Quick Start

```bash
# Make scripts executable
chmod +x *.sh

# Run end-to-end smoke test (recommended)
./run_e2e_smoke_test.sh
```

## 🔍 Test Descriptions

### 1. Quick Component Test (`run_quick_test.sh`)
- **Purpose**: Verify all imports work and models can be created
- **What it does**:
  - Tests Python imports
  - Creates baseline and TIDE-Lite models
  - Verifies encoding functions work
  - Tests individual components (time encoding, gating)
- **When to use**: After installation or code changes
- **No data downloads required**

### 2. Basic Smoke Test (`run_smoke_test.sh`)
- **Purpose**: Component-level testing without actual training
- **What it does**:
  - Runs integration tests
  - Runs unit tests (if pytest available)
  - Tests training pipeline in dry-run mode
  - Attempts data loading (may fail on first run)
- **When to use**: To verify setup before training
- **May require initial dataset downloads**

### 3. End-to-End Smoke Test (`run_e2e_smoke_test.sh`) ⭐ RECOMMENDED
- **Purpose**: **Actually trains and evaluates the model with tiny datasets**
- **What it does**:
  - Loads real data (100 STS-B samples)
  - Trains TIDE-Lite for 2 epochs
  - Saves model checkpoints
  - Evaluates on STS-B test set
  - Computes real metrics (Spearman correlation)
  - Produces complete output directory with results
- **When to use**: **To verify the entire pipeline works correctly**
- **This is a real end-to-end test!**

### 4. Full Training Flow (`run_full_flow.sh`)
- **Purpose**: Production training with full datasets
- **What it does**:
  - Downloads complete datasets (STS-B, Quora, TimeQA)
  - Trains for multiple epochs with proper hyperparameters
  - Evaluates on all benchmarks
  - Generates plots and reports
  - Saves production-ready models
- **When to use**: For actual experiments and model development

## 📊 Expected Results

### E2E Smoke Test Results
After running `run_e2e_smoke_test.sh`, you should see:
```
outputs/smoke_test_YYYYMMDD_HHMMSS/
├── smoke_config.yaml      # Configuration used
├── training.log           # Training progress
├── model/                 # Trained model
│   ├── best.pt           # Best checkpoint
│   └── final_model/      # Final model files
├── eval_stsb.log         # Evaluation log
└── eval_stsb/
    └── results.json      # Metrics (Spearman ~0.5-0.7 expected)
```

### What Success Looks Like
✅ **E2E Smoke Test Passes** when:
- Training completes without errors
- Model files are saved
- STS-B evaluation produces Spearman correlation
- Results are saved to JSON

Example output:
```
📊 Smoke Test Results:
==================================================
✅ Training: COMPLETE
   Model files saved: 2
✅ STS-B Evaluation: COMPLETE
   Spearman: 0.6234
   Samples: 100
==================================================
```

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**
   - Run `run_quick_test.sh` first to identify import issues
   - Check Python path and package installation

2. **Data loading fails**
   - Network issues downloading datasets
   - Run with `trust_remote_code=True` for Quora
   - Temporal datasets (TimeQA/TempLAMA) are optional

3. **Out of memory**
   - Reduce batch size in smoke test
   - Use CPU if GPU memory insufficient

4. **Training doesn't converge**
   - Normal for smoke test (only 2 epochs)
   - Full training needed for good results

## 📈 Progression Path

1. **Start**: Run `run_quick_test.sh` to verify installation
2. **Validate**: Run `run_e2e_smoke_test.sh` to verify pipeline
3. **Develop**: Modify code and re-run E2E smoke test
4. **Production**: Run `run_full_flow.sh` for final results

## 💡 Tips

- The E2E smoke test is the most important validation
- It runs actual training, not just mocks
- Results in 5-10 minutes show if everything works
- Use smoke test outputs to debug issues
- Check logs in output directory for details

## 📝 Notes

- First run may take longer due to model downloads
- Datasets are cached in `./data/` directory
- All tests save outputs to `outputs/` directory
- GPU speeds up training but is not required for smoke tests
