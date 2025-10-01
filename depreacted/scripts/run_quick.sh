#!/bin/bash

# TIDE-Lite Quick Run Script
# Usage: ./scripts/run_quick.sh [cpu|gpu|test]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default mode
MODE=${1:-cpu}

echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     TIDE-Lite Quick Runner v1.0       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""

# Function to check if in virtual environment
check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        echo -e "${YELLOW}⚠ Not in virtual environment. Creating one...${NC}"
        python -m venv venv
        source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
    else
        echo -e "${GREEN}✓ Virtual environment active${NC}"
    fi
}

# Function to install dependencies
install_deps() {
    echo -e "${YELLOW}📦 Installing dependencies...${NC}"
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Function to run CPU smoke test
run_cpu_smoke() {
    echo -e "\n${YELLOW}🔬 Running CPU smoke test (2-5 min)...${NC}"
    
    # Training
    echo -e "${YELLOW}→ Training (1 epoch, small batch)...${NC}"
    python -m src.tide_lite.cli.train_cli \
        --batch-size 8 \
        --num-epochs 1 \
        --warmup-steps 10 \
        --save-every-n-steps 50 \
        --eval-every-n-steps 50 \
        --output-dir results/smoke_cpu \
        --dry-run
    
    # Evaluation
    echo -e "${YELLOW}→ Evaluating STS-B...${NC}"
    python -m src.tide_lite.cli.eval_stsb_cli \
        --model-path results/smoke_cpu \
        --batch-size 16 \
        --dry-run
    
    # Plots
    echo -e "${YELLOW}→ Generating plots...${NC}"
    python scripts/plot.py \
        --output-dir outputs/smoke \
        --dry-run
    
    echo -e "${GREEN}✓ CPU smoke test complete!${NC}"
}

# Function to run GPU test
run_gpu_test() {
    echo -e "\n${YELLOW}🚀 Running GPU test (10-15 min)...${NC}"
    
    # Check CUDA
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
    echo -e "${GREEN}✓ CUDA available${NC}"
    
    # Training with AMP
    echo -e "${YELLOW}→ Training (3 epochs, AMP enabled)...${NC}"
    python -m src.tide_lite.cli.train_cli \
        --batch-size 32 \
        --num-epochs 3 \
        --use-amp \
        --warmup-steps 100 \
        --output-dir results/gpu_quick
    
    # Full evaluation
    echo -e "${YELLOW}→ Running evaluation suite...${NC}"
    python -m src.tide_lite.cli.eval_stsb_cli \
        --model-path results/gpu_quick \
        --batch-size 64
    
    # Generate plots
    echo -e "${YELLOW}→ Creating visualizations...${NC}"
    python scripts/plot.py \
        --input results/gpu_quick/metrics_all.json \
        --output-dir outputs/gpu_quick
    
    echo -e "${GREEN}✓ GPU test complete!${NC}"
}

# Function to run tests
run_tests() {
    echo -e "\n${YELLOW}🧪 Running test suite...${NC}"
    
    # Unit tests
    if [ -d "tests" ]; then
        python -m pytest tests/ -v --tb=short
        echo -e "${GREEN}✓ Unit tests passed${NC}"
    else
        echo -e "${YELLOW}⚠ No tests directory found${NC}"
    fi
    
    # Format check
    echo -e "${YELLOW}→ Checking code format...${NC}"
    black src/ tests/ scripts/ --check --quiet || echo -e "${YELLOW}⚠ Some files need formatting${NC}"
    
    # Linting
    echo -e "${YELLOW}→ Running linter...${NC}"
    ruff check src/ --quiet || echo -e "${YELLOW}⚠ Some linting issues found${NC}"
    
    echo -e "${GREEN}✓ Tests complete!${NC}"
}

# Main execution
check_venv
install_deps

case $MODE in
    cpu)
        run_cpu_smoke
        ;;
    gpu)
        run_gpu_test
        ;;
    test)
        run_tests
        ;;
    all)
        run_cpu_smoke
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            run_gpu_test
        fi
        run_tests
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Usage: $0 [cpu|gpu|test|all]"
        exit 1
        ;;
esac

# Summary
echo -e "\n${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}✨ TIDE-Lite Quick Run Complete! ✨${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"

if [ "$MODE" != "test" ]; then
    echo -e "\n📊 Expected artifacts:"
    echo "  • results/${MODE}_*/config_used.json"
    echo "  • results/${MODE}_*/metrics_train.json"
    echo "  • outputs/${MODE}/fig_*.png"
    echo "  • outputs/REPORT.md"
fi

echo -e "\n${YELLOW}Next steps:${NC}"
echo "  1. Check outputs/ for generated plots"
echo "  2. Review outputs/REPORT.md for results"
echo "  3. Run full training: python -m src.tide_lite.cli.train_cli"
echo ""
