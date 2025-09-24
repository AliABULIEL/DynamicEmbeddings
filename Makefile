# TIDE-Lite Development Makefile
# Provides convenient shortcuts for common development tasks

SHELL := /bin/bash
PYTHON := python3
PIP := $(PYTHON) -m pip
PROJECT := tide_lite
TESTS := tests/

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "TIDE-Lite Development Commands"
	@echo "=============================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ============================================================================
# Environment Setup
# ============================================================================

.PHONY: install
install: ## Install package and dependencies
	$(PIP) install -e .
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev: install ## Install development dependencies
	$(PIP) install pytest pytest-cov black isort mypy flake8

.PHONY: clean
clean: ## Clean build artifacts and cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/

# ============================================================================
# Code Quality
# ============================================================================

.PHONY: format
format: ## Format code with black and isort
	black src/ tests/ --line-length 100
	isort src/ tests/ --profile black --line-length 100

.PHONY: lint
lint: ## Run linting checks with flake8
	flake8 src/ tests/ --max-line-length 100 --ignore E203,W503

.PHONY: typecheck
typecheck: ## Run type checking with mypy
	mypy src/$(PROJECT) --ignore-missing-imports

.PHONY: test
test: ## Run unit tests with pytest
	pytest $(TESTS) -v --tb=short

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	pytest $(TESTS) --cov=$(PROJECT) --cov-report=term-missing --cov-report=html

.PHONY: check
check: format lint typecheck test ## Run all checks (format, lint, typecheck, test)

# ============================================================================
# Training & Evaluation (Dry-Run Shortcuts)
# ============================================================================

.PHONY: train-dry
train-dry: ## Show training plan (dry-run)
	$(PYTHON) -m $(PROJECT).cli.tide train \
		--output-dir results/test-run \
		--num-epochs 3

.PHONY: bench-all-dry
bench-all-dry: ## Show benchmark plan for all evaluations (dry-run)
	$(PYTHON) -m $(PROJECT).cli.tide bench-all \
		--model minilm \
		--type baseline

.PHONY: ablation-dry
ablation-dry: ## Show ablation study plan (dry-run)
	$(PYTHON) -m $(PROJECT).cli.tide ablation \
		--time-mlp-hidden 64,128 \
		--consistency-weight 0.1,0.2 \
		--time-encoding sinusoidal,learnable

.PHONY: aggregate-dry
aggregate-dry: ## Show aggregation plan (dry-run)
	$(PYTHON) -m $(PROJECT).cli.tide aggregate \
		--results-dir results/

.PHONY: report-dry
report-dry: ## Show report generation plan (dry-run)
	$(PYTHON) -m $(PROJECT).cli.tide report \
		--input results/summary.json \
		--output-dir reports/

# ============================================================================
# Actual Execution (with --run flag)
# ============================================================================

.PHONY: train-run
train-run: ## Actually train model (execute)
	$(PYTHON) -m $(PROJECT).cli.tide train \
		--output-dir results/test-run \
		--num-epochs 3 \
		--run

.PHONY: bench-baseline
bench-baseline: ## Benchmark all baseline models
	@echo "Benchmarking MiniLM..."
	$(PYTHON) -m $(PROJECT).cli.tide bench-all --model minilm --type baseline --run
	@echo "Benchmarking E5-Base..."
	$(PYTHON) -m $(PROJECT).cli.tide bench-all --model e5-base --type baseline --run
	@echo "Benchmarking BGE-Base..."
	$(PYTHON) -m $(PROJECT).cli.tide bench-all --model bge-base --type baseline --run

# ============================================================================
# Documentation
# ============================================================================

.PHONY: docs
docs: ## Generate documentation
	@echo "Generating documentation..."
	$(PYTHON) -m pydoc -w src/$(PROJECT)

# ============================================================================
# Quick Demos
# ============================================================================

.PHONY: demo-cpu
demo-cpu: ## Run quick CPU demo (dry-run)
	@echo "========================================="
	@echo "TIDE-Lite CPU Demo (Dry-Run Mode)"
	@echo "========================================="
	$(PYTHON) -m $(PROJECT).cli.tide train --num-epochs 1 --batch-size 8
	$(PYTHON) -m $(PROJECT).cli.tide bench-all --model minilm --type baseline

.PHONY: demo-full
demo-full: install ## Run full demo pipeline (dry-run)
	@echo "========================================="
	@echo "TIDE-Lite Full Pipeline Demo (Dry-Run)"
	@echo "========================================="
	@echo ""
	@echo "1. Training plan..."
	$(PYTHON) -m $(PROJECT).cli.tide train --output-dir results/demo
	@echo ""
	@echo "2. Evaluation plan..."
	$(PYTHON) -m $(PROJECT).cli.tide bench-all --model results/demo/checkpoints/best_model.pt
	@echo ""
	@echo "3. Ablation plan..."
	$(PYTHON) -m $(PROJECT).cli.tide ablation --time-mlp-hidden 64,128
	@echo ""
	@echo "4. Report plan..."
	$(PYTHON) -m $(PROJECT).cli.tide aggregate --results-dir results/
	$(PYTHON) -m $(PROJECT).cli.tide report --input results/summary.json

# ============================================================================
# Development Shortcuts
# ============================================================================

.PHONY: watch
watch: ## Watch for changes and run tests
	@while true; do \
		clear; \
		make test; \
		inotifywait -qre modify src/ tests/ 2>/dev/null || sleep 2; \
	done

.PHONY: todo
todo: ## Show all TODO comments in code
	@grep -r "TODO\|FIXME\|XXX" src/ tests/ --exclude-dir=__pycache__ || echo "No TODOs found!"

.PHONY: stats
stats: ## Show code statistics
	@echo "Code Statistics:"
	@echo "================"
	@echo -n "Python files: "
	@find src/ tests/ -name "*.py" | wc -l
	@echo -n "Lines of code: "
	@find src/ tests/ -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $$1}'
	@echo -n "Test files: "
	@find tests/ -name "test_*.py" | wc -l
