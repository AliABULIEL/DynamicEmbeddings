# TIDE-Lite Development Makefile

.PHONY: help install format lint typecheck test clean
.PHONY: train-dry bench-dry aggregate-dry report-dry

# Default target
help:
	@echo "TIDE-Lite Development Commands"
	@echo ""
	@echo "Setup & Development:"
	@echo "  make install      Install dependencies"
	@echo "  make format       Format code with black"
	@echo "  make lint         Lint code with flake8"
	@echo "  make typecheck    Type check with mypy"
	@echo "  make test         Run unit tests"
	@echo "  make clean        Clean generated files"
	@echo ""
	@echo "Dry-Run Commands (show plan only):"
	@echo "  make train-dry    Show training plan"
	@echo "  make bench-dry    Show benchmark plan"
	@echo "  make aggregate-dry Show aggregation plan"
	@echo "  make report-dry   Show report generation plan"

# Development commands
install:
	pip install -r requirements.txt
	pip install -e .

format:
	black src/ tests/ --line-length 100
	isort src/ tests/ --profile black

lint:
	flake8 src/ tests/ --max-line-length 100 --ignore E203,W503

typecheck:
	mypy src/tide_lite --ignore-missing-imports

test:
	pytest tests/ -v --cov=src/tide_lite --cov-report=term-missing

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Dry-run shortcuts
train-dry:
	python -m tide_lite.cli.tide train \
		--output-dir results/test-run \
		--num-epochs 1 \
		--batch-size 32

bench-dry:
	python -m tide_lite.cli.tide bench-all \
		--model minilm \
		--type baseline \
		--output-dir results/benchmark

aggregate-dry:
	python -m tide_lite.cli.tide aggregate \
		--results-dir results/ \
		--output results/summary.json

report-dry:
	python -m tide_lite.cli.tide report \
		--input results/summary.json \
		--output-dir reports/

# Full pipeline (dry-run)
pipeline-dry: train-dry bench-dry aggregate-dry report-dry
	@echo "Full pipeline plan complete (dry-run)"

# Quick test
smoke-test:
	python -m tide_lite.cli.tide bench-all \
		--model minilm \
		--type baseline \
		--max-corpus 100 \
		--max-queries 10
