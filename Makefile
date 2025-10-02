.PHONY: help install format lint test data train index eval viz export ablate env clean

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies and pre-commit hooks
	pip install -r requirements.txt
	pre-commit install

format:  ## Format code with black
	black src/ tests/ --config pyproject.toml
	ruff check src/ tests/ --fix

lint:  ## Lint code with ruff
	ruff check src/ tests/
	black src/ tests/ --check --config pyproject.toml

test:  ## Run pytest with coverage
	pytest tests/ -v --cov=src/temporal_lora --cov-report=term-missing

test-fast:  ## Run pytest without coverage
	pytest tests/ -q

env:  ## Dump environment info to deliverables/repro/
	python -m temporal_lora.cli env-dump

data:  ## Prepare dataset with default settings
	python -m temporal_lora.cli prepare-data --max_per_bucket 6000

train:  ## Train temporal LoRA adapters
	python -m temporal_lora.cli train-adapters --epochs 2 --lora_r 16 --cross_period_negatives true

index:  ## Build FAISS indexes for all time buckets
	python -m temporal_lora.cli build-indexes

eval:  ## Run evaluation across all scenarios
	python -m temporal_lora.cli evaluate --scenarios within cross all --mode multi-index --merge softmax

viz:  ## Generate all visualizations
	python -m temporal_lora.cli visualize

export:  ## Export deliverables (results, figures, repro info)
	python -m temporal_lora.cli export-deliverables

ablate:  ## Run ablation studies
	python -m temporal_lora.cli ablate

clean:  ## Clean generated files and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage

pipeline:  ## Run full pipeline (data -> train -> eval -> viz)
	$(MAKE) data
	$(MAKE) train
	$(MAKE) index
	$(MAKE) eval
	$(MAKE) viz
	$(MAKE) export
