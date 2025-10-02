.PHONY: install format lint test data train index eval viz export clean help

help:
	@echo "Temporal LoRA - Makefile Targets"
	@echo "================================="
	@echo "install    - Install dependencies"
	@echo "format     - Format code with black"
	@echo "lint       - Lint code with ruff"
	@echo "test       - Run pytest"
	@echo "data       - Prepare dataset"
	@echo "train      - Train LoRA adapters"
	@echo "index      - Build FAISS indexes"
	@echo "eval       - Run evaluation"
	@echo "viz        - Generate visualizations"
	@echo "export     - Export deliverables"
	@echo "clean      - Remove generated files"

install:
	pip install -r requirements.txt
	pre-commit install

format:
	black src/ tests/
	ruff check --fix src/ tests/

lint:
	ruff check src/ tests/
	black --check src/ tests/

test:
	pytest -v tests/

data:
	python -m temporal_lora.cli prepare-data --max_per_bucket 6000

train:
	python -m temporal_lora.cli train-adapters --epochs 2 --lora_r 16

index:
	python -m temporal_lora.cli build-indexes

eval:
	python -m temporal_lora.cli evaluate --scenarios within cross all --mode multi-index

viz:
	python -m temporal_lora.cli visualize

export:
	python -m temporal_lora.cli export-deliverables

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/
