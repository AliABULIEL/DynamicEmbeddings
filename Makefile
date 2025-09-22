# TIDE-Lite Makefile
# Provides targets for formatting, linting, type checking, testing, and dry-run commands

.PHONY: help format lint typecheck test clean install dev-install dry-run-train dry-run-eval-stsb dry-run-eval-quora dry-run-eval-temporal all-checks

# Python executable
PYTHON := python3
PIP := $(PYTHON) -m pip

# Directories
SRC_DIR := src
TEST_DIR := tests
SCRIPTS_DIR := scripts

# Default target: show help
help:
	@echo "TIDE-Lite Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install          - Install package in production mode"
	@echo "  dev-install      - Install package in development mode with all dependencies"
	@echo "  format           - Format code with black and isort"
	@echo "  lint             - Run flake8 linting"
	@echo "  typecheck        - Run mypy type checking"
	@echo "  test             - Run unit tests with pytest"
	@echo "  all-checks       - Run format, lint, typecheck, and test"
	@echo "  clean            - Remove build artifacts and cache files"
	@echo ""
	@echo "Dry-run commands (show what would be executed):"
	@echo "  dry-run-train    - Show train command without execution"
	@echo "  dry-run-eval-stsb    - Show STS-B evaluation command"
	@echo "  dry-run-eval-quora   - Show Quora evaluation command"
	@echo "  dry-run-eval-temporal - Show temporal evaluation command"
	@echo ""

# Installation targets
install:
	$(PIP) install -e .

dev-install:
	$(PIP) install -e ".[dev]"
	$(PIP) install black isort flake8 mypy pytest pytest-cov

# Format code
format:
	@echo "Formatting code with black..."
	black $(SRC_DIR) $(TEST_DIR) --line-length 100 --target-version py38
	@echo "Organizing imports with isort..."
	isort $(SRC_DIR) $(TEST_DIR) --profile black --line-length 100

# Lint code
lint:
	@echo "Running flake8 linting..."
	flake8 $(SRC_DIR) $(TEST_DIR) \
		--max-line-length=100 \
		--extend-ignore=E203,W503 \
		--exclude=__pycache__,.git,build,dist

# Type checking
typecheck:
	@echo "Running mypy type checking..."
	mypy $(SRC_DIR) --config-file mypy.ini

# Run tests
test:
	@echo "Running unit tests with pytest..."
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

# Run tests with minimal output
test-quiet:
	pytest $(TEST_DIR) -q --tb=short

# Run specific test file
test-file:
	@echo "Usage: make test-file FILE=tests/test_utils.py"
	pytest $(FILE) -v

# Run all quality checks
all-checks: format lint typecheck test
	@echo "All checks completed successfully!"

# Clean build artifacts and cache
clean:
	@echo "Cleaning build artifacts and cache..."
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete

# Dry-run commands (show what would be executed without running)
dry-run-train:
	@echo "Dry-run: Training TIDE-Lite model"
	@echo "Command that would be executed:"
	@echo "python -m src.tide_lite.cli.train_cli \\"
	@echo "    --config configs/defaults.yaml \\"
	@echo "    --data-dir data/processed \\"
	@echo "    --output-dir results/experiment_$(shell date +%Y%m%d_%H%M%S) \\"
	@echo "    --batch-size 32 \\"
	@echo "    --lr 1e-4 \\"
	@echo "    --epochs 10 \\"
	@echo "    --encoder-name sentence-transformers/all-MiniLM-L6-v2 \\"
	@echo "    --freeze-encoder"

dry-run-eval-stsb:
	@echo "Dry-run: Evaluating on STS-B dataset"
	@echo "Command that would be executed:"
	@echo "python -m src.tide_lite.cli.eval_stsb_cli \\"
	@echo "    --checkpoint results/latest/model_best.pt \\"
	@echo "    --data-dir data/stsb \\"
	@echo "    --batch-size 64 \\"
	@echo "    --output-file results/stsb_eval.json"

dry-run-eval-quora:
	@echo "Dry-run: Evaluating on Quora dataset"
	@echo "Command that would be executed:"
	@echo "python -m src.tide_lite.cli.eval_quora_cli \\"
	@echo "    --checkpoint results/latest/model_best.pt \\"
	@echo "    --data-dir data/quora \\"
	@echo "    --batch-size 64 \\"
	@echo "    --output-file results/quora_eval.json"

dry-run-eval-temporal:
	@echo "Dry-run: Evaluating temporal performance"
	@echo "Command that would be executed:"
	@echo "python -m src.tide_lite.cli.eval_temporal_cli \\"
	@echo "    --checkpoint results/latest/model_best.pt \\"
	@echo "    --data-dir data/temporal \\"
	@echo "    --temporal-split week \\"
	@echo "    --batch-size 64 \\"
	@echo "    --output-file results/temporal_eval.json"

# Development workflow shortcuts
check: format lint typecheck
	@echo "Quick checks completed (format, lint, typecheck)"

ci: all-checks
	@echo "CI checks completed successfully"

# Watch for changes and run tests (requires pytest-watch)
watch:
	@command -v ptw >/dev/null 2>&1 || (echo "Installing pytest-watch..." && $(PIP) install pytest-watch)
	ptw $(TEST_DIR) -- -v --tb=short

# Generate coverage report
coverage:
	pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# Run security checks (requires bandit)
security:
	@command -v bandit >/dev/null 2>&1 || (echo "Installing bandit..." && $(PIP) install bandit)
	bandit -r $(SRC_DIR) -ll

# Create a source distribution
dist:
	$(PYTHON) setup.py sdist bdist_wheel

# Upload to PyPI (requires twine)
upload: dist
	@command -v twine >/dev/null 2>&1 || (echo "Installing twine..." && $(PIP) install twine)
	twine upload dist/*

# Development server for documentation (if using mkdocs)
docs-serve:
	@command -v mkdocs >/dev/null 2>&1 || (echo "Installing mkdocs..." && $(PIP) install mkdocs mkdocs-material)
	mkdocs serve

# Build documentation
docs-build:
	@command -v mkdocs >/dev/null 2>&1 || (echo "Installing mkdocs..." && $(PIP) install mkdocs mkdocs-material)
	mkdocs build
