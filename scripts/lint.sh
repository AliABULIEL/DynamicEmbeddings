#!/bin/bash

# Linting script using ruff

echo "🔍 Linting code with ruff..."

# Run ruff on source code
echo "Checking src/..."
ruff check src/ --fix --show-fixes

# Run ruff on tests
echo "Checking tests/..."
ruff check tests/ --fix --show-fixes

# Run ruff on scripts
echo "Checking scripts/..."
ruff check scripts/*.py --fix --show-fixes

# Type checking with mypy (if available)
if command -v mypy &> /dev/null; then
    echo ""
    echo "🔎 Type checking with mypy..."
    mypy src/ --ignore-missing-imports --no-error-summary
else
    echo "ℹ️  mypy not installed, skipping type checks"
fi

echo ""
echo "✓ Linting complete!"

# Summary
if ruff check src/ tests/ scripts/*.py --quiet; then
    echo "✨ No linting issues found!"
else
    echo "⚠️  Some issues remain. Run 'ruff check src/' for details."
fi
