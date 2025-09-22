#!/bin/bash

# Code formatting script using black and isort

echo "🎨 Formatting code with black and isort..."

# Define directories to format
DIRS="src tests scripts"

# Run isort first (import sorting)
echo "→ Sorting imports with isort..."
isort $DIRS --quiet

# Run black (code formatting)
echo "→ Formatting with black..."
black $DIRS --quiet

# Exclude outputs and data directories
find outputs data -name "*.py" 2>/dev/null | xargs -r rm 2>/dev/null || true

echo "✓ Code formatting complete!"

# Check if any files were changed
if git diff --exit-code > /dev/null; then
    echo "✅ No formatting changes needed."
else
    echo "⚠️  Some files were formatted. Review changes with: git diff"
    echo "   Run 'git add -p' to selectively stage changes."
fi
