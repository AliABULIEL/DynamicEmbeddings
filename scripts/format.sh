#!/bin/bash

# Code formatting script using black

echo "🎨 Formatting code with black..."

# Format source code
black src/ --line-length 100 --target-version py38

# Format tests
black tests/ --line-length 100 --target-version py38

# Format scripts
black scripts/*.py --line-length 100 --target-version py38

# Format notebooks if they exist
if [ -d "notebooks" ]; then
    black notebooks/*.py --line-length 100 --target-version py38 2>/dev/null || true
fi

echo "✓ Code formatting complete!"

# Check if any files were changed
if git diff --exit-code > /dev/null; then
    echo "No formatting changes needed."
else
    echo "Some files were formatted. Review changes with: git diff"
fi
