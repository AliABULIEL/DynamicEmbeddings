#!/bin/bash
# Temporal LoRA - Quick Start Script
set -e

echo "======================================"
echo "Temporal LoRA - Quick Start"
echo "======================================"

# Check Python version
echo "Checking Python version..."
python --version

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Dump environment
echo "Capturing environment..."
python -m temporal_lora.cli env-dump

# Prepare data
echo "Preparing data (max 6000 per bucket)..."
python -m temporal_lora.cli prepare-data --max_per_bucket 6000

# Train adapters
echo "Training LoRA adapters (2 epochs, rank 16)..."
python -m temporal_lora.cli train-adapters --epochs 2 --lora_r 16 --cross_period_negatives true

# Build indexes
echo "Building FAISS indexes..."
python -m temporal_lora.cli build-indexes

# Evaluate
echo "Running evaluation..."
python -m temporal_lora.cli evaluate --scenarios within cross all --mode multi-index --merge softmax

# Visualize
echo "Generating visualizations..."
python -m temporal_lora.cli visualize

# Export deliverables
echo "Exporting deliverables..."
python -m temporal_lora.cli export-deliverables

echo "======================================"
echo "Quick start complete!"
echo "Results available in: deliverables/"
echo "======================================"
