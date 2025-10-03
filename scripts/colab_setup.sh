#!/bin/bash
# Colab Setup Script for Temporal LoRA Project
# This ensures clean package installation with exact versions

echo "🔄 Removing all conflicting packages..."
pip uninstall -y sentence-transformers transformers torch accelerate peft numpy typer click -q

# Install exact working versions in correct order
echo ""
echo "📦 Installing torch first..."
pip install torch==2.2.1 --no-cache-dir -q

echo "📦 Installing numpy..."
pip install "numpy>=1.26.0,<2.0.0" --no-cache-dir -q

echo "📦 Installing transformers..."
pip install transformers==4.40.0 --no-cache-dir -q

echo "📦 Installing sentence-transformers..."
pip install sentence-transformers==3.0.1 --no-cache-dir -q

echo "📦 Installing PEFT libraries..."
pip install accelerate==0.29.0 peft==0.10.0 --no-cache-dir -q

echo "📦 Installing CLI tools..."
pip install "typer[all]==0.9.0" "click>=8.0.0,<8.2.0" --no-cache-dir -q

echo "📦 Installing other dependencies..."
pip install datasets faiss-cpu pyyaml umap-learn scikit-learn matplotlib seaborn pandas --no-cache-dir -q

echo "📦 Installing project..."
pip install -e . --no-cache-dir -q

echo ""
echo "============================================================"
echo "✅ Installation complete!"
echo "============================================================"
echo ""
echo "⚠️  IMPORTANT: You MUST restart runtime now!"
echo "   Go to: Runtime → Restart runtime"
echo "============================================================"
