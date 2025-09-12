#!/bin/bash
# setup.sh - Run this to set up the project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Download models (optional - they'll download on first use anyway)
python scripts/download_models.py

# Run a quick test
python main.py --experiments classification --datasets ag_news --composition-method weighted_sum