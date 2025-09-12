"""
Central configuration file for all project settings
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Domain configuration
DOMAINS = ['scientific', 'news', 'medical', 'legal', 'social']

# Model settings
EMBEDDING_DIM = 768
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 32

# Evaluation settings
EVALUATION_DATASETS = ['ag_news', 'dbpedia_14', 'stsb']
RANDOM_SEED = 42
TEST_SAMPLE_SIZE = 1000  # For quick testing, use None for full dataset

# Composition methods
COMPOSITION_METHODS = ['weighted_sum', 'attention', 'max_pooling', 'learned_gate']

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"