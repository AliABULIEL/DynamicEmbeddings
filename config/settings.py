"""
Simplified settings
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

for dir_path in [RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Simplified domain list (not really used anymore)
DOMAINS = ['news']  # Only MPNet matters

# Model settings
EMBEDDING_DIM = 768
BATCH_SIZE = 32
TEST_SAMPLE_SIZE = 1000  # Use smaller sample for speed

# Remove all the complex settings we don't need