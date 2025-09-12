"""
Pre-download all required models to avoid downloading during experiments
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from config.model_configs import DOMAIN_MODELS, ZERO_SHOT_MODELS, BASELINE_MODELS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_domain_models():
    """Download all domain-specific models"""
    logger.info("Downloading domain-specific models...")

    for domain, config in DOMAIN_MODELS.items():
        logger.info(f"Downloading {domain} model: {config['model_name']}")

        try:
            if config['type'] == 'sentence-transformer':
                model = SentenceTransformer(config['model_name'])
                logger.info(f"✓ Successfully downloaded {domain} model")
            else:
                tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
                model = AutoModel.from_pretrained(config['model_name'])
                logger.info(f"✓ Successfully downloaded {domain} model")
        except Exception as e:
            logger.error(f"✗ Failed to download {domain} model: {e}")


def download_zero_shot_models():
    """Download zero-shot classification models"""
    logger.info("Downloading zero-shot classification models...")

    for name, model_name in ZERO_SHOT_MODELS.items():
        logger.info(f"Downloading {name} zero-shot model: {model_name}")

        try:
            from transformers import pipeline
            classifier = pipeline("zero-shot-classification", model=model_name)
            logger.info(f"✓ Successfully downloaded {name} zero-shot model")
        except Exception as e:
            logger.error(f"✗ Failed to download {name} model: {e}")


def download_baseline_models():
    """Download baseline models"""
    logger.info("Downloading baseline models...")

    for name, model_name in BASELINE_MODELS.items():
        logger.info(f"Downloading baseline {name}: {model_name}")

        try:
            model = SentenceTransformer(model_name)
            logger.info(f"✓ Successfully downloaded {name}")
        except Exception as e:
            logger.error(f"✗ Failed to download {name}: {e}")


def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA version: {torch.version.cuda}")
    else:
        logger.warning("✗ No GPU available, will use CPU (slower)")


def main():
    """Main function to download all models"""
    logger.info("=" * 60)
    logger.info("DOWNLOADING ALL REQUIRED MODELS")
    logger.info("=" * 60)

    # Check GPU availability
    check_gpu()

    # Download all models
    logger.info("\n" + "-" * 40)
    download_zero_shot_models()

    logger.info("\n" + "-" * 40)
    download_domain_models()

    logger.info("\n" + "-" * 40)
    download_baseline_models()

    logger.info("\n" + "=" * 60)
    logger.info("✓ All models downloaded successfully!")
    logger.info("You can now run experiments without waiting for downloads")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()