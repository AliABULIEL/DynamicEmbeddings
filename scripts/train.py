"""
Main training script for Dynamic Embeddings
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
import datasets
from tqdm import tqdm

# Import our modules
from config.model_config import CompleteConfig, get_task_specific_config
from src.models.dynamic_model import DynamicEmbeddingModel
from src.training.trainer import DynamicEmbeddingTrainer
from src.training.dataset import create_training_dataset
from src.evaluation.mteb_evaluator import MTEBEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_training_data(config: CompleteConfig):
    """
    Load training data from various sources
    """
    logger.info("Loading training data...")

    all_texts = []
    all_labels = []
    all_domains = []

    # Load diverse datasets for training
    dataset_configs = [
        ('ag_news', 'train', 'text', 'label', 'news'),
        ('imdb', 'train', 'text', 'label', 'reviews'),
        ('banking77', 'train', 'text', 'label', 'financial'),
        ('squad', 'train', 'question', None, 'qa'),
    ]

    for dataset_name, split, text_col, label_col, domain in dataset_configs:
        try:
            logger.info(f"  Loading {dataset_name}...")

            # Load dataset
            if dataset_name == 'banking77':
                # Banking77 is in different format
                dataset = datasets.load_dataset('banking77', split=split)
            else:
                dataset = datasets.load_dataset(dataset_name, split=f'{split}[:5000]')

            # Extract texts and labels
            for item in dataset:
                text = item[text_col]
                label = item[label_col] if label_col else len(all_labels)

                all_texts.append(text)
                all_labels.append(label)
                all_domains.append(domain)

                if len(all_texts) >= 20000:  # Limit total size
                    break

            logger.info(f"    Loaded {len(dataset)} samples from {dataset_name}")

        except Exception as e:
            logger.warning(f"  Failed to load {dataset_name}: {e}")
            continue

    logger.info(f"Total training samples: {len(all_texts)}")
    logger.info(f"Domain distribution: {dict(zip(*np.unique(all_domains, return_counts=True)))}")

    return all_texts, all_labels, all_domains


def create_data_loaders(texts, labels, domains, config: CompleteConfig):
    """
    Create training and validation data loaders
    """
    from sklearn.model_selection import train_test_split
    from src.training.dataset import DynamicEmbeddingDataset

    # Split data
    train_texts, val_texts, train_labels, val_labels, train_domains, val_domains = \
        train_test_split(
            texts, labels, domains,
            test_size=config.data.validation_split,
            random_state=config.seed,
            stratify=domains
        )

    logger.info(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

    # Create datasets
    train_dataset = DynamicEmbeddingDataset(
        train_texts, train_labels, train_domains,
        augment=config.data.use_augmentation
    )

    val_dataset = DynamicEmbeddingDataset(
        val_texts, val_labels, val_domains,
        augment=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


def evaluate_on_mteb(model, config: CompleteConfig):
    """
    Evaluate model on MTEB benchmarks
    """
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on MTEB Benchmarks")
    logger.info("=" * 60)

    evaluator = MTEBEvaluator(model, config)

    # Select subset of tasks for quick evaluation
    tasks = ['Banking77Classification', 'MSMARCO', 'STSBenchmark']

    results = {}
    for task in tasks:
        logger.info(f"\nEvaluating on {task}...")
        try:
            task_results = evaluator.evaluate_task(task)
            results[task] = task_results
            logger.info(f"  Results: {task_results}")
        except Exception as e:
            logger.error(f"  Failed: {e}")

    return results


def main(args):
    """
    Main training function
    """
    # Load configuration
    if args.config:
        config = CompleteConfig.load(args.config)
    else:
        config = get_task_specific_config(args.task) if args.task else CompleteConfig()

    # Override config with command line arguments
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.finetune_epochs = args.epochs

    # Set seed
    set_seed(config.seed)

    # Create output directory
    output_dir = Path(config.data.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config.save(output_dir / "config.json")

    logger.info("=" * 60)
    logger.info("Dynamic Embeddings Training")
    logger.info("=" * 60)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {config.training.device}")

    # Initialize model
    logger.info("\nInitializing model...")
    model = DynamicEmbeddingModel(config)

    # Load training data
    texts, labels, domains = load_training_data(config)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        texts, labels, domains, config
    )

    # Initialize trainer
    trainer = DynamicEmbeddingTrainer(
        model, config,
        use_wandb=args.wandb
    )

    # Train model
    logger.info("\nStarting training...")
    trainer.train(
        train_loader, val_loader,
        epochs=args.epochs
    )

    # Save final model
    model.save(output_dir / "final_model.pt")

    # Evaluate on MTEB
    if args.evaluate:
        mteb_results = evaluate_on_mteb(model, config)

        # Save results
        import json
        with open(output_dir / "mteb_results.json", "w") as f:
            json.dump(mteb_results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dynamic Embeddings Model")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="dynamic_embeddings_exp",
        help="Experiment name"
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["retrieval", "classification", "similarity", "clustering"],
        help="Task-specific configuration"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate on MTEB after training"
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )

    args = parser.parse_args()
    main(args)