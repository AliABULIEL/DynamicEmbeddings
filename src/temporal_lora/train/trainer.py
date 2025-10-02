"""Training utilities for LoRA adapters."""

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

from ..models.lora_model import create_lora_model, assert_trainable_ratio, save_lora_adapter
from ..utils.logging import get_logger
from ..utils.seeding import set_seed

logger = get_logger(__name__)


class LoRATrainer:
    """Trainer for LoRA adapters on sentence transformers."""
    
    def __init__(
        self,
        base_model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        epochs: int = 2,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        fp16: bool = True,
        seed: int = 42,
    ):
        """Initialize trainer.
        
        Args:
            base_model_name: HuggingFace model identifier.
            lora_r: LoRA rank.
            lora_alpha: LoRA alpha.
            lora_dropout: LoRA dropout.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            warmup_ratio: Warmup ratio for scheduler.
            weight_decay: Weight decay.
            max_grad_norm: Gradient clipping value.
            fp16: Use mixed precision training if CUDA available.
            seed: Random seed.
        """
        self.base_model_name = base_model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16 and torch.cuda.is_available()
        self.seed = seed
        
        set_seed(seed)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        if self.fp16:
            logger.info("✓ Mixed precision (fp16) enabled")
    
    def create_model(self) -> SentenceTransformer:
        """Create a new model with LoRA adapters.
        
        Returns:
            SentenceTransformer with LoRA.
        """
        model, target_modules = create_lora_model(
            base_model_name=self.base_model_name,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )
        
        # Verify trainable ratio
        ratio = assert_trainable_ratio(model, max_ratio=0.01, raise_error=True)
        
        logger.info(f"✓ Model created with target modules: {target_modules}")
        logger.info(f"✓ Trainable parameter ratio: {ratio:.4%}")
        
        return model
    
    def train_bucket(
        self,
        bucket_name: str,
        train_examples: List[InputExample],
        val_examples: Optional[List[InputExample]] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Train LoRA adapter for a single time bucket.
        
        Args:
            bucket_name: Name of the time bucket.
            train_examples: Training examples.
            val_examples: Validation examples (optional).
            output_dir: Directory to save adapter and logs.
            
        Returns:
            Training metrics dictionary.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Training LoRA adapter for bucket: {bucket_name}")
        logger.info(f"{'='*80}")
        
        # Create model
        model = self.create_model()
        model.to(self.device)
        
        # Create data loader
        train_dataloader = DataLoader(
            train_examples,
            batch_size=self.batch_size,
            shuffle=True,
        )
        
        # Create loss function
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        # Calculate training steps
        num_train_steps = len(train_dataloader) * self.epochs
        warmup_steps = int(num_train_steps * self.warmup_ratio)
        
        logger.info(f"Training steps: {num_train_steps} (warmup: {warmup_steps})")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path(f"models/adapters/{bucket_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV logger
        log_path = output_dir / "training_log.csv"
        csv_file = open(log_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "step", "loss", "time"])
        
        # Training loop
        start_time = time.time()
        global_step = 0
        
        # Configure optimizer
        optimizer_params = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
        
        # Use sentence-transformers fit method
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=warmup_steps,
            optimizer_params=optimizer_params,
            use_amp=self.fp16,
            show_progress_bar=True,
        )
        
        total_time = time.time() - start_time
        
        # Save final adapter
        save_lora_adapter(model, output_dir)
        
        # Close CSV logger
        csv_file.close()
        
        metrics = {
            "bucket": bucket_name,
            "epochs": self.epochs,
            "train_examples": len(train_examples),
            "total_time": total_time,
            "output_dir": str(output_dir),
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Training complete for bucket: {bucket_name}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Adapter saved to: {output_dir}")
        logger.info(f"{'='*80}\n")
        
        return metrics


def train_all_buckets(
    data_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Train LoRA adapters for all time buckets.
    
    Args:
        data_dir: Directory containing bucket subdirectories with parquet files.
        output_dir: Base directory for saving adapters.
        config: Training configuration.
        
    Returns:
        Dictionary mapping bucket names to training metrics.
    """
    from .data_loader import load_bucket_data
    
    # Extract config
    model_config = config.get("model", {})
    train_config = config.get("training", {})
    
    base_model = model_config.get("base_model", {}).get(
        "name", "sentence-transformers/all-MiniLM-L6-v2"
    )
    lora_config = model_config.get("lora", {})
    
    # Initialize trainer
    trainer = LoRATrainer(
        base_model_name=base_model,
        lora_r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        epochs=train_config.get("epochs", 2),
        batch_size=train_config.get("batch_size", 32),
        learning_rate=train_config.get("learning_rate", 2e-5),
        warmup_ratio=train_config.get("warmup_ratio", 0.1),
        weight_decay=train_config.get("weight_decay", 0.01),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        fp16=train_config.get("fp16", True),
        seed=train_config.get("seed", 42),
    )
    
    # Train each bucket
    all_metrics = {}
    
    bucket_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(bucket_dirs)} buckets to train")
    
    for bucket_dir in bucket_dirs:
        bucket_name = bucket_dir.name
        
        # Load data
        train_examples, val_examples, _ = load_bucket_data(bucket_dir)
        
        if not train_examples:
            logger.warning(f"No training data found for bucket: {bucket_name}, skipping")
            continue
        
        # Train
        bucket_output_dir = output_dir / bucket_name
        metrics = trainer.train_bucket(
            bucket_name=bucket_name,
            train_examples=train_examples,
            val_examples=val_examples if val_examples else None,
            output_dir=bucket_output_dir,
        )
        
        all_metrics[bucket_name] = metrics
    
    return all_metrics
