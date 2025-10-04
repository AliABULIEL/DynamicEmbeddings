"""Training utilities for LoRA adapters and full fine-tuning.

Supports multiple training modes:
- lora: Freeze backbone, train LoRA adapters (parameter-efficient)
- full_ft: Full fine-tuning per time bin (separate model per bin)
- seq_ft: Sequential fine-tuning across bins in chronological order
"""

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal

import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

from ..models.lora_model import create_lora_model, assert_trainable_ratio, save_lora_adapter
from ..utils.logging import get_logger
from ..utils.seeding import set_seed

logger = get_logger(__name__)

TrainingMode = Literal["lora", "full_ft", "seq_ft"]


def log_model_info(model: SentenceTransformer, mode: TrainingMode) -> None:
    """Log model configuration and trainable parameters.
    
    Args:
        model: SentenceTransformer model.
        mode: Training mode.
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0
    
    logger.info(f"\n{'='*80}")
    logger.info(f"MODEL CONFIGURATION")
    logger.info(f"{'='*80}")
    logger.info(f"Training mode: {mode.upper()}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable ratio: {trainable_ratio:.4%}")
    
    if mode == "lora":
        # Log LoRA target modules
        target_modules = []
        for name, module in model.named_modules():
            if hasattr(module, "lora_A"):
                target_modules.append(name)
        
        if target_modules:
            logger.info(f"LoRA target modules: {', '.join(target_modules[:3])}...")
            logger.info(f"Total LoRA modules: {len(target_modules)}")
        
        # Assert trainable ratio for LoRA
        if trainable_ratio >= 0.01:
            logger.warning(
                f"⚠ Trainable ratio ({trainable_ratio:.4%}) >= 1% for LoRA mode. "
                f"This is unusually high and may indicate LoRA is not properly attached."
            )
    
    logger.info(f"{'='*80}\n")


def check_cuda_oom_risk(
    batch_size: int,
    model_size_mb: float,
    mode: TrainingMode,
) -> None:
    """Check for potential CUDA OOM and suggest mitigations.
    
    Args:
        batch_size: Training batch size.
        model_size_mb: Approximate model size in MB.
        mode: Training mode.
    """
    if not torch.cuda.is_available():
        return
    
    # Get GPU memory
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Rough heuristic for OOM risk
    if mode == "full_ft" and batch_size > 16 and gpu_mem_gb < 8:
        logger.warning(
            f"\n⚠ OOM Risk: GPU memory={gpu_mem_gb:.1f}GB, batch_size={batch_size}, mode={mode}"
        )
        logger.warning("Suggestions to reduce memory:")
        logger.warning("  1. Reduce --batch-size (try 8 or 16)")
        logger.warning("  2. Use --mode lora instead of full_ft")
        logger.warning("  3. Disable --fp16 (uses more memory but may help)")
        logger.warning("")


class UnifiedTrainer:
    """Unified trainer supporting LoRA, full fine-tuning, and sequential fine-tuning."""
    
    def __init__(
        self,
        base_model_name: str,
        mode: TrainingMode = "lora",
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
            mode: Training mode (lora/full_ft/seq_ft).
            lora_r: LoRA rank (only for lora mode).
            lora_alpha: LoRA alpha (only for lora mode).
            lora_dropout: LoRA dropout (only for lora mode).
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
        self.mode = mode
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
        
        # Check OOM risk
        model_size_mb = 100  # Rough estimate for MiniLM
        check_cuda_oom_risk(self.batch_size, model_size_mb, self.mode)
    
    def create_model(self, existing_model: Optional[SentenceTransformer] = None) -> SentenceTransformer:
        """Create a new model based on training mode.
        
        Args:
            existing_model: For seq_ft mode, pass previous model to continue training.
        
        Returns:
            SentenceTransformer (with or without LoRA).
        """
        if self.mode == "lora":
            # Create model with LoRA adapters
            model, target_modules = create_lora_model(
                base_model_name=self.base_model_name,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
            )
            
            # Verify trainable ratio
            ratio = assert_trainable_ratio(model, max_ratio=0.01, raise_error=False)
            
            if ratio >= 0.01:
                logger.error(
                    f"✗ Trainable ratio ({ratio:.4%}) >= 1% for LoRA mode!\n"
                    f"LoRA should freeze the backbone and train <1% of parameters.\n"
                    f"Check that PEFT is correctly attached to target modules."
                )
                raise ValueError(f"LoRA trainable ratio too high: {ratio:.4%}")
            
        elif self.mode == "full_ft":
            # Full fine-tuning: unfreeze all parameters
            if existing_model is not None:
                model = existing_model
            else:
                model = SentenceTransformer(self.base_model_name)
            
            # Unfreeze all parameters
            for param in model.parameters():
                param.requires_grad = True
        
        elif self.mode == "seq_ft":
            # Sequential fine-tuning: continue from existing model or start fresh
            if existing_model is not None:
                logger.info(f"Continuing sequential training from existing model")
                model = existing_model
            else:
                logger.info(f"Starting sequential training from base model")
                model = SentenceTransformer(self.base_model_name)
            
            # Unfreeze all parameters
            for param in model.parameters():
                param.requires_grad = True
        
        else:
            raise ValueError(f"Unknown training mode: {self.mode}")
        
        # Log model info
        log_model_info(model, self.mode)
        
        return model
    
    def train_bucket(
        self,
        bucket_name: str,
        train_examples: List[InputExample],
        val_examples: Optional[List[InputExample]] = None,
        output_dir: Optional[Path] = None,
        existing_model: Optional[SentenceTransformer] = None,
    ) -> Dict[str, Any]:
        """Train model for a single time bucket.
        
        Args:
            bucket_name: Name of the time bucket.
            train_examples: Training examples.
            val_examples: Validation examples (optional).
            output_dir: Directory to save model/adapter and logs.
            existing_model: For seq_ft, continue from this model.
            
        Returns:
            Training metrics dictionary.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Training {self.mode.upper()} for bucket: {bucket_name}")
        logger.info(f"{'='*80}")
        
        # Create model
        model = self.create_model(existing_model=existing_model)
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
        
        logger.info(f"Training examples: {len(train_examples)}")
        logger.info(f"Training steps: {num_train_steps} (warmup: {warmup_steps})")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Epochs: {self.epochs}")
        
        # Setup output directory
        if output_dir is None:
            if self.mode == "lora":
                output_dir = Path(f"models/adapters/{bucket_name}")
            elif self.mode == "full_ft":
                output_dir = Path(f"artifacts/full_ft/{bucket_name}")
            elif self.mode == "seq_ft":
                output_dir = Path(f"artifacts/seq_ft/step_{bucket_name}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV logger
        log_path = output_dir / "training_log.csv"
        csv_file = open(log_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "step", "loss", "time"])
        
        # Training loop
        start_time = time.time()
        
        try:
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=self.epochs,
                warmup_steps=warmup_steps,
                optimizer_params={
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                },
                use_amp=self.fp16,
                show_progress_bar=True,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    f"\n✗ CUDA Out of Memory!\n"
                    f"Suggestions:\n"
                    f"  1. Reduce --batch-size (current: {self.batch_size})\n"
                    f"  2. Use --mode lora instead of {self.mode}\n"
                    f"  3. Reduce --lora-r (current: {self.lora_r})\n"
                )
            raise
        
        total_time = time.time() - start_time
        
        # Save model
        if self.mode == "lora":
            save_lora_adapter(model, output_dir)
            logger.info(f"✓ LoRA adapter saved to: {output_dir}")
        else:
            # Save full model checkpoint
            model.save(str(output_dir))
            logger.info(f"✓ Full model saved to: {output_dir}")
        
        # Close CSV logger
        csv_file.close()
        
        metrics = {
            "bucket": bucket_name,
            "mode": self.mode,
            "epochs": self.epochs,
            "train_examples": len(train_examples),
            "total_time": total_time,
            "output_dir": str(output_dir),
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Training complete for bucket: {bucket_name}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Output: {output_dir}")
        logger.info(f"{'='*80}\n")
        
        return metrics, model


def train_all_buckets(
    data_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
    mode: TrainingMode = "lora",
    use_hard_negatives: bool = False,
    neg_k: int = 4,
) -> Dict[str, Dict[str, Any]]:
    """Train models for all time buckets.
    
    Args:
        data_dir: Directory containing bucket subdirectories with parquet files.
        output_dir: Base directory for saving models/adapters.
        config: Training configuration.
        mode: Training mode (lora/full_ft/seq_ft).
        use_hard_negatives: Whether to use hard temporal negatives.
        neg_k: Number of hard negatives per positive.
        
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
    trainer = UnifiedTrainer(
        base_model_name=base_model,
        mode=mode,
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
    
    # Get all buckets in sorted order (chronological for seq_ft)
    bucket_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(bucket_dirs)} buckets to train")
    
    # Prepare for hard negatives if needed
    all_bins = None
    if use_hard_negatives:
        from .hard_negatives import add_hard_temporal_negatives
        
        all_bins = [d.name for d in bucket_dirs]
        logger.info(f"✓ Hard temporal negatives enabled (neg_k={neg_k})")
        logger.info(f"Bins (chronological): {all_bins}")
    
    # Train each bucket
    all_metrics = {}
    current_model = None  # For seq_ft mode
    
    for bucket_dir in bucket_dirs:
        bucket_name = bucket_dir.name
        
        # Load data
        train_examples, val_examples, _ = load_bucket_data(bucket_dir)
        
        if not train_examples:
            logger.warning(f"No training data found for bucket: {bucket_name}, skipping")
            continue
        
        # Augment with hard negatives if enabled
        if use_hard_negatives and all_bins:
            train_examples = add_hard_temporal_negatives(
                data_dir=data_dir,
                all_bins=all_bins,
                bucket_name=bucket_name,
                train_examples=train_examples,
                neg_k=neg_k,
                seed=train_config.get("seed", 42),
            )
        
        # Train
        bucket_output_dir = output_dir / bucket_name
        
        metrics, trained_model = trainer.train_bucket(
            bucket_name=bucket_name,
            train_examples=train_examples,
            val_examples=val_examples if val_examples else None,
            output_dir=bucket_output_dir,
            existing_model=current_model if mode == "seq_ft" else None,
        )
        
        all_metrics[bucket_name] = metrics
        
        # For seq_ft, keep the model for next bucket
        if mode == "seq_ft":
            current_model = trained_model
    
    return all_metrics
