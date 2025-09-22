"""Training orchestration for TIDE-Lite models.

This module provides the main training loop with support for
mixed precision, checkpointing, and comprehensive logging.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.collate import STSBCollator, TextBatcher
from ..data.datasets import DatasetConfig, load_stsb_with_timestamps
from ..models.tide_lite import TIDELite, TIDELiteConfig
from ..utils.config import set_global_seed, setup_logging
from .losses import combined_tide_loss, cosine_regression_loss, temporal_consistency_loss

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for TIDE-Lite training.
    
    Attributes:
        # Model
        encoder_name: Base encoder model name.
        hidden_dim: Hidden dimension of encoder.
        time_encoding_dim: Dimension for temporal encoding.
        mlp_hidden_dim: Hidden dimension of temporal MLP.
        mlp_dropout: Dropout in temporal MLP.
        freeze_encoder: Whether to freeze encoder weights.
        
        # Data
        batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        max_seq_length: Maximum sequence length.
        num_workers: DataLoader workers.
        
        # Training
        num_epochs: Number of training epochs.
        learning_rate: Peak learning rate.
        warmup_steps: Linear warmup steps.
        weight_decay: AdamW weight decay.
        gradient_clip: Max gradient norm.
        
        # Loss weights
        temporal_weight: Weight for temporal consistency loss.
        preservation_weight: Weight for base embedding preservation.
        tau_seconds: Time constant for temporal loss.
        
        # Mixed precision
        use_amp: Whether to use automatic mixed precision.
        
        # Checkpointing
        save_every_n_steps: Checkpoint frequency.
        eval_every_n_steps: Evaluation frequency.
        
        # Paths
        output_dir: Directory for outputs.
        checkpoint_dir: Directory for checkpoints.
        
        # Misc
        seed: Random seed.
        log_level: Logging verbosity.
        dry_run: Whether to perform dry run only.
    """
    # Model
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    hidden_dim: int = 384
    time_encoding_dim: int = 32
    mlp_hidden_dim: int = 128
    mlp_dropout: float = 0.1
    freeze_encoder: bool = True
    pooling_strategy: str = "mean"  # "mean", "cls", or "max"
    gate_activation: str = "sigmoid"  # "sigmoid" or "tanh"
    
    # Data
    batch_size: int = 32
    eval_batch_size: int = 64
    max_seq_length: int = 128
    num_workers: int = 2
    
    # Training
    num_epochs: int = 3
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Loss weights
    temporal_weight: float = 0.1
    preservation_weight: float = 0.05
    tau_seconds: float = 86400.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    save_every_n_steps: int = 500
    eval_every_n_steps: int = 500
    
    # Paths
    output_dir: str = "results/default_run"
    checkpoint_dir: Optional[str] = None  # Uses output_dir/checkpoints if None
    
    # Misc
    seed: int = 42
    log_level: str = "INFO"
    dry_run: bool = False
    device: Optional[str] = None  # "cpu", "cuda", or None (auto-detect)
    temporal_enabled: bool = True  # Enable temporal module
    
    def __post_init__(self) -> None:
        """Post-initialization setup."""
        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(Path(self.output_dir) / "checkpoints")


class TIDETrainer:
    """Trainer for TIDE-Lite models.
    
    Handles the complete training pipeline including data loading,
    optimization, checkpointing, and metric logging.
    """
    
    def __init__(
        self,
        model: TIDELite,
        config: TrainingConfig,
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: TIDE-Lite model to train.
            config: Training configuration.
        """
        self.model = model
        self.config = config
        
        # Setup paths
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / "training.log"
        setup_logging(config.log_level, log_file)
        
        # Set seed for reproducibility
        set_global_seed(config.seed)
        
        # Move model to device
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Initialize components (will be setup in prepare_training)
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        
        # Metrics tracking
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "val_spearman": [],
            "learning_rate": [],
            "epoch_times": [],
        }
        self.global_step = 0
        
        logger.info(f"Initialized trainer with device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation dataloaders.
        
        Returns:
            Tuple of (train_loader, val_loader).
        """
        logger.info("Loading STS-B dataset with temporal augmentation")
        
        # Load dataset
        dataset_config = DatasetConfig(
            seed=self.config.seed,
            cache_dir="./data",
            timestamp_start="2020-01-01",
            timestamp_end="2024-01-01",
            temporal_noise_std=7.0,
        )
        
        datasets = load_stsb_with_timestamps(dataset_config)
        
        # Create tokenizer and collator
        tokenizer = TextBatcher(
            model_name=self.config.encoder_name,
            max_length=self.config.max_seq_length,
        )
        
        collator = STSBCollator(tokenizer, include_timestamps=True)
        
        # Create dataloaders
        train_loader = DataLoader(
            datasets["train"],
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            datasets["validation"],
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        
        logger.info(
            f"Prepared dataloaders - Train: {len(train_loader)} batches, "
            f"Val: {len(val_loader)} batches"
        )
        
        return train_loader, val_loader
    
    def prepare_optimizer(self) -> Tuple[torch.optim.Optimizer, Any]:
        """Prepare optimizer and learning rate scheduler.
        
        Returns:
            Tuple of (optimizer, scheduler).
        """
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        logger.info(
            f"Optimizing {len(trainable_params)} parameter groups "
            f"({sum(p.numel() for p in trainable_params):,} parameters)"
        )
        
        # Create optimizer
        optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-8,
        )
        
        # Create scheduler with warmup + cosine annealing
        total_steps = len(self.train_loader) * self.config.num_epochs
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=self.config.warmup_steps,
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - self.config.warmup_steps,
            eta_min=1e-7,
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps],
        )
        
        logger.info(
            f"Created optimizer with lr={self.config.learning_rate}, "
            f"warmup={self.config.warmup_steps}, total_steps={total_steps}"
        )
        
        return optimizer, scheduler
    
    def prepare_training(self) -> None:
        """Prepare all components for training."""
        # Prepare data
        self.train_loader, self.val_loader = self.prepare_data()
        
        # Prepare optimizer
        self.optimizer, self.scheduler = self.prepare_optimizer()
        
        # Setup mixed precision
        if self.config.use_amp:
            self.scaler = GradScaler()
            logger.info("Enabled automatic mixed precision training")
        
        # Save initial configuration
        self.save_config()
    
    def train_step(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """Perform single training step.
        
        Args:
            batch: Batch from dataloader.
            
        Returns:
            Tuple of (loss, loss_components).
        """
        # Move batch to device
        sent1_inputs = {k: v.to(self.device) for k, v in batch["sentence1_inputs"].items()}
        sent2_inputs = {k: v.to(self.device) for k, v in batch["sentence2_inputs"].items()}
        labels = batch["labels"].to(self.device) / 5.0  # Normalize to [0, 1]
        timestamps1 = batch["timestamps1"].to(self.device)
        timestamps2 = batch["timestamps2"].to(self.device)
        
        # Determine device type for autocast
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        
        # Forward pass with mixed precision
        with autocast(device_type=device_type, enabled=self.config.use_amp):
            # Encode both sentences
            temporal_emb1, base_emb1 = self.model(
                sent1_inputs["input_ids"],
                sent1_inputs["attention_mask"],
                timestamps1,
            )
            temporal_emb2, base_emb2 = self.model(
                sent2_inputs["input_ids"],
                sent2_inputs["attention_mask"],
                timestamps2,
            )
            
            # Concatenate for combined loss
            temporal_emb = torch.cat([temporal_emb1, temporal_emb2], dim=0)
            base_emb = torch.cat([base_emb1, base_emb2], dim=0)
            timestamps = torch.cat([timestamps1, timestamps2], dim=0)
            
            # Compute combined loss
            loss, loss_components = combined_tide_loss(
                temporal_emb,
                base_emb,
                timestamps,
                target_scores=labels,
                alpha=self.config.temporal_weight,
                beta=self.config.preservation_weight,
                tau_seconds=self.config.tau_seconds,
            )
        
        return loss, loss_components
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number.
            
        Returns:
            Dictionary of epoch metrics.
        """
        self.model.train()
        
        epoch_losses = []
        epoch_components = {"task": [], "temporal": [], "preservation": []}
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            disable=self.config.dry_run,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            loss, components = self.train_step(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            epoch_components["task"].append(components["task_loss"])
            epoch_components["temporal"].append(components["temporal_loss"])
            epoch_components["preservation"].append(components["preservation_loss"])
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
            
            self.global_step += 1
            
            # Checkpoint if needed
            if self.global_step % self.config.save_every_n_steps == 0:
                self.save_checkpoint(f"step_{self.global_step}")
            
            # Evaluate if needed
            if self.global_step % self.config.eval_every_n_steps == 0:
                val_metrics = self.evaluate()
                self.model.train()
                logger.info(f"Step {self.global_step} - Val loss: {val_metrics['loss']:.4f}")
        
        # Compute epoch statistics
        epoch_metrics = {
            "loss": sum(epoch_losses) / len(epoch_losses),
            "task_loss": sum(epoch_components["task"]) / len(epoch_components["task"]),
            "temporal_loss": sum(epoch_components["temporal"]) / len(epoch_components["temporal"]),
            "preservation_loss": sum(epoch_components["preservation"]) / len(epoch_components["preservation"]),
            "learning_rate": self.scheduler.get_last_lr()[0],
        }
        
        return epoch_metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set.
        
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        val_losses = []
        predictions = []
        gold_scores = []
        
        for batch in tqdm(self.val_loader, desc="Evaluating", disable=self.config.dry_run):
            # Move batch to device
            sent1_inputs = {k: v.to(self.device) for k, v in batch["sentence1_inputs"].items()}
            sent2_inputs = {k: v.to(self.device) for k, v in batch["sentence2_inputs"].items()}
            labels = batch["labels"].to(self.device)
            timestamps1 = batch["timestamps1"].to(self.device)
            timestamps2 = batch["timestamps2"].to(self.device)
            
            # Forward pass
            temporal_emb1, _ = self.model(
                sent1_inputs["input_ids"],
                sent1_inputs["attention_mask"],
                timestamps1,
            )
            temporal_emb2, _ = self.model(
                sent2_inputs["input_ids"],
                sent2_inputs["attention_mask"],
                timestamps2,
            )
            
            # Compute cosine similarity
            emb1_norm = torch.nn.functional.normalize(temporal_emb1, p=2, dim=1)
            emb2_norm = torch.nn.functional.normalize(temporal_emb2, p=2, dim=1)
            cosine_sim = torch.sum(emb1_norm * emb2_norm, dim=1)
            
            # Scale back to [0, 5] range
            pred_scores = (cosine_sim + 1.0) * 2.5
            
            predictions.extend(pred_scores.cpu().numpy())
            gold_scores.extend(labels.cpu().numpy())
            
            # Compute loss
            loss = cosine_regression_loss(temporal_emb1, temporal_emb2, labels)
            val_losses.append(loss.item())
        
        # Compute Spearman correlation
        from scipy.stats import spearmanr
        spearman_corr = spearmanr(predictions, gold_scores)[0]
        
        val_metrics = {
            "loss": sum(val_losses) / len(val_losses),
            "spearman": spearman_corr,
        }
        
        return val_metrics
    
    def train(self) -> Dict[str, Any]:
        """Run complete training pipeline.
        
        Returns:
            Dictionary of final metrics and statistics.
        """
        if self.config.dry_run:
            return self.dry_run_summary()
        
        logger.info("Starting training")
        start_time = time.time()
        
        # Prepare training
        self.prepare_training()
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics = self.evaluate()
            
            # Track metrics
            self.metrics["train_loss"].append(train_metrics["loss"])
            self.metrics["val_loss"].append(val_metrics["loss"])
            self.metrics["val_spearman"].append(val_metrics["spearman"])
            self.metrics["learning_rate"].append(train_metrics["learning_rate"])
            self.metrics["epoch_times"].append(time.time() - epoch_start)
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train loss: {train_metrics['loss']:.4f}, "
                f"Val loss: {val_metrics['loss']:.4f}, "
                f"Val Spearman: {val_metrics['spearman']:.4f}"
            )
            
            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}")
            
            # Save metrics
            self.save_metrics()
        
        # Save final model
        self.save_checkpoint("final")
        
        # Training summary
        total_time = time.time() - start_time
        final_metrics = {
            "total_time_seconds": total_time,
            "final_train_loss": self.metrics["train_loss"][-1],
            "final_val_loss": self.metrics["val_loss"][-1],
            "final_val_spearman": self.metrics["val_spearman"][-1],
            "best_val_spearman": max(self.metrics["val_spearman"]),
            "total_steps": self.global_step,
        }
        
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        logger.info(f"Final validation Spearman: {final_metrics['final_val_spearman']:.4f}")
        
        return final_metrics
    
    def save_checkpoint(self, name: str) -> None:
        """Save model checkpoint.
        
        Args:
            name: Checkpoint identifier.
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{name}.pt"
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "metrics": self.metrics,
            "config": asdict(self.config),
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Saved checkpoint to {checkpoint_path}")
    
    def save_metrics(self) -> None:
        """Save training metrics to JSON."""
        metrics_path = self.output_dir / "metrics_train.json"
        
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.debug(f"Saved metrics to {metrics_path}")
    
    def save_config(self) -> None:
        """Save configuration used for training."""
        config_path = self.output_dir / "config_used.json"
        
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
    
    def dry_run_summary(self) -> Dict[str, Any]:
        """Generate dry-run summary without executing training.
        
        Returns:
            Dictionary describing what would be executed.
        """
        # Mock data loader creation for step calculation
        from math import ceil
        
        estimated_train_samples = 5749  # STS-B train size
        estimated_val_samples = 1500    # STS-B validation size
        
        steps_per_epoch = ceil(estimated_train_samples / self.config.batch_size)
        total_steps = steps_per_epoch * self.config.num_epochs
        
        # Parameter count
        param_summary = self.model.get_parameter_summary()
        
        # Time estimates (rough)
        time_per_step = 0.2  # seconds, for small model on GPU
        estimated_time = total_steps * time_per_step
        
        summary = {
            "dry_run": True,
            "model": {
                "encoder": self.config.encoder_name,
                "trainable_params": param_summary["trainable_params"],
                "extra_params": param_summary["extra_params"],
            },
            "data": {
                "dataset": "STS-B with synthetic timestamps",
                "train_samples": estimated_train_samples,
                "val_samples": estimated_val_samples,
                "batch_size": self.config.batch_size,
                "steps_per_epoch": steps_per_epoch,
            },
            "training": {
                "num_epochs": self.config.num_epochs,
                "total_steps": total_steps,
                "warmup_steps": self.config.warmup_steps,
                "learning_rate": self.config.learning_rate,
                "optimizer": "AdamW",
                "scheduler": "LinearWarmup + CosineAnnealing",
            },
            "loss": {
                "task_loss": "Cosine Regression (STS-B)",
                "temporal_weight": self.config.temporal_weight,
                "preservation_weight": self.config.preservation_weight,
                "tau_seconds": self.config.tau_seconds,
            },
            "compute": {
                "device": str(self.device),
                "mixed_precision": self.config.use_amp,
                "estimated_time_seconds": estimated_time,
                "estimated_time_hours": estimated_time / 3600,
            },
            "outputs": {
                "output_dir": str(self.output_dir),
                "checkpoint_dir": str(self.checkpoint_dir),
                "checkpoint_frequency": self.config.save_every_n_steps,
                "eval_frequency": self.config.eval_every_n_steps,
            },
        }
        
        # Print summary
        logger.info("=" * 60)
        logger.info("DRY RUN SUMMARY - No training will be performed")
        logger.info("=" * 60)
        logger.info(json.dumps(summary, indent=2))
        logger.info("=" * 60)
        
        # Save summary
        summary_path = self.output_dir / "dry_run_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        return summary
