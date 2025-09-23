"""Trainer for TIDE-Lite model.

This module provides the main training logic for TIDE-Lite,
training only the temporal MLP while keeping the encoder frozen.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.tide_lite import TIDELite, TIDELiteConfig
from ..data.dataloaders import create_stsb_dataloaders
from .losses import TIDELiteLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for TIDE-Lite training.
    
    Attributes:
        # Model
        encoder_name: Base encoder model name.
        time_encoding_dim: Dimension of temporal encoding.
        mlp_hidden_dim: Hidden dimension of temporal MLP.
        mlp_dropout: Dropout rate in MLP.
        gate_activation: Activation for gating ('sigmoid' or 'tanh').
        
        # Training
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay for AdamW.
        warmup_steps: Number of warmup steps.
        
        # Loss weights
        temporal_weight: Weight for temporal consistency loss.
        preservation_weight: Weight for preservation loss.
        tau_seconds: Time constant for temporal consistency.
        
        # Data
        max_seq_length: Maximum sequence length.
        num_workers: Number of dataloader workers.
        cache_dir: Cache directory for datasets.
        
        # Optimization
        use_amp: Whether to use automatic mixed precision.
        gradient_clip: Gradient clipping value.
        
        # Checkpointing
        output_dir: Output directory for results.
        save_every: Save checkpoint every N steps (0 = epoch only).
        eval_every: Evaluate every N steps (0 = epoch only).
        
        # Other
        seed: Random seed.
        dry_run: Whether to just print the plan without execution.
    """
    # Model
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    time_encoding_dim: int = 32
    mlp_hidden_dim: int = 128
    mlp_dropout: float = 0.1
    gate_activation: str = "sigmoid"
    
    # Training
    num_epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Loss weights
    temporal_weight: float = 0.1
    preservation_weight: float = 0.05
    tau_seconds: float = 86400.0
    
    # Data
    max_seq_length: int = 128
    num_workers: int = 2
    cache_dir: str = "./data"
    
    # Optimization
    use_amp: bool = True
    gradient_clip: float = 1.0
    
    # Checkpointing
    output_dir: Optional[str] = None
    save_every: int = 0
    eval_every: int = 0
    
    # Other
    seed: int = 42
    dry_run: bool = True  # Default to dry-run


class TIDELiteTrainer:
    """Trainer for TIDE-Lite model."""
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[TIDELite] = None,
    ) -> None:
        """Initialize trainer.
        
        Args:
            config: Training configuration.
            model: Optional pre-initialized model.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        # Initialize model
        if model is None:
            model_config = TIDELiteConfig(
                encoder_name=config.encoder_name,
                time_encoding_dim=config.time_encoding_dim,
                mlp_hidden_dim=config.mlp_hidden_dim,
                mlp_dropout=config.mlp_dropout,
                gate_activation=config.gate_activation,
                freeze_encoder=True,  # Always freeze encoder
                max_seq_length=config.max_seq_length,
            )
            self.model = TIDELite(model_config)
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # Setup output directory
        if config.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            config.output_dir = f"results/run-{timestamp}"
        
        self.output_dir = Path(config.output_dir)
        if not config.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loss function
        self.loss_fn = TIDELiteLoss(
            temporal_weight=config.temporal_weight,
            preservation_weight=config.preservation_weight,
            tau_seconds=config.tau_seconds,
        )
        
        # Training components (initialized in setup_training)
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Metrics tracking
        self.metrics = {
            "train": [],
            "val": [],
            "test": [],
        }
        
        logger.info(f"Initialized TIDELiteTrainer with device: {self.device}")
        logger.info(f"Model has {self.model.count_extra_parameters():,} trainable parameters")
    
    def setup_training(self) -> None:
        """Setup training components: optimizer, scheduler, dataloaders."""
        # Only optimize temporal MLP parameters
        trainable_params = [
            p for p in self.model.temporal_gate.parameters() 
            if p.requires_grad
        ]
        
        # Optimizer
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Scheduler
        total_steps = self.config.num_epochs * len(self.train_loader) if self.train_loader else 1000
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
        )
        
        # Mixed precision scaler
        if self.config.use_amp:
            self.scaler = GradScaler()
        
        logger.info(f"Setup optimizer with {len(trainable_params)} parameter groups")
    
    def setup_dataloaders(self) -> None:
        """Setup dataloaders for STS-B."""
        logger.info("Setting up STS-B dataloaders")
        
        data_config = {
            "cache_dir": self.config.cache_dir,
            "seed": self.config.seed,
            "model_name": self.config.encoder_name,
        }
        
        self.train_loader, self.val_loader, self.test_loader = create_stsb_dataloaders(
            cfg=data_config,
            batch_size=self.config.batch_size,
            eval_batch_size=self.config.eval_batch_size,
            max_seq_length=self.config.max_seq_length,
            num_workers=self.config.num_workers,
        )
        
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info(f"Test batches: {len(self.test_loader)}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number.
            
        Returns:
            Dictionary of average metrics for the epoch.
        """
        self.model.train()
        epoch_metrics = {
            "loss": 0.0,
            "cosine_loss": 0.0,
            "temporal_loss": 0.0,
            "preservation_loss": 0.0,
        }
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            disable=self.config.dry_run,
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            sent1_inputs = {k: v.to(self.device) for k, v in batch["sentence1_inputs"].items()}
            sent2_inputs = {k: v.to(self.device) for k, v in batch["sentence2_inputs"].items()}
            labels = batch["labels"].to(self.device)
            
            # Generate random timestamps for training
            batch_size = labels.shape[0]
            timestamps1 = torch.rand(batch_size, device=self.device) * 1e9  # Random timestamps
            timestamps2 = timestamps1 + torch.randn(batch_size, device=self.device) * 3600  # ±1 hour
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                # Get embeddings
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
                
                # Compute loss
                loss_dict = self.loss_fn(
                    temporal_emb1=temporal_emb1,
                    temporal_emb2=temporal_emb2,
                    base_emb1=base_emb1,
                    base_emb2=base_emb2,
                    timestamps1=timestamps1,
                    timestamps2=timestamps2,
                    gold_scores=labels,
                )
                
                loss = loss_dict["total"]
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.temporal_gate.parameters(),
                        self.config.gradient_clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.temporal_gate.parameters(),
                        self.config.gradient_clip,
                    )
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update metrics
            for key, value in loss_dict.items():
                if key == "total":
                    epoch_metrics["loss"] += value.item()
                else:
                    epoch_metrics[key] += value.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "lr": self.optimizer.param_groups[0]["lr"],
            })
            
            # Save checkpoint if needed
            global_step = epoch * len(self.train_loader) + step
            if self.config.save_every > 0 and global_step % self.config.save_every == 0:
                self.save_checkpoint(f"step-{global_step}")
        
        # Average metrics
        num_batches = len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, split: str = "val") -> Dict[str, float]:
        """Evaluate on a dataset.
        
        Args:
            dataloader: DataLoader to evaluate on.
            split: Name of the split for logging.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_cosine_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Evaluating {split}", disable=self.config.dry_run):
            # Move batch to device
            sent1_inputs = {k: v.to(self.device) for k, v in batch["sentence1_inputs"].items()}
            sent2_inputs = {k: v.to(self.device) for k, v in batch["sentence2_inputs"].items()}
            labels = batch["labels"].to(self.device)
            
            # Generate timestamps
            batch_size = labels.shape[0]
            timestamps1 = torch.rand(batch_size, device=self.device) * 1e9
            timestamps2 = timestamps1 + torch.randn(batch_size, device=self.device) * 3600
            
            # Forward pass
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
            
            # Compute loss
            loss_dict = self.loss_fn(
                temporal_emb1=temporal_emb1,
                temporal_emb2=temporal_emb2,
                base_emb1=base_emb1,
                base_emb2=base_emb2,
                timestamps1=timestamps1,
                timestamps2=timestamps2,
                gold_scores=labels,
            )
            
            total_loss += loss_dict["total"].item()
            total_cosine_loss += loss_dict["cosine_loss"].item()
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "cosine_loss": total_cosine_loss / num_batches,
        }
    
    def save_checkpoint(self, name: str = "checkpoint") -> None:
        """Save model checkpoint.
        
        Args:
            name: Name for the checkpoint.
        """
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would save checkpoint: {name}")
            return
        
        checkpoint_path = self.output_dir / f"{name}.pt"
        
        checkpoint = {
            "model_state_dict": self.model.temporal_gate.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": asdict(self.config),
            "metrics": self.metrics,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def save_metrics(self) -> None:
        """Save training metrics to JSON."""
        if self.config.dry_run:
            logger.info("[DRY RUN] Would save metrics to metrics_train.json")
            return
        
        metrics_path = self.output_dir / "metrics_train.json"
        
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")
    
    def save_config(self) -> None:
        """Save configuration used for training."""
        if self.config.dry_run:
            logger.info("[DRY RUN] Would save config to config_used.json")
            return
        
        config_path = self.output_dir / "config_used.json"
        
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Saved config to {config_path}")
    
    def train(self) -> None:
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("Starting TIDE-Lite Training")
        logger.info("=" * 60)
        
        if self.config.dry_run:
            logger.info("[DRY RUN MODE] Printing training plan:")
            logger.info(f"  Model: {self.config.encoder_name}")
            logger.info(f"  Trainable params: {self.model.count_extra_parameters():,}")
            logger.info(f"  Epochs: {self.config.num_epochs}")
            logger.info(f"  Batch size: {self.config.batch_size}")
            logger.info(f"  Learning rate: {self.config.learning_rate}")
            logger.info(f"  Output dir: {self.output_dir}")
            logger.info("  Would train temporal MLP on STS-B dataset")
            logger.info("  Would save checkpoints and metrics")
            logger.info("Exiting dry run.")
            return
        
        # Setup dataloaders and training components
        self.setup_dataloaders()
        self.setup_training()
        
        # Save initial configuration
        self.save_config()
        
        # Training loop
        best_val_loss = float("inf")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.metrics["train"].append({
                "epoch": epoch + 1,
                **train_metrics,
            })
            
            logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Cosine: {train_metrics['cosine_loss']:.4f}"
            )
            
            # Validate
            val_metrics = self.evaluate(self.val_loader, "val")
            self.metrics["val"].append({
                "epoch": epoch + 1,
                **val_metrics,
            })
            
            logger.info(
                f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"Cosine: {val_metrics['cosine_loss']:.4f}"
            )
            
            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best")
                logger.info("Saved best model")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch-{epoch + 1}")
        
        # Final evaluation on test set
        test_metrics = self.evaluate(self.test_loader, "test")
        self.metrics["test"].append(test_metrics)
        
        logger.info(
            f"\nTest - Loss: {test_metrics['loss']:.4f}, "
            f"Cosine: {test_metrics['cosine_loss']:.4f}"
        )
        
        # Save final results
        self.save_metrics()
        self.model.save_pretrained(self.output_dir / "final_model")
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 60)


def train_tide_lite(config: TrainingConfig) -> TIDELite:
    """Convenience function to train TIDE-Lite.
    
    Args:
        config: Training configuration.
        
    Returns:
        Trained TIDE-Lite model.
    """
    trainer = TIDELiteTrainer(config)
    trainer.train()
    return trainer.model
