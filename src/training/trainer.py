"""
Training module for Dynamic Embedding Model
Implements curriculum learning and contrastive objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import logging
import wandb
from collections import defaultdict

logger = logging.getLogger(__name__)


class DynamicEmbeddingTrainer:
    """
    Trainer for Dynamic Embedding Model
    Implements:
    - Curriculum learning (progressive training stages)
    - Contrastive learning objectives
    - Load balancing and diversity losses
    - Mixed precision training
    """

    def __init__(self, model, config, use_wandb: bool = False):
        """
        Initialize trainer

        Args:
            model: DynamicEmbeddingModel instance
            config: Training configuration
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.training.device)

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizers for different components
        self._setup_optimizers()

        # Loss functions
        self.contrastive_loss = ContrastiveLoss(
            temperature=config.training.temperature
        )

        # Mixed precision training
        self.use_amp = config.training.mixed_precision and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")

        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project="dynamic-embeddings",
                config=config.__dict__,
                name=config.experiment_name
            )

        # Training statistics
        self.training_stats = defaultdict(list)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _setup_optimizers(self):
        """Setup optimizers for different training stages"""
        config = self.config.training

        # Router optimizer
        self.router_optimizer = AdamW(
            self.model.router.parameters(),
            lr=config.router_lr,
            weight_decay=config.weight_decay
        )

        # Fusion optimizer
        fusion_params = list(self.model.fusion.parameters()) + \
                        list(self.model.output_projection.parameters())
        self.fusion_optimizer = AdamW(
            fusion_params,
            lr=config.fusion_lr,
            weight_decay=config.weight_decay
        )

        # Full model optimizer (for fine-tuning)
        self.full_optimizer = AdamW(
            self.model.parameters(),
            lr=config.overall_lr,
            weight_decay=config.weight_decay
        )

        # Learning rate schedulers
        self.router_scheduler = CosineAnnealingWarmRestarts(
            self.router_optimizer, T_0=10, T_mult=2
        )
        self.fusion_scheduler = CosineAnnealingWarmRestarts(
            self.fusion_optimizer, T_0=10, T_mult=2
        )
        self.full_scheduler = CosineAnnealingWarmRestarts(
            self.full_optimizer, T_0=10, T_mult=2
        )

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: Optional[int] = None):
        """
        Main training loop with curriculum learning

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Total epochs (overrides config if provided)
        """
        logger.info("=" * 60)
        logger.info("Starting Dynamic Embedding Training")
        logger.info("=" * 60)

        # Stage 1: Train router only
        logger.info("\nStage 1: Training Router")
        logger.info("-" * 40)
        self._train_stage(
            train_loader, val_loader,
            stage='router',
            epochs=self.config.training.router_epochs
        )

        # Stage 2: Train fusion only
        logger.info("\nStage 2: Training Fusion Layers")
        logger.info("-" * 40)
        self._train_stage(
            train_loader, val_loader,
            stage='fusion',
            epochs=self.config.training.fusion_epochs
        )

        # Stage 3: Fine-tune everything
        logger.info("\nStage 3: Fine-tuning All Components")
        logger.info("-" * 40)
        self._train_stage(
            train_loader, val_loader,
            stage='full',
            epochs=epochs or self.config.training.finetune_epochs
        )

        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)

    def _train_stage(self,
                     train_loader: DataLoader,
                     val_loader: Optional[DataLoader],
                     stage: str,
                     epochs: int):
        """
        Train a specific stage

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            stage: 'router', 'fusion', or 'full'
            epochs: Number of epochs for this stage
        """
        # Select optimizer and scheduler
        if stage == 'router':
            optimizer = self.router_optimizer
            scheduler = self.router_scheduler
            # Freeze other components
            self.model.freeze_experts()
            for param in self.model.fusion.parameters():
                param.requires_grad = False
        elif stage == 'fusion':
            optimizer = self.fusion_optimizer
            scheduler = self.fusion_scheduler
            # Freeze router and experts
            self.model.freeze_experts()
            for param in self.model.router.parameters():
                param.requires_grad = False
        else:  # full
            optimizer = self.full_optimizer
            scheduler = self.full_scheduler
            # Unfreeze everything
            self.model.unfreeze_experts()
            for param in self.model.parameters():
                param.requires_grad = True

        for epoch in range(epochs):
            # Training epoch
            train_loss, train_metrics = self._train_epoch(
                train_loader, optimizer, stage
            )

            # Validation
            if val_loader:
                val_loss, val_metrics = self._validate(val_loader)
            else:
                val_loss, val_metrics = None, {}

            # Update scheduler
            scheduler.step()

            # Logging
            log_str = f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f}"
            if val_loss:
                log_str += f" | Val Loss: {val_loss:.4f}"
            logger.info(log_str)

            if self.use_wandb:
                wandb.log({
                    f"{stage}/train_loss": train_loss,
                    f"{stage}/val_loss": val_loss,
                    f"{stage}/lr": optimizer.param_groups[0]['lr'],
                    **{f"{stage}/{k}": v for k, v in train_metrics.items()},
                    **{f"{stage}/val_{k}": v for k, v in val_metrics.items()}
                })

            # Early stopping
            if val_loss and stage == 'full':
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # Save best model
                    self.save_checkpoint(f"best_model_{stage}.pt")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.training.patience:
                        logger.info("Early stopping triggered")
                        break

        # Restore best model for this stage
        if val_loader and stage == 'full':
            self.load_checkpoint(f"best_model_{stage}.pt")

    def _train_epoch(self,
                     train_loader: DataLoader,
                     optimizer: torch.optim.Optimizer,
                     stage: str) -> Tuple[float, Dict]:
        """
        Train one epoch

        Returns:
            epoch_loss: Average loss for the epoch
            metrics: Dictionary of metrics
        """
        self.model.train()
        total_loss = 0
        metrics = defaultdict(float)

        progress_bar = tqdm(train_loader, desc=f"Training {stage}")

        for batch_idx, batch in enumerate(progress_bar):
            texts, labels = batch

            # Mixed precision context
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Forward pass
                outputs = self.model(texts, return_details=True)
                embeddings = outputs['embedding']
                routing_weights = outputs['routing_weights']
                auxiliary = outputs.get('auxiliary', {})

                # Contrastive loss
                loss = self.contrastive_loss(embeddings, labels)

                # Add auxiliary losses
                if auxiliary and 'load_balance_loss' in auxiliary:
                    loss += auxiliary['load_balance_loss']
                    metrics['load_balance_loss'] += auxiliary['load_balance_loss'].item()

                if auxiliary and 'diversity_loss' in auxiliary:
                    loss += self.config.training.diversity_weight * auxiliary['diversity_loss']
                    metrics['diversity_loss'] += auxiliary['diversity_loss'].item()

            # Backward pass
            optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                optimizer.step()

            # Update metrics
            total_loss += loss.item()
            metrics['routing_entropy'] += (-routing_weights * torch.log(routing_weights + 1e-10)).sum(
                dim=-1).mean().item()
            metrics['routing_sparsity'] += (routing_weights > 0.01).float().mean().item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'entropy': metrics['routing_entropy'] / (batch_idx + 1)
            })

        # Average metrics
        num_batches = len(train_loader)
        epoch_loss = total_loss / num_batches
        for key in metrics:
            metrics[key] /= num_batches

        return epoch_loss, dict(metrics)

    def _validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """
        Validate model

        Returns:
            val_loss: Average validation loss
            metrics: Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0
        metrics = defaultdict(float)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                texts, labels = batch

                # Forward pass
                outputs = self.model(texts, return_details=True)
                embeddings = outputs['embedding']
                routing_weights = outputs['routing_weights']

                # Contrastive loss
                loss = self.contrastive_loss(embeddings, labels)

                # Update metrics
                total_loss += loss.item()
                metrics['routing_entropy'] += (-routing_weights * torch.log(routing_weights + 1e-10)).sum(
                    dim=-1).mean().item()
                metrics['routing_sparsity'] += (routing_weights > 0.01).float().mean().item()

        # Average metrics
        num_batches = len(val_loader)
        val_loss = total_loss / num_batches
        for key in metrics:
            metrics[key] /= num_batches

        return val_loss, dict(metrics)

    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'router_optimizer': self.router_optimizer.state_dict(),
            'fusion_optimizer': self.fusion_optimizer.state_dict(),
            'full_optimizer': self.full_optimizer.state_dict(),
            'router_scheduler': self.router_scheduler.state_dict(),
            'fusion_scheduler': self.fusion_scheduler.state_dict(),
            'full_scheduler': self.full_scheduler.state_dict(),
            'training_stats': dict(self.training_stats),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        path = Path(self.config.data.output_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, filename: str):
        """Load training checkpoint"""
        path = Path(self.config.data.output_dir) / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.router_optimizer.load_state_dict(checkpoint['router_optimizer'])
        self.fusion_optimizer.load_state_dict(checkpoint['fusion_optimizer'])
        self.full_optimizer.load_state_dict(checkpoint['full_optimizer'])
        self.router_scheduler.load_state_dict(checkpoint['router_scheduler'])
        self.fusion_scheduler.load_state_dict(checkpoint['fusion_scheduler'])
        self.full_scheduler.load_state_dict(checkpoint['full_scheduler'])
        self.training_stats = defaultdict(list, checkpoint.get('training_stats', {}))
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        logger.info(f"Checkpoint loaded from {path}")


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for embedding learning
    Based on SimCLR/MoCo approaches
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute contrastive loss

        Args:
            embeddings: Tensor of shape [batch_size, hidden_dim]
            labels: Optional labels for supervised contrastive learning

        Returns:
            loss: Contrastive loss value
        """
        batch_size = embeddings.shape[0]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.t())
        similarity_matrix = similarity_matrix / self.temperature

        # Create labels (each sample is its own class for now)
        if labels is None:
            labels = torch.arange(batch_size, device=embeddings.device)

        # Mask out self-similarity
        mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        similarity_matrix.masked_fill_(mask, -9999)

        # Compute loss
        loss = self.cross_entropy(similarity_matrix, labels)

        return loss