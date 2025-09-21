"""Training manager for dynamic embeddings."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List, Tuple
import logging
from tqdm import tqdm
import wandb
from pathlib import Path


logger = logging.getLogger(__name__)


class DynamicEmbeddingTrainer:
    """Trainer for dynamic embedding models.

    Handles training loop, validation, checkpointing, and logging
    for all dynamic embedding model variants.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        gradient_clip: float = 1.0,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_wandb: bool = True,
        project_name: str = 'dynamic-embeddings'
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            max_steps: Maximum training steps
            gradient_clip: Gradient clipping value
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
            log_wandb: Whether to log to Weights & Biases
            project_name: W&B project name
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.gradient_clip = gradient_clip

        # Create optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler(warmup_steps, max_steps)

        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.log_wandb = log_wandb
        if log_wandb:
            wandb.init(project=project_name, config={
                'learning_rate': learning_rate,
                'warmup_steps': warmup_steps,
                'max_steps': max_steps,
                'model_config': model.config if hasattr(model, 'config') else {}
            })

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

        logger.info(f"Trainer initialized with {sum(p.numel() for p in model.parameters())} parameters")

    def _create_scheduler(self, warmup_steps: int, max_steps: int):
        """Create learning rate scheduler with linear warmup."""
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0,
                      float(max_steps - step) / float(max(1, max_steps - warmup_steps)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            batch: Batch of training data
            loss_fn: Loss function

        Returns:
            Dictionary of metrics
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Forward pass
        outputs = self.model(**batch)

        # Compute loss
        loss = loss_fn(outputs, batch)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # Update global step
        self.global_step += 1

        # Collect metrics
        metrics = {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0],
            'gradient_norm': self._compute_gradient_norm()
        }

        return metrics

    def validate(
        self,
        loss_fn: Callable
    ) -> Dict[str, float]:
        """Run validation.

        Args:
            loss_fn: Loss function

        Returns:
            Validation metrics
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = loss_fn(outputs, batch)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint('best_model.pt')

        return {'val_loss': avg_loss}

    def train(
        self,
        num_epochs: int,
        loss_fn: Callable,
        val_frequency: int = 1000,
        checkpoint_frequency: int = 5000,
        log_frequency: int = 100
    ):
        """Main training loop.

        Args:
            num_epochs: Number of training epochs
            loss_fn: Loss function
            val_frequency: Validation frequency in steps
            checkpoint_frequency: Checkpoint frequency in steps
            log_frequency: Logging frequency in steps
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            epoch_metrics = []

            with tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch in pbar:
                    # Training step
                    metrics = self.train_step(batch, loss_fn)
                    epoch_metrics.append(metrics)

                    # Update progress bar
                    pbar.set_postfix({
                        'loss': metrics['loss'],
                        'lr': metrics['learning_rate']
                    })

                    # Logging
                    if self.global_step % log_frequency == 0:
                        avg_metrics = self._average_metrics(epoch_metrics[-log_frequency:])
                        if self.log_wandb:
                            wandb.log(avg_metrics, step=self.global_step)
                        logger.info(f"Step {self.global_step}: {avg_metrics}")

                    # Validation
                    if self.global_step % val_frequency == 0:
                        val_metrics = self.validate(loss_fn)
                        if self.log_wandb:
                            wandb.log(val_metrics, step=self.global_step)
                        logger.info(f"Validation: {val_metrics}")

                    # Checkpointing
                    if self.global_step % checkpoint_frequency == 0:
                        self.save_checkpoint(f'checkpoint_{self.global_step}.pt')

            # End of epoch validation
            val_metrics = self.validate(loss_fn)
            logger.info(f"Epoch {epoch+1} validation: {val_metrics}")

        # Save final model
        self.save_checkpoint('final_model.pt')
        logger.info("Training completed")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Checkpoint loaded from {path}")

    def _compute_gradient_norm(self) -> float:
        """Compute gradient norm."""
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Average metrics over multiple steps."""
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
        return avg_metrics