"""Unit tests for training components."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import sys

sys.path.append('..')

from src.models.matryoshka_embedding import MatryoshkaEmbedding
from src.training.trainer import DynamicEmbeddingTrainer
from src.training.loss_functions import (
    ContrastiveLoss,
    MatryoshkaLoss,
    TemporalConsistencyLoss
)


class TestLossFunctions:
    """Tests for loss functions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 8
        self.embedding_dim = 128

    def test_contrastive_loss(self):
        """Test contrastive loss computation."""
        loss_fn = ContrastiveLoss(temperature=0.07)

        # Create embeddings
        embeddings = torch.randn(self.batch_size, self.embedding_dim)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        # Compute loss
        loss = loss_fn(embeddings, labels)

        assert isinstance(loss.item(), float)
        assert loss.item() > 0

        # Test with self-supervised mode (no labels)
        loss_self = loss_fn(embeddings)
        assert isinstance(loss_self.item(), float)
        assert loss_self.item() > 0

    def test_matryoshka_loss(self):
        """Test Matryoshka loss computation."""
        dimensions = [64, 128]
        loss_fn = MatryoshkaLoss(dimensions=dimensions, temperature=0.05)

        # Create embeddings at different dimensions
        embeddings_dict = {
            64: torch.randn(self.batch_size, 64),
            128: torch.randn(self.batch_size, 128)
        }

        batch = {'labels': torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])}

        # Compute loss
        loss = loss_fn(embeddings_dict, batch)

        assert isinstance(loss.item(), float)
        assert loss.item() > 0

    def test_temporal_consistency_loss(self):
        """Test temporal consistency loss."""
        loss_fn = TemporalConsistencyLoss(alpha=1.0, beta=0.5)

        # Create sequence embeddings
        embeddings = torch.randn(self.batch_size, 16, self.embedding_dim)
        timestamps = torch.arange(16).unsqueeze(0).expand(self.batch_size, -1).float()

        # Compute loss
        loss = loss_fn(embeddings, timestamps)

        assert isinstance(loss.item(), float)
        assert loss.item() >= 0


class TestTrainer:
    """Tests for the trainer class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = MatryoshkaEmbedding(
            input_dim=100,
            embedding_dim=64,
            nested_dims=[32, 64],
            num_layers=1
        )

        # Create dummy data
        inputs = torch.randint(0, 100, (32, 16))
        labels = torch.randint(0, 4, (32,))
        dataset = TensorDataset(inputs, labels)

        self.train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        self.val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    def test_trainer_initialization(self, tmp_path):
        """Test trainer initialization."""
        trainer = DynamicEmbeddingTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            val_dataloader=self.val_loader,
            learning_rate=1e-4,
            device='cpu',
            checkpoint_dir=str(tmp_path / 'checkpoints'),
            log_wandb=False
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.global_step == 0
        assert trainer.best_val_loss == float('inf')

    def test_train_step(self, tmp_path):
        """Test single training step."""
        trainer = DynamicEmbeddingTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            val_dataloader=self.val_loader,
            device='cpu',
            checkpoint_dir=str(tmp_path / 'checkpoints'),
            log_wandb=False
        )

        # Define simple loss function
        def loss_fn(outputs, batch):
            return outputs.mean()

        # Get batch
        batch = next(iter(self.train_loader))
        batch = {'inputs': batch[0], 'labels': batch[1]}

        # Run training step
        metrics = trainer.train_step(batch, loss_fn)

        assert 'loss' in metrics
        assert 'learning_rate' in metrics
        assert 'gradient_norm' in metrics
        assert trainer.global_step == 1

    def test_validation(self, tmp_path):
        """Test validation loop."""
        trainer = DynamicEmbeddingTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            val_dataloader=self.val_loader,
            device='cpu',
            checkpoint_dir=str(tmp_path / 'checkpoints'),
            log_wandb=False
        )

        def loss_fn(outputs, batch):
            return outputs.mean()

        # Run validation
        val_metrics = trainer.validate(loss_fn)

        assert 'val_loss' in val_metrics
        assert isinstance(val_metrics['val_loss'], float)

    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint saving and loading."""
        trainer = DynamicEmbeddingTrainer(
            model=self.model,
            train_dataloader=self.train_loader,
            val_dataloader=self.val_loader,
            device='cpu',
            checkpoint_dir=str(tmp_path / 'checkpoints'),
            log_wandb=False
        )

        # Modify state
        trainer.global_step = 100
        trainer.best_val_loss = 0.5

        # Save checkpoint
        trainer.save_checkpoint('test.pt')

        # Reset state
        trainer.global_step = 0
        trainer.best_val_loss = float('inf')

        # Load checkpoint
        trainer.load_checkpoint('test.pt')

        assert trainer.global_step == 100
        assert trainer.best_val_loss == 0.5


class TestOptimizers:
    """Tests for optimizer configurations."""

    def test_adamw_optimizer(self):
        """Test AdamW optimizer setup."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # Check optimizer state
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]['lr'] == 1e-4
        assert optimizer.param_groups[0]['weight_decay'] == 0.01

    def test_lr_scheduler(self):
        """Test learning rate scheduling."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Linear warmup + decay
        def lr_lambda(step):
            warmup_steps = 100
            max_steps = 1000
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0,
                       float(max_steps - step) / float(max(1, max_steps - warmup_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Test warmup phase
        lrs = []
        for step in range(200):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # Check warmup increases lr
        assert lrs[50] < lrs[99]
        # Check decay decreases lr
        assert lrs[150] < lrs[100]