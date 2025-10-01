#!/usr/bin/env python
"""Advanced training script for TIDE-Lite with temporal fixes.

This script uses real temporal data, advanced loss functions,
and proper temporal evaluation metrics.

Usage:
    python scripts/train_temporal.py --dataset reddit --num-epochs 20
    python scripts/train_temporal.py --dataset mixed --evaluate-only
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.tide_lite.models.tide_lite import TIDELite, TIDELiteConfig
from src.tide_lite.data.temporal_datasets import load_temporal_dataset
from src.tide_lite.data.collate import STSBCollator, TextBatcher
from src.tide_lite.train.advanced_losses import (
    advanced_temporal_loss,
    SemanticDriftLoss,
    EventAwareTemporalLoss,
)
from src.tide_lite.evaluation.temporal_metrics import evaluate_temporal_model
from src.tide_lite.utils.config import set_global_seed, setup_logging

logger = logging.getLogger(__name__)


class TemporalTrainer:
    """Advanced trainer for temporal embeddings."""
    
    def __init__(
        self,
        model: TIDELite,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
    ):
        """Initialize temporal trainer.
        
        Args:
            model: TIDE-Lite model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 3e-5),
            weight_decay=config.get("weight_decay", 0.01),
        )
        
        # Setup scheduler
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(train_loader) * 2,  # Restart every 2 epochs
            T_mult=2,  # Double period after each restart
        )
        
        # Initialize loss functions
        self.drift_loss_fn = SemanticDriftLoss(drift_model="adaptive")
        self.event_loss_fn = EventAwareTemporalLoss()
        
        # Tracking
        self.metrics_history = []
        self.best_score = 0.0
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        loss_components = {
            "task": [], "drift": [], "event": [], "cyclic": [], "preservation": []
        }
        
        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            # Move to device
            sent1_inputs = {k: v.to(self.device) for k, v in batch["sentence1_inputs"].items()}
            sent2_inputs = {k: v.to(self.device) for k, v in batch["sentence2_inputs"].items()}
            labels = (batch["label"].to(self.device) / 5.0)  # Normalize
            ts1 = batch["timestamp1"].to(self.device)
            ts2 = batch["timestamp2"].to(self.device)
            
            # Forward pass
            temporal_emb1, base_emb1 = self.model(
                sent1_inputs["input_ids"],
                sent1_inputs["attention_mask"],
                ts1
            )
            temporal_emb2, base_emb2 = self.model(
                sent2_inputs["input_ids"],
                sent2_inputs["attention_mask"],
                ts2
            )
            
            # Concatenate for loss
            temporal_emb = torch.cat([temporal_emb1, temporal_emb2], dim=0)
            base_emb = torch.cat([base_emb1, base_emb2], dim=0)
            timestamps = torch.cat([ts1, ts2], dim=0)
            
            # Compute advanced temporal loss
            loss, components = advanced_temporal_loss(
                temporal_emb,
                base_emb,
                timestamps,
                target_scores=labels,
                metadata=batch.get("metadata"),
                alpha_drift=self.config.get("alpha_drift", 0.05),
                alpha_event=self.config.get("alpha_event", 0.03),
                alpha_cyclic=self.config.get("alpha_cyclic", 0.02),
                beta_preservation=self.config.get("beta_preservation", 0.01),
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            for key in loss_components:
                if key in components:
                    loss_components[key].append(components[key])
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Compute epoch averages
        avg_loss = total_loss / len(self.train_loader)
        avg_components = {
            key: sum(values) / len(values) if values else 0
            for key, values in loss_components.items()
        }
        
        return avg_loss, avg_components
    
    def evaluate(self):
        """Evaluate model on validation set."""
        from src.tide_lite.evaluation.temporal_metrics import (
            evaluate_temporal_model,
            TemporalEvalResult
        )
        
        result = evaluate_temporal_model(
            self.model,
            self.val_loader,
            self.device,
            verbose=False
        )
        
        return result
    
    def train(self, num_epochs: int):
        """Run full training loop."""
        print("\n" + "="*60)
        print("TEMPORAL TIDE-LITE TRAINING")
        print("="*60)
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_components = self.train_epoch(epoch)
            
            # Evaluate
            eval_result = self.evaluate()
            
            # Compute combined score
            combined_score = (
                eval_result.spearman_correlation * 0.4 +
                eval_result.temporal_consistency * 0.2 +
                eval_result.drift_alignment * 0.2 +
                eval_result.event_robustness * 0.1 +
                eval_result.future_generalization * 0.1
            )
            
            # Track metrics
            self.metrics_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_components": train_components,
                "eval_metrics": eval_result.to_dict(),
                "combined_score": combined_score,
            })
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Components: task={train_components['task']:.4f}, "
                  f"drift={train_components['drift']:.4f}, "
                  f"event={train_components['event']:.4f}")
            print(f"  Eval: Spearman={eval_result.spearman_correlation:.4f}, "
                  f"Temporal={eval_result.temporal_consistency:.4f}, "
                  f"Drift={eval_result.drift_alignment:.4f}")
            print(f"  Combined score: {combined_score:.4f}")
            
            # Save best model
            if combined_score > self.best_score:
                self.best_score = combined_score
                self.save_checkpoint(f"best_model.pt")
                print(f"  ✨ New best model! Score: {combined_score:.4f}")
            
            # Save metrics
            self.save_metrics()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        output_dir = Path(self.config.get("output_dir", "results/temporal"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "best_score": self.best_score,
            "metrics_history": self.metrics_history,
        }
        
        torch.save(checkpoint, output_dir / filename)
        logger.info(f"Saved checkpoint to {output_dir / filename}")
    
    def save_metrics(self):
        """Save training metrics."""
        output_dir = Path(self.config.get("output_dir", "results/temporal"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(self.metrics_history, f, indent=2, default=str)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train temporal TIDE-Lite")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="mixed",
        choices=["reddit", "news", "wiki", "mixed"],
        help="Temporal dataset to use"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/temporal_run",
        help="Output directory"
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only run evaluation"
    )
    
    args = parser.parse_args()
    
    # Setup
    set_global_seed(42)
    setup_logging("INFO")
    
    # Initialize model
    print("Initializing model...")
    model_config = TIDELiteConfig(
        encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        mlp_hidden_dim=128,
        freeze_encoder=True,
    )
    model = TIDELite(model_config)
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    train_dataset = load_temporal_dataset(args.dataset, "train")
    val_dataset = load_temporal_dataset(args.dataset, "validation")
    
    # Create tokenizer and collator
    tokenizer = TextBatcher(
        model_name=model_config.encoder_name,
        max_length=128,
    )
    collator = STSBCollator(tokenizer, include_timestamps=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Training config
    config = {
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "alpha_drift": 0.05,
        "alpha_event": 0.03,
        "alpha_cyclic": 0.02,
        "beta_preservation": 0.01,
        "output_dir": args.output_dir,
    }
    
    # Initialize trainer
    trainer = TemporalTrainer(model, train_loader, val_loader, config)
    
    if args.evaluate_only:
        print("\nRunning evaluation only...")
        result = trainer.evaluate()
        print("\nEvaluation Results:")
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\nStarting training...")
        trainer.train(args.num_epochs)
        print(f"\nTraining complete! Best score: {trainer.best_score:.4f}")
        print(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
