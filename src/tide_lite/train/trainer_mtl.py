"""Multi-task training epoch function for TIDE-Lite."""

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def train_epoch_multitask(self, epoch: int) -> Dict[str, float]:
    """Train for one epoch with multi-task learning.
    
    Alternates between STS-B (semantic similarity) and temporal batches
    based on the configured ratio.
    
    Args:
        epoch: Current epoch number.
        
    Returns:
        Dictionary of average metrics for the epoch including:
        - loss: Total loss
        - cosine_loss: Semantic similarity loss from STS-B
        - temporal_loss: Temporal consistency loss
        - stsb_steps: Number of STS-B batches processed
        - temporal_steps: Number of temporal batches processed
    """
    self.model.train()
    
    # Initialize metrics tracking
    epoch_metrics = {
        "loss": 0.0,
        "cosine_loss": 0.0,
        "temporal_loss": 0.0,
        "preservation_loss": 0.0,
        "stsb_steps": 0,
        "temporal_steps": 0,
    }
    
    # Create iterators for both datasets
    stsb_iter = iter(self.stsb_train_loader)
    temporal_iter = iter(self.temporal_train_loader) if self.temporal_train_loader else None
    
    # Calculate total steps for this epoch
    total_stsb_batches = len(self.stsb_train_loader)
    total_temporal_batches = len(self.temporal_train_loader) if self.temporal_train_loader else 0
    
    # Determine the number of iterations based on mtl_ratio
    if temporal_iter is not None:
        # With multi-task learning, alternate based on ratio
        # For ratio N:1, we do N STS-B batches for every 1 temporal batch
        total_steps = total_stsb_batches + total_temporal_batches
    else:
        # Without temporal data, just do STS-B
        total_steps = total_stsb_batches
    
    progress_bar = tqdm(
        range(total_steps),
        desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
        disable=self.config.dry_run,
    )
    
    step_counter = 0
    stsb_counter = 0
    temporal_counter = 0
    
    for global_step in progress_bar:
        # Determine which dataset to use based on ratio
        use_stsb = True  # Default to STS-B
        
        if temporal_iter is not None:
            # Multi-task learning logic
            # After every mtl_ratio STS-B batches, do 1 temporal batch
            if (stsb_counter > 0 and 
                stsb_counter % self.config.mtl_ratio == 0 and 
                temporal_counter < total_temporal_batches):
                use_stsb = False
        
        # Get batch from appropriate dataset
        if use_stsb:
            try:
                batch = next(stsb_iter)
                dataset_type = "stsb"
                stsb_counter += 1
                epoch_metrics["stsb_steps"] += 1
            except StopIteration:
                # Restart STS-B iterator if exhausted
                stsb_iter = iter(self.stsb_train_loader)
                batch = next(stsb_iter)
                dataset_type = "stsb"
                stsb_counter += 1
                epoch_metrics["stsb_steps"] += 1
        else:
            try:
                batch = next(temporal_iter)
                dataset_type = "temporal"
                temporal_counter += 1
                epoch_metrics["temporal_steps"] += 1
            except StopIteration:
                # If temporal exhausted, switch back to STS-B
                use_stsb = True
                batch = next(stsb_iter)
                dataset_type = "stsb"
                stsb_counter += 1
                epoch_metrics["stsb_steps"] += 1
        
        # Process batch based on dataset type
        if dataset_type == "stsb":
            loss, loss_dict = self._process_stsb_batch(batch)
        else:
            loss, loss_dict = self._process_temporal_batch(batch)
        
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
        epoch_metrics["loss"] += loss.item()
        for key, value in loss_dict.items():
            if key in epoch_metrics and key != "total":
                epoch_metrics[key] += value
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": loss.item(),
            "type": dataset_type[:1].upper(),  # S for STS-B, T for Temporal
            "lr": self.optimizer.param_groups[0]["lr"],
        })
        
        # Save checkpoint if needed
        step_counter += 1
        if self.config.save_every > 0 and step_counter % self.config.save_every == 0:
            self.save_checkpoint(f"step-{epoch * total_steps + step_counter}")
    
    # Average metrics
    total_batches = epoch_metrics["stsb_steps"] + epoch_metrics["temporal_steps"]
    if total_batches > 0:
        epoch_metrics["loss"] /= total_batches
        
        # Average losses by their respective batch counts
        if epoch_metrics["stsb_steps"] > 0:
            epoch_metrics["cosine_loss"] /= epoch_metrics["stsb_steps"]
        if epoch_metrics["temporal_steps"] > 0:
            epoch_metrics["temporal_loss"] /= epoch_metrics["temporal_steps"]
        if epoch_metrics["preservation_loss"] > 0:
            epoch_metrics["preservation_loss"] /= total_batches
    
    return epoch_metrics


def _process_stsb_batch(self, batch) -> tuple:
    """Process a batch from STS-B dataset (semantic similarity).
    
    Args:
        batch: Batch from STS-B dataloader containing sentence pairs and labels.
        
    Returns:
        Tuple of (loss, loss_dict) where loss_dict contains individual components.
    """
    # Move batch to device
    sent1_inputs = {k: v.to(self.device) for k, v in batch["sentence1_inputs"].items()}
    sent2_inputs = {k: v.to(self.device) for k, v in batch["sentence2_inputs"].items()}
    labels = batch["labels"].to(self.device)
    
    with autocast(enabled=self.config.use_amp):
        # Get base embeddings (no temporal modulation needed for STS-B)
        emb1 = self.model.encode_base(
            sent1_inputs["input_ids"],
            sent1_inputs["attention_mask"],
        )
        emb2 = self.model.encode_base(
            sent2_inputs["input_ids"],
            sent2_inputs["attention_mask"],
        )
        
        # Compute cosine regression loss only
        from ..train.losses import cosine_regression_loss
        loss = cosine_regression_loss(emb1, emb2, labels)
    
    return loss, {"cosine_loss": loss.item(), "temporal_loss": 0.0}


def _process_temporal_batch(self, batch) -> tuple:
    """Process a batch from temporal dataset.
    
    Args:
        batch: Batch from temporal dataloader containing text and timestamps.
        
    Returns:
        Tuple of (loss, loss_dict) where loss_dict contains individual components.
    """
    # Move batch to device
    if "input_ids" in batch:
        # Single sequence format (e.g., from TimeQA/TempLAMA)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
    else:
        # Extract from nested format if needed
        inputs = {k: v.to(self.device) for k, v in batch["inputs"].items()}
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
    
    # Get timestamps from batch (these come from the dataset, not synthetic)
    if "timestamps" in batch:
        timestamps = batch["timestamps"].to(self.device).float()
    elif "timestamp" in batch:
        timestamps = batch["timestamp"].to(self.device).float()
    else:
        # If no timestamps in batch, skip this batch with zero loss
        logger.warning("No timestamps found in temporal batch, skipping")
        return torch.tensor(0.0, device=self.device), {"temporal_loss": 0.0, "cosine_loss": 0.0}
    
    # Ensure timestamps are float and properly shaped
    if timestamps.dim() == 0:
        timestamps = timestamps.unsqueeze(0)
    if timestamps.dim() == 2:
        timestamps = timestamps.squeeze(1)
    
    with autocast(enabled=self.config.use_amp):
        # Get temporal embeddings
        temporal_emb, base_emb = self.model(
            input_ids,
            attention_mask,
            timestamps,
        )
        
        # Compute temporal consistency loss
        from ..train.losses import temporal_consistency_loss, preservation_loss
        
        # Temporal consistency: embeddings should vary smoothly over time
        temp_loss = temporal_consistency_loss(
            temporal_emb,
            timestamps,
            tau_seconds=self.config.tau_seconds,
        )
        
        # Preservation: don't deviate too much from base
        pres_loss = preservation_loss(
            temporal_emb,
            base_emb,
            alpha=self.config.preservation_weight,
        )
        
        # Combined loss with lambda weighting
        loss = self.config.lambda_temporal * temp_loss + pres_loss
    
    return loss, {
        "temporal_loss": temp_loss.item() if temp_loss.requires_grad else temp_loss,
        "preservation_loss": pres_loss.item() if pres_loss.requires_grad else pres_loss,
        "cosine_loss": 0.0,
    }
