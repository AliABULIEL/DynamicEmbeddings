"""Temporal evaluation module for TIDE-Lite models.

This module evaluates temporal understanding through accuracy
and consistency metrics on time-aware QA tasks.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.collate import TemporalQACollator, TextBatcher
from ..data.datasets import DatasetConfig, load_timeqa_lite
from ..models.tide_lite import TIDELite
from ..models.baselines import BaselineEncoder

logger = logging.getLogger(__name__)


@dataclass
class TemporalMetrics:
    """Metrics for temporal evaluation.
    
    Attributes:
        temporal_accuracy_at_1: Top-1 accuracy within time window.
        temporal_accuracy_at_5: Top-5 accuracy within time window.
        temporal_consistency_score: Correlation between time distance and embedding distance.
        time_window_precision: Precision of temporal localization.
        time_drift_mae: Mean absolute error of temporal drift in days.
        avg_inference_time_ms: Average inference time per sample.
        total_eval_time_s: Total evaluation time in seconds.
        num_samples: Number of evaluated samples.
    """
    temporal_accuracy_at_1: float
    temporal_accuracy_at_5: float
    temporal_consistency_score: float
    time_window_precision: float
    time_drift_mae: float
    avg_inference_time_ms: float
    total_eval_time_s: float
    num_samples: int


class TemporalEvaluator:
    """Evaluator for temporal understanding tasks.
    
    Evaluates how well models capture temporal information
    through QA accuracy and embedding consistency metrics.
    """
    
    def __init__(
        self,
        model: Union[TIDELite, BaselineEncoder],
        device: Optional[torch.device] = None,
        time_window_seconds: float = 86400.0 * 30,  # 30 days
    ) -> None:
        """Initialize temporal evaluator.
        
        Args:
            model: Model to evaluate (TIDE-Lite or baseline).
            device: Device for computation (auto-detect if None).
            time_window_seconds: Acceptable time window for temporal accuracy.
        """
        self.model = model
        self.time_window_seconds = time_window_seconds
        self.is_tide_lite = isinstance(model, TIDELite)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(
            f"Initialized temporal evaluator with model type: {type(model).__name__}, "
            f"device: {self.device}, time_window: {time_window_seconds/86400:.1f} days"
        )
    
    def prepare_dataloader(
        self,
        batch_size: int = 32,
        max_samples: Optional[int] = None,
    ) -> DataLoader:
        """Prepare TimeQA-lite dataloader for evaluation.
        
        Args:
            batch_size: Batch size for evaluation.
            max_samples: Maximum number of samples (None for all).
            
        Returns:
            DataLoader for temporal QA evaluation.
        """
        # Load dataset
        dataset_config = DatasetConfig(
            seed=42,
            cache_dir="./data",
            max_samples=max_samples,
            timestamp_start="2020-01-01",
            timestamp_end="2024-01-01",
        )
        
        dataset = load_timeqa_lite(dataset_config)
        
        # Create tokenizer and collator
        tokenizer = TextBatcher(
            model_name=getattr(self.model, "config", None).encoder_name 
            if hasattr(self.model, "config") else "sentence-transformers/all-MiniLM-L6-v2",
            max_length=128,
        )
        
        collator = TemporalQACollator(
            tokenizer,
            max_context_length=384,
            max_question_length=64,
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )
        
        logger.info(f"Prepared TimeQA-lite dataloader with {len(dataloader)} batches")
        
        return dataloader
    
    @torch.no_grad()
    def encode_with_time(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text with temporal information.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            timestamps: Unix timestamps [batch_size].
            
        Returns:
            Embeddings [batch_size, hidden_dim].
        """
        if self.is_tide_lite:
            # TIDE-Lite with temporal modulation
            embeddings, _ = self.model(input_ids, attention_mask, timestamps)
        elif hasattr(self.model, "encode_base"):
            # Baseline or TIDE-Lite without temporal
            embeddings = self.model.encode_base(input_ids, attention_mask)
        else:
            # Generic forward pass
            embeddings, _ = self.model(input_ids, attention_mask, None)
        
        return embeddings
    
    def compute_temporal_accuracy(
        self,
        question_embs: torch.Tensor,
        context_embs: torch.Tensor,
        answer_indices: List[int],
        timestamps_q: torch.Tensor,
        timestamps_c: torch.Tensor,
        k: int = 5,
    ) -> Tuple[float, float]:
        """Compute temporal accuracy metrics.
        
        Args:
            question_embs: Question embeddings [n_questions, hidden_dim].
            context_embs: Context embeddings [n_contexts, hidden_dim].
            answer_indices: Correct answer indices for each question.
            timestamps_q: Question timestamps [n_questions].
            timestamps_c: Context timestamps [n_contexts].
            k: Top-k for accuracy computation.
            
        Returns:
            Tuple of (accuracy@1, accuracy@k).
        """
        # Normalize embeddings
        question_embs = F.normalize(question_embs, p=2, dim=1)
        context_embs = F.normalize(context_embs, p=2, dim=1)
        
        # Compute similarity matrix
        similarities = torch.matmul(question_embs, context_embs.T)
        
        # Get top-k predictions
        top_k_scores, top_k_indices = torch.topk(similarities, k=min(k, context_embs.size(0)), dim=1)
        
        correct_at_1 = 0
        correct_at_k = 0
        
        for i, correct_idx in enumerate(answer_indices):
            # Check if correct answer is in top-1
            if top_k_indices[i, 0].item() == correct_idx:
                # Check temporal constraint
                time_diff = abs(timestamps_q[i] - timestamps_c[correct_idx])
                if time_diff <= self.time_window_seconds:
                    correct_at_1 += 1
                    correct_at_k += 1
            # Check if correct answer is in top-k
            elif correct_idx in top_k_indices[i]:
                time_diff = abs(timestamps_q[i] - timestamps_c[correct_idx])
                if time_diff <= self.time_window_seconds:
                    correct_at_k += 1
        
        acc_at_1 = correct_at_1 / len(answer_indices) if answer_indices else 0.0
        acc_at_k = correct_at_k / len(answer_indices) if answer_indices else 0.0
        
        return acc_at_1, acc_at_k
    
    def compute_temporal_consistency(
        self,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> float:
        """Compute temporal consistency score.
        
        This metric measures how well embedding distances correlate
        with temporal distances, indicating temporal awareness.
        
        Args:
            embeddings: Embeddings [n_samples, hidden_dim].
            timestamps: Unix timestamps [n_samples].
            
        Returns:
            Temporal consistency score (Spearman correlation).
        """
        n_samples = embeddings.size(0)
        
        if n_samples < 2:
            return 0.0
        
        # Sample pairs for efficiency (max 1000 pairs)
        max_pairs = min(1000, n_samples * (n_samples - 1) // 2)
        
        if n_samples * (n_samples - 1) // 2 <= max_pairs:
            # Use all pairs
            indices = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)]
        else:
            # Random sample
            indices = []
            while len(indices) < max_pairs:
                i = np.random.randint(0, n_samples)
                j = np.random.randint(0, n_samples)
                if i != j and (i, j) not in indices and (j, i) not in indices:
                    indices.append((min(i, j), max(i, j)))
        
        # Compute distances
        time_distances = []
        embedding_distances = []
        
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        for i, j in indices:
            # Temporal distance (in days)
            time_dist = abs(timestamps[i] - timestamps[j]).item() / 86400.0
            time_distances.append(time_dist)
            
            # Embedding distance (1 - cosine similarity)
            cos_sim = torch.dot(embeddings_norm[i], embeddings_norm[j]).item()
            emb_dist = 1.0 - cos_sim
            embedding_distances.append(emb_dist)
        
        # Compute Spearman correlation
        # We expect: small time distance → small embedding distance
        # So we want negative correlation
        correlation, _ = spearmanr(time_distances, embedding_distances)
        
        # Convert to positive score (0 = no consistency, 1 = perfect consistency)
        consistency_score = max(0, -correlation)
        
        return consistency_score
    
    def compute_time_window_precision(
        self,
        predictions: List[int],
        ground_truth: List[int],
        timestamps: torch.Tensor,
    ) -> float:
        """Compute precision of temporal window predictions.
        
        Args:
            predictions: Predicted indices.
            ground_truth: Ground truth indices.
            timestamps: Timestamps for all samples.
            
        Returns:
            Temporal window precision.
        """
        if not predictions:
            return 0.0
        
        correct_window = 0
        
        for pred, gt in zip(predictions, ground_truth):
            time_diff = abs(timestamps[pred] - timestamps[gt]).item()
            if time_diff <= self.time_window_seconds:
                correct_window += 1
        
        return correct_window / len(predictions)
    
    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None,
        batch_size: int = 32,
        max_samples: Optional[int] = None,
    ) -> TemporalMetrics:
        """Evaluate model on temporal understanding tasks.
        
        Args:
            dataloader: Optional pre-configured dataloader.
            batch_size: Batch size if dataloader not provided.
            max_samples: Maximum samples if dataloader not provided.
            
        Returns:
            TemporalMetrics with evaluation results.
        """
        if dataloader is None:
            dataloader = self.prepare_dataloader(batch_size, max_samples)
        
        logger.info("Starting temporal evaluation")
        
        all_question_embs = []
        all_context_embs = []
        all_timestamps = []
        all_predictions = []
        all_ground_truth = []
        inference_times = []
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating temporal")):
            batch_start = time.perf_counter()
            
            # Move batch to device
            question_inputs = {k: v.to(self.device) for k, v in batch["question_inputs"].items()}
            context_inputs = {k: v.to(self.device) for k, v in batch["context_inputs"].items()}
            timestamps = batch["timestamps"].to(self.device)
            
            # Encode questions and contexts
            question_embs = self.encode_with_time(
                question_inputs["input_ids"],
                question_inputs["attention_mask"],
                timestamps,
            )
            
            context_embs = self.encode_with_time(
                context_inputs["input_ids"],
                context_inputs["attention_mask"],
                timestamps,
            )
            
            # Store for later analysis
            all_question_embs.append(question_embs)
            all_context_embs.append(context_embs)
            all_timestamps.append(timestamps)
            
            # Simple prediction: most similar context for each question
            q_norm = F.normalize(question_embs, p=2, dim=1)
            c_norm = F.normalize(context_embs, p=2, dim=1)
            similarities = torch.matmul(q_norm, c_norm.T)
            predictions = torch.argmax(similarities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            # For TimeQA-lite, assume diagonal is correct (question i → context i)
            all_ground_truth.extend(range(len(question_embs)))
            
            batch_time = (time.perf_counter() - batch_start) * 1000  # ms
            inference_times.append(batch_time / len(question_embs))
        
        # Concatenate all embeddings
        all_question_embs = torch.cat(all_question_embs, dim=0)
        all_context_embs = torch.cat(all_context_embs, dim=0)
        all_timestamps = torch.cat(all_timestamps, dim=0)
        
        # Compute temporal accuracy
        acc_at_1, acc_at_5 = self.compute_temporal_accuracy(
            all_question_embs,
            all_context_embs,
            all_ground_truth,
            all_timestamps,
            all_timestamps,  # Same timestamps for Q&A in TimeQA-lite
            k=5,
        )
        
        # Compute temporal consistency
        # Combine all embeddings for consistency analysis
        all_embeddings = torch.cat([all_question_embs, all_context_embs], dim=0)
        all_times_combined = torch.cat([all_timestamps, all_timestamps], dim=0)
        consistency_score = self.compute_temporal_consistency(all_embeddings, all_times_combined)
        
        # Compute time window precision
        window_precision = self.compute_time_window_precision(
            all_predictions,
            all_ground_truth,
            all_timestamps,
        )
        
        # Compute temporal drift
        time_drifts = []
        for pred, gt in zip(all_predictions, all_ground_truth):
            if pred < len(all_timestamps) and gt < len(all_timestamps):
                drift = abs(all_timestamps[pred] - all_timestamps[gt]).item() / 86400.0  # days
                time_drifts.append(drift)
        
        time_drift_mae = np.mean(time_drifts) if time_drifts else 0.0
        
        total_time = time.time() - start_time
        
        metrics = TemporalMetrics(
            temporal_accuracy_at_1=float(acc_at_1),
            temporal_accuracy_at_5=float(acc_at_5),
            temporal_consistency_score=float(consistency_score),
            time_window_precision=float(window_precision),
            time_drift_mae=float(time_drift_mae),
            avg_inference_time_ms=float(np.mean(inference_times)),
            total_eval_time_s=total_time,
            num_samples=len(all_predictions),
        )
        
        logger.info(
            f"Temporal evaluation complete - "
            f"Acc@1: {metrics.temporal_accuracy_at_1:.4f}, "
            f"Consistency: {metrics.temporal_consistency_score:.4f}, "
            f"Drift: {metrics.time_drift_mae:.1f} days"
        )
        
        return metrics
    
    def save_results(
        self,
        metrics: TemporalMetrics,
        output_dir: Union[str, Path],
        model_name: str = "model",
    ) -> Path:
        """Save evaluation metrics to JSON file.
        
        Args:
            metrics: Evaluation metrics.
            output_dir: Directory to save results.
            model_name: Model identifier for filename.
            
        Returns:
            Path to saved metrics file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        metrics_file = output_dir / f"metrics_temporal_{model_name}.json"
        
        # Prepare metrics dict
        metrics_dict = asdict(metrics)
        metrics_dict["model_name"] = model_name
        metrics_dict["task"] = "Temporal QA"
        metrics_dict["time_window_days"] = self.time_window_seconds / 86400.0
        
        # Save to JSON
        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Saved temporal metrics to {metrics_file}")
        
        return metrics_file
    
    def compare_temporal_awareness(
        self,
        baseline_model: BaselineEncoder,
        dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, TemporalMetrics]:
        """Compare temporal awareness between TIDE-Lite and baseline.
        
        Args:
            baseline_model: Baseline model for comparison.
            dataloader: Optional shared dataloader.
            
        Returns:
            Dictionary with metrics for both models.
        """
        logger.info("Evaluating TIDE-Lite temporal awareness")
        tide_metrics = self.evaluate(dataloader)
        
        logger.info("Evaluating baseline temporal awareness")
        baseline_evaluator = TemporalEvaluator(
            baseline_model,
            device=self.device,
            time_window_seconds=self.time_window_seconds,
        )
        baseline_metrics = baseline_evaluator.evaluate(dataloader)
        
        # Compute improvements
        consistency_improvement = (
            tide_metrics.temporal_consistency_score - 
            baseline_metrics.temporal_consistency_score
        )
        accuracy_improvement = (
            tide_metrics.temporal_accuracy_at_1 - 
            baseline_metrics.temporal_accuracy_at_1
        )
        
        logger.info(
            f"Temporal improvements over baseline - "
            f"Consistency: {consistency_improvement:+.4f}, "
            f"Accuracy: {accuracy_improvement:+.4f}"
        )
        
        return {
            "tide_lite": tide_metrics,
            "baseline": baseline_metrics,
        }
