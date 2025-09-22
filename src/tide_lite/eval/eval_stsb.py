"""STS-B evaluation module for TIDE-Lite models.

This module evaluates models on the Semantic Textual Similarity Benchmark,
computing Spearman correlation between predicted and gold similarity scores.
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
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.collate import STSBCollator, TextBatcher
from ..data.datasets import DatasetConfig, load_stsb_with_timestamps
from ..models.tide_lite import TIDELite
from ..models.baselines import BaselineEncoder

logger = logging.getLogger(__name__)


@dataclass
class STSBMetrics:
    """Metrics for STS-B evaluation.
    
    Attributes:
        spearman: Spearman correlation coefficient.
        pearson: Pearson correlation coefficient.
        mse: Mean squared error.
        mae: Mean absolute error.
        num_samples: Number of evaluated samples.
        avg_inference_time_ms: Average inference time per sample.
        total_eval_time_s: Total evaluation time in seconds.
    """
    spearman: float
    pearson: float
    mse: float
    mae: float
    num_samples: int
    avg_inference_time_ms: float
    total_eval_time_s: float


class STSBEvaluator:
    """Evaluator for STS-B semantic similarity task.
    
    Computes cosine similarity between sentence pairs and correlates
    with gold standard human similarity judgments.
    """
    
    def __init__(
        self,
        model: Union[TIDELite, BaselineEncoder],
        device: Optional[torch.device] = None,
        use_temporal: bool = True,
    ) -> None:
        """Initialize STS-B evaluator.
        
        Args:
            model: Model to evaluate (TIDE-Lite or baseline).
            device: Device for computation (auto-detect if None).
            use_temporal: Whether to use temporal modulation for TIDE-Lite.
        """
        self.model = model
        self.use_temporal = use_temporal and isinstance(model, TIDELite)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(
            f"Initialized STS-B evaluator with model type: {type(model).__name__}, "
            f"device: {self.device}, temporal: {self.use_temporal}"
        )
    
    def prepare_dataloader(
        self,
        split: str = "test",
        batch_size: int = 64,
        num_workers: int = 2,
    ) -> DataLoader:
        """Prepare STS-B dataloader for evaluation.
        
        Args:
            split: Dataset split ('validation' or 'test').
            batch_size: Batch size for evaluation.
            num_workers: DataLoader workers.
            
        Returns:
            DataLoader for the specified split.
        """
        # Load dataset
        dataset_config = DatasetConfig(
            seed=42,
            cache_dir="./data",
            timestamp_start="2020-01-01",
            timestamp_end="2024-01-01",
        )
        
        datasets = load_stsb_with_timestamps(dataset_config)
        
        if split not in datasets:
            raise ValueError(f"Split '{split}' not found. Choose from: {list(datasets.keys())}")
        
        # Create tokenizer and collator
        tokenizer = TextBatcher(
            model_name=getattr(self.model, "config", None).encoder_name 
            if hasattr(self.model, "config") else "sentence-transformers/all-MiniLM-L6-v2",
            max_length=128,
        )
        
        collator = STSBCollator(tokenizer, include_timestamps=self.use_temporal)
        
        # Create dataloader
        dataloader = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        logger.info(f"Prepared STS-B {split} dataloader with {len(dataloader)} batches")
        
        return dataloader
    
    @torch.no_grad()
    def encode_sentences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode sentences to embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            timestamps: Optional timestamps for temporal modulation.
            
        Returns:
            Sentence embeddings [batch_size, hidden_dim].
        """
        if self.use_temporal and timestamps is not None:
            # TIDE-Lite with temporal modulation
            embeddings, _ = self.model(input_ids, attention_mask, timestamps)
        elif hasattr(self.model, "encode_base"):
            # TIDE-Lite without temporal modulation or baseline with encode_base
            embeddings = self.model.encode_base(input_ids, attention_mask)
        else:
            # Generic forward pass
            embeddings, _ = self.model(input_ids, attention_mask, timestamps)
        
        return embeddings
    
    def compute_similarities(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarities between embedding pairs.
        
        Args:
            embeddings1: First set of embeddings [batch_size, hidden_dim].
            embeddings2: Second set of embeddings [batch_size, hidden_dim].
            
        Returns:
            Cosine similarities [batch_size].
        """
        # Normalize embeddings
        embeddings1_norm = F.normalize(embeddings1, p=2, dim=1)
        embeddings2_norm = F.normalize(embeddings2, p=2, dim=1)
        
        # Compute cosine similarity
        similarities = torch.sum(embeddings1_norm * embeddings2_norm, dim=1)
        
        return similarities
    
    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None,
        split: str = "test",
        batch_size: int = 64,
    ) -> STSBMetrics:
        """Evaluate model on STS-B dataset.
        
        Args:
            dataloader: Optional pre-configured dataloader.
            split: Dataset split if dataloader not provided.
            batch_size: Batch size if dataloader not provided.
            
        Returns:
            STSBMetrics with evaluation results.
        """
        if dataloader is None:
            dataloader = self.prepare_dataloader(split, batch_size)
        
        logger.info(f"Starting STS-B evaluation on {split} split")
        
        all_predictions = []
        all_gold_scores = []
        inference_times = []
        
        start_time = time.time()
        
        for batch in tqdm(dataloader, desc="Evaluating STS-B"):
            batch_start = time.perf_counter()
            
            # Move batch to device
            sent1_inputs = {k: v.to(self.device) for k, v in batch["sentence1_inputs"].items()}
            sent2_inputs = {k: v.to(self.device) for k, v in batch["sentence2_inputs"].items()}
            labels = batch["labels"]
            
            # Get timestamps if available
            timestamps1 = batch.get("timestamps1")
            timestamps2 = batch.get("timestamps2")
            
            if timestamps1 is not None:
                timestamps1 = timestamps1.to(self.device)
            if timestamps2 is not None:
                timestamps2 = timestamps2.to(self.device)
            
            # Encode sentences
            emb1 = self.encode_sentences(
                sent1_inputs["input_ids"],
                sent1_inputs["attention_mask"],
                timestamps1,
            )
            emb2 = self.encode_sentences(
                sent2_inputs["input_ids"],
                sent2_inputs["attention_mask"],
                timestamps2,
            )
            
            # Compute similarities
            similarities = self.compute_similarities(emb1, emb2)
            
            # Convert to [0, 5] scale
            predictions = (similarities.cpu().numpy() + 1.0) * 2.5
            
            all_predictions.extend(predictions)
            all_gold_scores.extend(labels.numpy())
            
            batch_time = (time.perf_counter() - batch_start) * 1000  # ms
            inference_times.append(batch_time / len(labels))
        
        total_time = time.time() - start_time
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        gold_scores = np.array(all_gold_scores)
        
        # Compute metrics
        spearman_corr, spearman_p = spearmanr(predictions, gold_scores)
        pearson_corr, pearson_p = pearsonr(predictions, gold_scores)
        mse = np.mean((predictions - gold_scores) ** 2)
        mae = np.mean(np.abs(predictions - gold_scores))
        
        metrics = STSBMetrics(
            spearman=float(spearman_corr),
            pearson=float(pearson_corr),
            mse=float(mse),
            mae=float(mae),
            num_samples=len(predictions),
            avg_inference_time_ms=float(np.mean(inference_times)),
            total_eval_time_s=total_time,
        )
        
        logger.info(
            f"STS-B evaluation complete - "
            f"Spearman: {metrics.spearman:.4f}, "
            f"Pearson: {metrics.pearson:.4f}, "
            f"MSE: {metrics.mse:.4f}"
        )
        
        return metrics
    
    def save_results(
        self,
        metrics: STSBMetrics,
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
        metrics_file = output_dir / f"metrics_stsb_{model_name}.json"
        
        # Prepare metrics dict
        metrics_dict = asdict(metrics)
        metrics_dict["model_name"] = model_name
        metrics_dict["task"] = "STS-B"
        metrics_dict["use_temporal"] = self.use_temporal
        
        # Save to JSON
        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Saved STS-B metrics to {metrics_file}")
        
        return metrics_file
    
    def compare_with_baseline(
        self,
        baseline_model: BaselineEncoder,
        dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, STSBMetrics]:
        """Compare TIDE-Lite with baseline model.
        
        Args:
            baseline_model: Baseline model for comparison.
            dataloader: Optional shared dataloader.
            
        Returns:
            Dictionary with metrics for both models.
        """
        logger.info("Evaluating TIDE-Lite model")
        tide_metrics = self.evaluate(dataloader)
        
        logger.info("Evaluating baseline model")
        baseline_evaluator = STSBEvaluator(
            baseline_model,
            device=self.device,
            use_temporal=False,
        )
        baseline_metrics = baseline_evaluator.evaluate(dataloader)
        
        # Compute improvements
        spearman_improvement = tide_metrics.spearman - baseline_metrics.spearman
        pearson_improvement = tide_metrics.pearson - baseline_metrics.pearson
        
        logger.info(
            f"Improvements over baseline - "
            f"Spearman: {spearman_improvement:+.4f}, "
            f"Pearson: {pearson_improvement:+.4f}"
        )
        
        return {
            "tide_lite": tide_metrics,
            "baseline": baseline_metrics,
        }


def load_model_for_evaluation(
    model_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Union[TIDELite, BaselineEncoder]:
    """Load model from checkpoint or path.
    
    Args:
        model_path: Path to model checkpoint or directory.
        device: Device to load model on.
        
    Returns:
        Loaded model ready for evaluation.
        
    Raises:
        FileNotFoundError: If model path doesn't exist.
        RuntimeError: If model loading fails.
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        if model_path.is_dir():
            # Load TIDE-Lite from directory
            model = TIDELite.from_pretrained(str(model_path))
        else:
            # Load from checkpoint file
            checkpoint = torch.load(model_path, map_location=device)
            
            # Reconstruct model from checkpoint
            if "config" in checkpoint:
                from ..models.tide_lite import TIDELiteConfig
                config = TIDELiteConfig(**checkpoint["config"])
                model = TIDELite(config)
            else:
                # Default config
                model = TIDELite()
            
            # Load state dict
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")
