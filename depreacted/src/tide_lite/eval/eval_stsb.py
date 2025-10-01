"""STS-B evaluation for TIDE-Lite and baseline models.

This module evaluates models on the STS-B benchmark using
Spearman's correlation with bootstrap confidence intervals.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from ..models.tide_lite import TIDELite
from ..models.baselines import load_baseline
from ..data.dataloaders import create_stsb_dataloaders

logger = logging.getLogger(__name__)


@dataclass
class STSBMetrics:
    """Metrics for STS-B evaluation.
    
    Attributes:
        spearman_rho: Spearman's rank correlation coefficient.
        spearman_ci_lower: Lower bound of 95% CI for Spearman.
        spearman_ci_upper: Upper bound of 95% CI for Spearman.
        pearson_r: Pearson correlation coefficient.
        pearson_ci_lower: Lower bound of 95% CI for Pearson.
        pearson_ci_upper: Upper bound of 95% CI for Pearson.
        mse: Mean squared error.
        num_samples: Number of evaluation samples.
    """
    spearman_rho: float
    spearman_ci_lower: float
    spearman_ci_upper: float
    pearson_r: float
    pearson_ci_lower: float
    pearson_ci_upper: float
    mse: float
    num_samples: int


def bootstrap_confidence_interval(
    predictions: np.ndarray,
    gold_scores: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.
    
    Args:
        predictions: Predicted similarity scores.
        gold_scores: Gold standard scores.
        metric_fn: Function to compute metric (e.g., spearmanr).
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (metric_value, ci_lower, ci_upper).
    """
    np.random.seed(seed)
    n_samples = len(predictions)
    
    # Compute metric on original data
    if metric_fn == spearmanr:
        metric_value = metric_fn(predictions, gold_scores)[0]
    elif metric_fn == pearsonr:
        metric_value = metric_fn(predictions, gold_scores)[0]
    else:
        metric_value = metric_fn(predictions, gold_scores)
    
    # Bootstrap resampling
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        pred_sample = predictions[indices]
        gold_sample = gold_scores[indices]
        
        # Compute metric on resampled data
        try:
            if metric_fn == spearmanr:
                bootstrap_metric = metric_fn(pred_sample, gold_sample)[0]
            elif metric_fn == pearsonr:
                bootstrap_metric = metric_fn(pred_sample, gold_sample)[0]
            else:
                bootstrap_metric = metric_fn(pred_sample, gold_sample)
            bootstrap_metrics.append(bootstrap_metric)
        except:
            # Handle edge cases where correlation can't be computed
            continue
    
    # Compute confidence interval
    bootstrap_metrics = np.array(bootstrap_metrics)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
    
    return metric_value, ci_lower, ci_upper


class STSBEvaluator:
    """Evaluator for STS-B benchmark."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 64,
        max_seq_length: int = 128,
    ) -> None:
        """Initialize STS-B evaluator.
        
        Args:
            device: Device to use (auto-detect if None).
            batch_size: Batch size for evaluation.
            max_seq_length: Maximum sequence length.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        
        logger.info(f"Initialized STSBEvaluator with device: {self.device}")
    
    def load_model(
        self,
        model_id_or_path: str,
        model_type: str = "tide_lite",
    ) -> Union[TIDELite, torch.nn.Module]:
        """Load a model for evaluation.
        
        Args:
            model_id_or_path: Model identifier or path to saved model.
            model_type: Type of model ('tide_lite' or baseline name).
            
        Returns:
            Loaded model.
        """
        if model_type == "tide_lite":
            # Load TIDE-Lite model
            if Path(model_id_or_path).exists():
                # Load from local path
                model = TIDELite.from_pretrained(model_id_or_path)
                logger.info(f"Loaded TIDE-Lite from {model_id_or_path}")
            else:
                # Initialize new model with encoder name
                from ..models.tide_lite import TIDELiteConfig
                config = TIDELiteConfig(encoder_name=model_id_or_path)
                model = TIDELite(config)
                logger.info(f"Initialized TIDE-Lite with encoder {model_id_or_path}")
        else:
            # Load baseline model
            model = load_baseline(model_type)
            logger.info(f"Loaded baseline model: {model_type}")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    @torch.no_grad()
    def compute_embeddings(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        use_timestamps: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
        """Compute embeddings for all samples in dataloader.
        
        Args:
            model: Model to evaluate.
            dataloader: DataLoader with STS-B samples.
            use_timestamps: Whether to use temporal modulation (TIDE-Lite only).
            
        Returns:
            Tuple of (embeddings1, embeddings2, gold_scores).
        """
        all_emb1 = []
        all_emb2 = []
        all_scores = []
        
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            # Move batch to device
            sent1_inputs = {k: v.to(self.device) for k, v in batch["sentence1_inputs"].items()}
            sent2_inputs = {k: v.to(self.device) for k, v in batch["sentence2_inputs"].items()}
            labels = batch["labels"]
            
            # Generate timestamps if using temporal
            if use_timestamps and isinstance(model, TIDELite):
                batch_size = labels.shape[0]
                timestamps1 = torch.rand(batch_size, device=self.device) * 1e9
                timestamps2 = timestamps1 + torch.randn(batch_size, device=self.device) * 3600
                
                # Get temporal embeddings
                emb1, _ = model(
                    sent1_inputs["input_ids"],
                    sent1_inputs["attention_mask"],
                    timestamps1,
                )
                emb2, _ = model(
                    sent2_inputs["input_ids"],
                    sent2_inputs["attention_mask"],
                    timestamps2,
                )
            else:
                # Get base embeddings
                if hasattr(model, 'encode_base'):
                    emb1 = model.encode_base(
                        sent1_inputs["input_ids"],
                        sent1_inputs["attention_mask"],
                    )
                    emb2 = model.encode_base(
                        sent2_inputs["input_ids"],
                        sent2_inputs["attention_mask"],
                    )
                else:
                    # For baseline models
                    emb1, _ = model(
                        sent1_inputs["input_ids"],
                        sent1_inputs["attention_mask"],
                    )
                    emb2, _ = model(
                        sent2_inputs["input_ids"],
                        sent2_inputs["attention_mask"],
                    )
            
            all_emb1.append(emb1.cpu())
            all_emb2.append(emb2.cpu())
            all_scores.extend(labels.tolist())
        
        return all_emb1, all_emb2, all_scores
    
    def compute_metrics(
        self,
        embeddings1: List[torch.Tensor],
        embeddings2: List[torch.Tensor],
        gold_scores: List[float],
        n_bootstrap: int = 1000,
    ) -> STSBMetrics:
        """Compute evaluation metrics with confidence intervals.
        
        Args:
            embeddings1: First set of embeddings.
            embeddings2: Second set of embeddings.
            gold_scores: Gold similarity scores [0, 5].
            n_bootstrap: Number of bootstrap iterations.
            
        Returns:
            STSBMetrics with all computed metrics.
        """
        # Concatenate all embeddings
        emb1 = torch.cat(embeddings1, dim=0)
        emb2 = torch.cat(embeddings2, dim=0)
        gold_scores = np.array(gold_scores)
        
        # Normalize embeddings
        emb1_norm = F.normalize(emb1, p=2, dim=1)
        emb2_norm = F.normalize(emb2, p=2, dim=1)
        
        # Compute cosine similarities
        cosine_sims = torch.sum(emb1_norm * emb2_norm, dim=1).numpy()
        
        # Scale predictions to [0, 5] for comparison
        pred_scores = cosine_sims * 5.0
        
        # Compute Spearman correlation with bootstrap CI
        spearman_rho, spearman_lower, spearman_upper = bootstrap_confidence_interval(
            pred_scores,
            gold_scores,
            spearmanr,
            n_bootstrap=n_bootstrap,
        )
        
        # Compute Pearson correlation with bootstrap CI
        pearson_r, pearson_lower, pearson_upper = bootstrap_confidence_interval(
            pred_scores,
            gold_scores,
            pearsonr,
            n_bootstrap=n_bootstrap,
        )
        
        # Compute MSE
        mse = np.mean((pred_scores - gold_scores) ** 2)
        
        return STSBMetrics(
            spearman_rho=float(spearman_rho),
            spearman_ci_lower=float(spearman_lower),
            spearman_ci_upper=float(spearman_upper),
            pearson_r=float(pearson_r),
            pearson_ci_lower=float(pearson_lower),
            pearson_ci_upper=float(pearson_upper),
            mse=float(mse),
            num_samples=len(gold_scores),
        )
    
    def evaluate(
        self,
        model_id_or_path: str,
        model_type: str = "tide_lite",
        split: str = "test",
        use_timestamps: bool = False,
        n_bootstrap: int = 1000,
        output_dir: Optional[str] = None,
        save_results: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, any]:
        """Evaluate a model on STS-B.
        
        Args:
            model_id_or_path: Model identifier or path.
            model_type: Type of model.
            split: Dataset split to evaluate on.
            use_timestamps: Whether to use temporal modulation.
            n_bootstrap: Number of bootstrap iterations for CI.
            output_dir: Directory to save results.
            save_results: Whether to save results to JSON.
            dry_run: If True, just print plan without execution.
            
        Returns:
            Dictionary with evaluation results.
        """
        if dry_run:
            logger.info("[DRY RUN] Would evaluate model on STS-B:")
            logger.info(f"  Model: {model_id_or_path}")
            logger.info(f"  Type: {model_type}")
            logger.info(f"  Split: {split}")
            logger.info(f"  Temporal: {use_timestamps}")
            logger.info(f"  Bootstrap iterations: {n_bootstrap}")
            if save_results:
                model_name = Path(model_id_or_path).name if "/" in model_id_or_path else model_id_or_path
                output_file = f"results/metrics_stsb_{model_name}.json"
                logger.info(f"  Would save to: {output_file}")
            return {
                "dry_run": True,
                "model": model_id_or_path,
                "split": split,
            }
        
        # Load model
        logger.info(f"Loading model: {model_id_or_path}")
        model = self.load_model(model_id_or_path, model_type)
        
        # Load data
        logger.info(f"Loading STS-B {split} split")
        data_config = {
            "cache_dir": "./data",
            "seed": 42,
            "model_name": model.config.encoder_name if hasattr(model, 'config') else model.model_name,
        }
        
        train_loader, val_loader, test_loader = create_stsb_dataloaders(
            cfg=data_config,
            batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            num_workers=2,
        )
        
        # Select appropriate loader
        if split == "train":
            dataloader = train_loader
        elif split == "validation":
            dataloader = val_loader
        else:
            dataloader = test_loader
        
        # Compute embeddings
        logger.info("Computing embeddings...")
        embeddings1, embeddings2, gold_scores = self.compute_embeddings(
            model, dataloader, use_timestamps
        )
        
        # Compute metrics
        logger.info("Computing metrics with bootstrap CI...")
        metrics = self.compute_metrics(
            embeddings1, embeddings2, gold_scores, n_bootstrap
        )
        
        # Prepare results
        results = {
            "model": model_id_or_path,
            "model_type": model_type,
            "split": split,
            "use_timestamps": use_timestamps,
            "metrics": asdict(metrics),
            "config": {
                "batch_size": self.batch_size,
                "max_seq_length": self.max_seq_length,
                "n_bootstrap": n_bootstrap,
            },
        }
        
        # Print results
        logger.info("=" * 60)
        logger.info("STS-B Evaluation Results")
        logger.info("=" * 60)
        logger.info(f"Model: {model_id_or_path}")
        logger.info(f"Split: {split}")
        logger.info(f"Samples: {metrics.num_samples}")
        logger.info("-" * 40)
        logger.info(f"Spearman ρ: {metrics.spearman_rho:.4f}")
        logger.info(f"  95% CI: [{metrics.spearman_ci_lower:.4f}, {metrics.spearman_ci_upper:.4f}]")
        logger.info(f"Pearson r: {metrics.pearson_r:.4f}")
        logger.info(f"  95% CI: [{metrics.pearson_ci_lower:.4f}, {metrics.pearson_ci_upper:.4f}]")
        logger.info(f"MSE: {metrics.mse:.4f}")
        logger.info("=" * 60)
        
        # Save results
        if save_results:
            if output_dir is None:
                output_dir = "results"
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create filename based on model
            model_name = Path(model_id_or_path).name if "/" in model_id_or_path else model_id_or_path
            model_name = model_name.replace("/", "_")
            output_file = output_path / f"metrics_stsb_{model_name}.json"
            
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to: {output_file}")
        
        return results


def evaluate_stsb(
    model_id_or_path: str,
    model_type: str = "tide_lite",
    split: str = "test",
    use_timestamps: bool = False,
    n_bootstrap: int = 1000,
    batch_size: int = 64,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, any]:
    """Convenience function to evaluate a model on STS-B.
    
    Args:
        model_id_or_path: Model identifier or path.
        model_type: Type of model.
        split: Dataset split to evaluate on.
        use_timestamps: Whether to use temporal modulation.
        n_bootstrap: Number of bootstrap iterations.
        batch_size: Batch size for evaluation.
        output_dir: Directory to save results.
        dry_run: If True, just print plan without execution.
        
    Returns:
        Dictionary with evaluation results.
    """
    evaluator = STSBEvaluator(batch_size=batch_size)
    
    return evaluator.evaluate(
        model_id_or_path=model_id_or_path,
        model_type=model_type,
        split=split,
        use_timestamps=use_timestamps,
        n_bootstrap=n_bootstrap,
        output_dir=output_dir,
        dry_run=dry_run,
    )
