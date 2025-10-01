"""Temporal evaluation metrics and benchmarks for dynamic embeddings.

This module provides comprehensive evaluation tools specifically designed
for temporal embedding models, going beyond standard similarity metrics.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TemporalEvalResult:
    """Results from temporal evaluation."""
    
    # Standard metrics
    spearman_correlation: float
    pearson_correlation: float
    mse: float
    mae: float
    
    # Temporal-specific metrics
    temporal_consistency: float
    drift_alignment: float
    event_robustness: float
    future_generalization: float
    
    # Detailed breakdowns
    metrics_by_time_delta: Dict[str, float]
    metrics_by_domain: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "standard": {
                "spearman": self.spearman_correlation,
                "pearson": self.pearson_correlation,
                "mse": self.mse,
                "mae": self.mae,
            },
            "temporal": {
                "consistency": self.temporal_consistency,
                "drift_alignment": self.drift_alignment,
                "event_robustness": self.event_robustness,
                "future_generalization": self.future_generalization,
            },
            "breakdowns": {
                "by_time_delta": self.metrics_by_time_delta,
                "by_domain": self.metrics_by_domain,
            }
        }


class TemporalConsistencyMetric:
    """Measure temporal consistency of embeddings.
    
    Tests whether embeddings maintain consistent relationships
    over time and evolve predictably.
    """
    
    def __init__(self, model, device=None):
        """Initialize metric.
        
        Args:
            model: The TIDE-Lite model to evaluate
            device: Computation device
        """
        self.model = model
        self.device = device or torch.device("cpu")
    
    def compute(
        self,
        texts: List[str],
        base_timestamp: float,
        time_deltas: List[float] = [3600, 86400, 604800, 2592000],  # 1hr, 1day, 1week, 1month
    ) -> float:
        """Compute temporal consistency score.
        
        Args:
            texts: List of texts to embed
            base_timestamp: Starting timestamp
            time_deltas: Time differences to test (in seconds)
        
        Returns:
            Consistency score between 0 and 1
        """
        self.model.eval()
        consistencies = []
        
        with torch.no_grad():
            for text in texts:
                embeddings = []
                
                # Embed text at different timestamps
                for delta in [0] + time_deltas:
                    timestamp = torch.tensor([base_timestamp + delta], device=self.device)
                    
                    # Tokenize (simplified - in practice use proper tokenizer)
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(self.model.config.encoder_name)
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get embedding
                    emb, _ = self.model(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        timestamp
                    )
                    embeddings.append(emb)
                
                # Stack embeddings
                embeddings = torch.cat(embeddings, dim=0)  # [num_timestamps, hidden_dim]
                
                # Compute consistency: correlation between consecutive embeddings
                for i in range(len(embeddings) - 1):
                    emb1 = F.normalize(embeddings[i:i+1], p=2, dim=1)
                    emb2 = F.normalize(embeddings[i+1:i+2], p=2, dim=1)
                    similarity = torch.sum(emb1 * emb2).item()
                    
                    # Weight by time delta (expect more change over longer periods)
                    expected_sim = np.exp(-time_deltas[i] / 86400 / 30)  # Decay over months
                    consistency = 1.0 - abs(similarity - expected_sim)
                    consistencies.append(consistency)
        
        return np.mean(consistencies)


class SemanticDriftAlignment:
    """Evaluate how well model captures real semantic drift patterns."""
    
    def __init__(self, model, device=None):
        """Initialize drift alignment metric."""
        self.model = model
        self.device = device or torch.device("cpu")
    
    def compute(
        self,
        text_pairs: List[Tuple[str, str, float, float, float]],
    ) -> float:
        """Compute drift alignment score.
        
        Args:
            text_pairs: List of (text1, text2, timestamp1, timestamp2, expected_drift)
                where expected_drift is ground truth semantic change
        
        Returns:
            Alignment score (higher is better)
        """
        self.model.eval()
        predicted_drifts = []
        expected_drifts = []
        
        with torch.no_grad():
            for text1, text2, ts1, ts2, expected_drift in text_pairs:
                # Tokenize texts
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model.config.encoder_name)
                
                inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
                inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
                
                inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
                inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}
                
                # Get embeddings at respective timestamps
                ts1_tensor = torch.tensor([ts1], device=self.device)
                ts2_tensor = torch.tensor([ts2], device=self.device)
                
                emb1, _ = self.model(inputs1["input_ids"], inputs1["attention_mask"], ts1_tensor)
                emb2, _ = self.model(inputs2["input_ids"], inputs2["attention_mask"], ts2_tensor)
                
                # Compute drift (semantic distance)
                emb1_norm = F.normalize(emb1, p=2, dim=1)
                emb2_norm = F.normalize(emb2, p=2, dim=1)
                similarity = torch.sum(emb1_norm * emb2_norm).item()
                predicted_drift = (1 - similarity) / 2  # Scale to [0, 1]
                
                predicted_drifts.append(predicted_drift)
                expected_drifts.append(expected_drift)
        
        # Compute correlation between predicted and expected drift
        if len(predicted_drifts) > 1:
            correlation, _ = spearmanr(predicted_drifts, expected_drifts)
            mse = mean_squared_error(expected_drifts, predicted_drifts)
            
            # Combine metrics (correlation for direction, MSE for magnitude)
            alignment_score = correlation * (1 / (1 + mse))
        else:
            alignment_score = 0.0
        
        return alignment_score


class EventRobustnessMetric:
    """Test model's handling of discontinuous events."""
    
    def __init__(self, model, device=None):
        """Initialize event robustness metric."""
        self.model = model
        self.device = device or torch.device("cpu")
        
        # Define test events with known impacts
        self.test_events = [
            {
                "name": "COVID-19",
                "before_text": "Working from the office is standard practice",
                "after_text": "Remote work has become the new normal",
                "event_date": datetime(2020, 3, 15),
                "expected_change": 0.8
            },
            {
                "name": "ChatGPT",
                "before_text": "AI is a specialized tool for researchers",
                "after_text": "AI assistants are part of everyday life",
                "event_date": datetime(2022, 11, 30),
                "expected_change": 0.7
            }
        ]
    
    def compute(self) -> float:
        """Compute event robustness score.
        
        Returns:
            Robustness score (0-1, higher is better)
        """
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for event in self.test_events:
                # Test embedding change across event boundary
                before_ts = (event["event_date"] - timedelta(days=7)).timestamp()
                after_ts = (event["event_date"] + timedelta(days=7)).timestamp()
                
                # Tokenize
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model.config.encoder_name)
                
                # Embed before text at before timestamp
                inputs_before = tokenizer(event["before_text"], return_tensors="pt", 
                                         padding=True, truncation=True)
                inputs_before = {k: v.to(self.device) for k, v in inputs_before.items()}
                ts_before = torch.tensor([before_ts], device=self.device)
                emb_before, _ = self.model(
                    inputs_before["input_ids"], 
                    inputs_before["attention_mask"],
                    ts_before
                )
                
                # Embed after text at after timestamp
                inputs_after = tokenizer(event["after_text"], return_tensors="pt",
                                        padding=True, truncation=True)
                inputs_after = {k: v.to(self.device) for k, v in inputs_after.items()}
                ts_after = torch.tensor([after_ts], device=self.device)
                emb_after, _ = self.model(
                    inputs_after["input_ids"],
                    inputs_after["attention_mask"], 
                    ts_after
                )
                
                # Compute semantic change
                emb_before_norm = F.normalize(emb_before, p=2, dim=1)
                emb_after_norm = F.normalize(emb_after, p=2, dim=1)
                similarity = torch.sum(emb_before_norm * emb_after_norm).item()
                actual_change = (1 - similarity) / 2
                
                # Compare to expected change
                error = abs(actual_change - event["expected_change"])
                score = 1.0 - min(error, 1.0)
                scores.append(score)
                
                logger.info(f"Event {event['name']}: expected={event['expected_change']:.2f}, "
                          f"actual={actual_change:.2f}, score={score:.2f}")
        
        return np.mean(scores) if scores else 0.0


class FutureGeneralizationMetric:
    """Test model's ability to generalize to future timestamps."""
    
    def __init__(self, model, device=None):
        """Initialize future generalization metric."""
        self.model = model
        self.device = device or torch.device("cpu")
    
    def compute(
        self,
        test_data: List[Dict],
        future_timestamp: float
    ) -> float:
        """Compute future generalization score.
        
        Args:
            test_data: Test examples with text and similarity labels
            future_timestamp: Future timestamp to test (beyond training data)
        
        Returns:
            Generalization score
        """
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for item in test_data:
                text1 = item["text1"]
                text2 = item["text2"]
                similarity = item["similarity"]
                
                # Tokenize
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model.config.encoder_name)
                
                inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
                inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
                
                inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
                inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}
                
                # Embed at future timestamp
                ts_future = torch.tensor([future_timestamp], device=self.device)
                
                emb1, _ = self.model(inputs1["input_ids"], inputs1["attention_mask"], ts_future)
                emb2, _ = self.model(inputs2["input_ids"], inputs2["attention_mask"], ts_future)
                
                # Compute similarity
                emb1_norm = F.normalize(emb1, p=2, dim=1)
                emb2_norm = F.normalize(emb2, p=2, dim=1)
                pred_sim = torch.sum(emb1_norm * emb2_norm).item()
                
                predictions.append(pred_sim)
                targets.append(similarity)
        
        # Compute correlation
        if len(predictions) > 1:
            correlation, _ = spearmanr(predictions, targets)
            return max(0, correlation)  # Clip negative correlations to 0
        return 0.0
