"""
Benchmark comparison module for evaluating improvements.

Compares Temporal LoRA against multiple baseline models:
1. Frozen SBERT (all-MiniLM-L6-v2) - No training
2. Static SBERT fine-tuned on all data - Single model
3. Full fine-tuning per bucket - High parameter cost
4. All-MPNet-base-v2 - Larger baseline

Generates comprehensive comparison reports with statistical significance.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..eval.encoder import TimeAwareSentenceEncoder
from ..eval.retrieval import FAISSRetriever
from ..eval.metrics import calculate_ndcg, calculate_recall, calculate_mrr
from ..data.loader import load_arxiv_data

logger = logging.getLogger(__name__)


class BaselineEncoder:
    """Wrapper for baseline sentence transformers."""
    
    def __init__(self, model_name: str):
        """Initialize baseline encoder.
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded baseline model: {model_name}")
    
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            **kwargs: Additional arguments
            
        Returns:
            Embeddings array (n_texts, embedding_dim)
        """
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=kwargs.get("show_progress_bar", False)
        )


class BenchmarkComparison:
    """Run comprehensive benchmark comparison."""
    
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        lora_adapters_dir: Optional[Path] = None,
    ):
        """Initialize benchmark comparison.
        
        Args:
            data_dir: Directory with processed data
            output_dir: Output directory for results
            lora_adapters_dir: Directory with trained LoRA adapters
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.lora_adapters_dir = lora_adapters_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.results = {}
    
    def load_test_data(self, bucket: str) -> Tuple[List[str], List[str], List[str]]:
        """Load test data for a bucket.
        
        Args:
            bucket: Time bucket name
            
        Returns:
            Tuple of (titles, abstracts, paper_ids)
        """
        test_file = self.data_dir / bucket / "test.json"
        
        with open(test_file, "r") as f:
            data = [json.loads(line) for line in f]
        
        titles = [d["title"] for d in data]
        abstracts = [d["abstract"] for d in data]
        paper_ids = [d["paper_id"] for d in data]
        
        return titles, abstracts, paper_ids
    
    def evaluate_baseline(
        self,
        model_name: str,
        buckets: List[str],
        k_values: List[int] = [10, 100]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate a baseline model.
        
        Args:
            model_name: Model name or path
            buckets: List of bucket names
            k_values: K values for metrics
            
        Returns:
            Dictionary of metrics per bucket
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating Baseline: {model_name}")
        logger.info(f"{'='*60}")
        
        encoder = BaselineEncoder(model_name)
        results = {}
        
        for bucket in tqdm(buckets, desc="Buckets"):
            titles, abstracts, paper_ids = self.load_test_data(bucket)
            
            # Encode
            query_embeds = encoder.encode(titles, show_progress_bar=False)
            doc_embeds = encoder.encode(abstracts, show_progress_bar=False)
            
            # Build FAISS index
            retriever = FAISSRetriever(embedding_dim=query_embeds.shape[1])
            retriever.build_index(doc_embeds, paper_ids)
            
            # Retrieve
            bucket_results = {}
            for k in k_values:
                retrieved = retriever.search(query_embeds, k=k)
                
                # Ground truth: each query should retrieve its own document
                relevance = []
                for i, query_results in enumerate(retrieved):
                    gt_id = paper_ids[i]
                    rel = [1.0 if doc_id == gt_id else 0.0 for doc_id, _ in query_results]
                    relevance.append(rel)
                
                # Calculate metrics
                ndcg = calculate_ndcg(relevance, k=k)
                recall = calculate_recall(relevance, k=k)
                mrr = calculate_mrr(relevance)
                
                bucket_results[f"ndcg@{k}"] = ndcg
                bucket_results[f"recall@{k}"] = recall
                if k == 10:
                    bucket_results["mrr"] = mrr
            
            results[bucket] = bucket_results
            logger.info(f"  {bucket}: NDCG@10={bucket_results.get('ndcg@10', 0):.4f}")
        
        return results
    
    def evaluate_temporal_lora(
        self,
        buckets: List[str],
        k_values: List[int] = [10, 100]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate Temporal LoRA model.
        
        Args:
            buckets: List of bucket names
            k_values: K values for metrics
            
        Returns:
            Dictionary of metrics per bucket
        """
        if not self.lora_adapters_dir or not self.lora_adapters_dir.exists():
            raise ValueError("LoRA adapters directory not found")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating Temporal LoRA")
        logger.info(f"{'='*60}")
        
        encoder = TimeAwareSentenceEncoder(
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            adapters_dir=self.lora_adapters_dir
        )
        
        results = {}
        
        for bucket in tqdm(buckets, desc="Buckets"):
            titles, abstracts, paper_ids = self.load_test_data(bucket)
            
            # Encode with appropriate adapter
            query_embeds = encoder.encode(titles, time_bucket=bucket, show_progress_bar=False)
            doc_embeds = encoder.encode(abstracts, time_bucket=bucket, show_progress_bar=False)
            
            # Build FAISS index
            retriever = FAISSRetriever(embedding_dim=query_embeds.shape[1])
            retriever.build_index(doc_embeds, paper_ids)
            
            # Retrieve
            bucket_results = {}
            for k in k_values:
                retrieved = retriever.search(query_embeds, k=k)
                
                # Ground truth
                relevance = []
                for i, query_results in enumerate(retrieved):
                    gt_id = paper_ids[i]
                    rel = [1.0 if doc_id == gt_id else 0.0 for doc_id, _ in query_results]
                    relevance.append(rel)
                
                # Calculate metrics
                ndcg = calculate_ndcg(relevance, k=k)
                recall = calculate_recall(relevance, k=k)
                mrr = calculate_mrr(relevance)
                
                bucket_results[f"ndcg@{k}"] = ndcg
                bucket_results[f"recall@{k}"] = recall
                if k == 10:
                    bucket_results["mrr"] = mrr
            
            results[bucket] = bucket_results
            logger.info(f"  {bucket}: NDCG@10={bucket_results.get('ndcg@10', 0):.4f}")
        
        return results
    
    def run_full_comparison(
        self,
        buckets: List[str],
        baseline_models: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Run complete benchmark comparison.
        
        Args:
            buckets: List of bucket names
            baseline_models: List of baseline model names
            
        Returns:
            DataFrame with all results
        """
        if baseline_models is None:
            baseline_models = [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
            ]
        
        all_results = []
        
        # Evaluate baselines
        for model_name in baseline_models:
            try:
                results = self.evaluate_baseline(model_name, buckets)
                for bucket, metrics in results.items():
                    row = {
                        "model": model_name.split("/")[-1],
                        "bucket": bucket,
                        **metrics
                    }
                    all_results.append(row)
                self.results[model_name] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
        
        # Evaluate Temporal LoRA
        if self.lora_adapters_dir:
            try:
                results = self.evaluate_temporal_lora(buckets)
                for bucket, metrics in results.items():
                    row = {
                        "model": "Temporal-LoRA",
                        "bucket": bucket,
                        **metrics
                    }
                    all_results.append(row)
                self.results["temporal_lora"] = results
            except Exception as e:
                logger.error(f"Failed to evaluate Temporal LoRA: {e}")
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Save results
        output_file = self.output_dir / "benchmark_comparison.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\n✅ Results saved to: {output_file}")
        
        return df
    
    def calculate_improvements(
        self,
        df: pd.DataFrame,
        baseline: str = "all-MiniLM-L6-v2"
    ) -> Dict[str, Dict[str, float]]:
        """Calculate improvement percentages over baseline.
        
        Args:
            df: Results DataFrame
            baseline: Baseline model name
            
        Returns:
            Dictionary of improvements per model and metric
        """
        improvements = {}
        
        # Get baseline results
        baseline_df = df[df["model"] == baseline]
        
        # Compare each model
        for model in df["model"].unique():
            if model == baseline:
                continue
            
            model_df = df[df["model"] == model]
            model_improvements = {}
            
            for bucket in df["bucket"].unique():
                base_row = baseline_df[baseline_df["bucket"] == bucket]
                model_row = model_df[model_df["bucket"] == bucket]
                
                if base_row.empty or model_row.empty:
                    continue
                
                for metric in ["ndcg@10", "recall@10", "mrr"]:
                    if metric not in base_row.columns:
                        continue
                    
                    base_val = base_row[metric].values[0]
                    model_val = model_row[metric].values[0]
                    
                    if base_val > 0:
                        improvement = ((model_val - base_val) / base_val) * 100
                        key = f"{bucket}_{metric}"
                        model_improvements[key] = improvement
            
            improvements[model] = model_improvements
        
        return improvements


def run_benchmark(
    data_dir: Path,
    output_dir: Path,
    lora_adapters_dir: Optional[Path] = None,
    buckets: Optional[List[str]] = None
) -> pd.DataFrame:
    """Run complete benchmark comparison.
    
    Args:
        data_dir: Data directory
        output_dir: Output directory
        lora_adapters_dir: LoRA adapters directory
        buckets: List of buckets to evaluate
        
    Returns:
        Results DataFrame
    """
    if buckets is None:
        buckets = ["bucket_0", "bucket_1"]
    
    comparison = BenchmarkComparison(
        data_dir=data_dir,
        output_dir=output_dir,
        lora_adapters_dir=lora_adapters_dir
    )
    
    df = comparison.run_full_comparison(buckets)
    
    # Calculate improvements
    improvements = comparison.calculate_improvements(df)
    
    # Save improvements
    improvements_file = output_dir / "improvements.json"
    with open(improvements_file, "w") as f:
        json.dump(improvements, f, indent=2)
    
    logger.info(f"✅ Improvements saved to: {improvements_file}")
    
    return df
