"""Encoder utilities for caching embeddings."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..models.lora_model import load_lora_adapter
from ..utils.logging import get_logger

logger = get_logger(__name__)


def encode_batch(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode texts in batches.
    
    Args:
        model: SentenceTransformer model.
        texts: List of texts to encode.
        batch_size: Batch size for encoding.
        show_progress: Show progress bar.
        
    Returns:
        Numpy array of embeddings (n_texts, embedding_dim).
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # For cosine similarity via dot product
    )
    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    ids: List[str],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Save embeddings and IDs to disk.
    
    Args:
        embeddings: Embeddings array (n, dim).
        ids: List of IDs corresponding to embeddings.
        output_dir: Output directory.
        
    Returns:
        Tuple of (embeddings_path, ids_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    
    # Save IDs as JSONL
    ids_path = output_dir / "ids.jsonl"
    with open(ids_path, "w") as f:
        for id_val in ids:
            f.write(json.dumps({"id": id_val}) + "\n")
    
    logger.info(f"Saved {len(ids)} embeddings to {output_dir}")
    return embeddings_path, ids_path


def load_embeddings(embeddings_dir: Path) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings and IDs from disk.
    
    Args:
        embeddings_dir: Directory containing embeddings.npy and ids file.
        
    Returns:
        Tuple of (embeddings, ids).
    """
    embeddings_path = embeddings_dir / "embeddings.npy"
    
    # Try jsonl format first, then fall back to txt
    ids_jsonl = embeddings_dir / "ids.jsonl"
    ids_txt = embeddings_dir / "ids.txt"
    
    embeddings = np.load(embeddings_path)
    
    ids = []
    if ids_jsonl.exists():
        # JSONL format
        with open(ids_jsonl, "r") as f:
            for line in f:
                ids.append(json.loads(line)["id"])
    elif ids_txt.exists():
        # Simple text format (one ID per line)
        with open(ids_txt, "r") as f:
            ids = [line.strip() for line in f if line.strip()]
    else:
        raise FileNotFoundError(f"No ids file found in {embeddings_dir}")
    
    return embeddings, ids


def encode_and_cache_bucket(
    bucket_name: str,
    bucket_data_path: Path,
    adapter_dir: Optional[Path],
    base_model_name: str,
    output_dir: Path,
    batch_size: int = 32,
    use_lora: bool = True,
) -> Dict[str, Path]:
    """Encode all data for a bucket and cache embeddings.
    
    Args:
        bucket_name: Name of the bucket.
        bucket_data_path: Path to bucket parquet files.
        adapter_dir: Path to LoRA adapter (if use_lora=True).
        base_model_name: Base model name.
        output_dir: Output directory for cached embeddings.
        batch_size: Batch size for encoding.
        use_lora: Whether to use LoRA adapter.
        
    Returns:
        Dictionary mapping split -> output directory path.
    """
    import pandas as pd
    
    logger.info(f"Encoding bucket: {bucket_name} (use_lora={use_lora})")
    
    # Load model
    if use_lora and adapter_dir:
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"Adapter directory not found: {adapter_dir}. "
                f"Run train-adapters first."
            )
        logger.info(f"Loading LoRA adapter from: {adapter_dir}")
        model = load_lora_adapter(base_model_name, adapter_dir)
    else:
        logger.info(f"Loading base model: {base_model_name}")
        model = SentenceTransformer(base_model_name)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Using device: {device}")
    
    # Process each split
    output_paths = {}
    for split in ["train", "val", "test"]:
        split_path = bucket_data_path / f"{split}.parquet"
        if not split_path.exists():
            logger.warning(f"Split not found: {split_path}, skipping")
            continue
        
        # Load data
        df = pd.read_parquet(split_path)
        logger.info(f"Encoding {len(df)} samples for {bucket_name}/{split}")
        
        # Combine title + abstract for encoding
        texts = (df["text_a"] + " " + df["text_b"]).tolist()
        ids = df["paper_id"].tolist()
        
        # Encode
        embeddings = encode_batch(model, texts, batch_size=batch_size)
        
        # Save
        split_output_dir = output_dir / bucket_name / split
        save_embeddings(embeddings, ids, split_output_dir)
        output_paths[split] = split_output_dir
    
    return output_paths


class TimeAwareSentenceEncoder:
    """Encoder that uses different LoRA adapters per time bucket."""
    
    def __init__(self, base_model: str, adapters_dir: Path):
        """Initialize time-aware encoder.
        
        Args:
            base_model: Base model name
            adapters_dir: Directory containing LoRA adapters per bucket
        """
        self.base_model_name = base_model
        self.adapters_dir = Path(adapters_dir)
        self.models = {}
        
        # Load all available adapters
        if self.adapters_dir.exists():
            for adapter_path in self.adapters_dir.iterdir():
                if adapter_path.is_dir():
                    bucket_name = adapter_path.name
                    logger.info(f"Loading adapter for bucket: {bucket_name}")
                    self.models[bucket_name] = load_lora_adapter(
                        base_model, adapter_path
                    )
        
        # Also load base model as fallback
        self.base_model = SentenceTransformer(base_model)
        
        logger.info(f"Loaded {len(self.models)} adapters")
    
    def encode(
        self,
        texts: List[str],
        time_bucket: Optional[str] = None,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode texts with appropriate time-aware model.
        
        Args:
            texts: Texts to encode
            time_bucket: Time bucket to use (None = base model)
            batch_size: Batch size
            show_progress_bar: Show progress
            
        Returns:
            Embeddings array
        """
        # Select model
        if time_bucket and time_bucket in self.models:
            model = self.models[time_bucket]
        else:
            model = self.base_model
        
        # Encode
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        return embeddings
