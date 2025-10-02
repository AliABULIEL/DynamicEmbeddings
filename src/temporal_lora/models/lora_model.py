"""LoRA model wrapper with auto-detection of attention modules."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer
from peft import get_peft_model, LoraConfig, TaskType

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Common attention module naming patterns
ATTENTION_PATTERNS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "query",
    "key",
    "value",
    "Wq",
    "Wk",
    "Wv",
]


def print_module_tree(model: torch.nn.Module, max_depth: int = 3) -> None:
    """Print module tree for debugging.
    
    Args:
        model: Model to inspect.
        max_depth: Maximum depth to print.
    """
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE (for debugging)")
    print("=" * 80)
    
    def _print_tree(module: torch.nn.Module, prefix: str = "", depth: int = 0) -> None:
        if depth >= max_depth:
            return
        
        for name, child in module.named_children():
            print(f"{prefix}├── {name}: {child.__class__.__name__}")
            _print_tree(child, prefix + "│   ", depth + 1)
    
    _print_tree(model)
    print("=" * 80 + "\n")


def find_attention_modules(model: torch.nn.Module) -> List[str]:
    """Auto-detect attention Q/K/V module names.
    
    Args:
        model: Model to inspect.
        
    Returns:
        List of detected module names.
    """
    attention_modules = []
    
    for name, module in model.named_modules():
        # Check if module name contains any attention pattern
        for pattern in ATTENTION_PATTERNS:
            if pattern in name and isinstance(module, torch.nn.Linear):
                # Get the base name (last component)
                base_name = name.split(".")[-1]
                if base_name not in attention_modules:
                    attention_modules.append(base_name)
                break
    
    return attention_modules


def create_lora_model(
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> Tuple[SentenceTransformer, List[str]]:
    """Create a SentenceTransformer with LoRA adapters.
    
    Args:
        base_model_name: HuggingFace model identifier.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout rate.
        target_modules: Specific modules to target. If None, auto-detect.
        
    Returns:
        Tuple of (model with LoRA, list of target modules).
        
    Raises:
        ValueError: If no attention modules found.
    """
    logger.info(f"Loading base model: {base_model_name}")
    base_model = SentenceTransformer(base_model_name)
    
    # Get the underlying transformer model
    # SentenceTransformer wraps a HF model in base_model[0]
    transformer = base_model[0].auto_model
    
    # Auto-detect attention modules if not provided
    if target_modules is None:
        logger.info("Auto-detecting attention modules...")
        detected_modules = find_attention_modules(transformer)
        
        if not detected_modules:
            print_module_tree(transformer, max_depth=4)
            raise ValueError(
                "❌ No attention modules (Q/K/V) detected!\n\n"
                "The model architecture above doesn't contain recognized attention patterns.\n"
                f"Expected patterns: {ATTENTION_PATTERNS}\n\n"
                "REMEDIATION:\n"
                "1. Inspect the module tree above to identify attention projection layers\n"
                "2. Update ATTENTION_PATTERNS in lora_model.py, or\n"
                "3. Manually specify target_modules in model.yaml config\n\n"
                "Example: target_modules: ['q_lin', 'k_lin', 'v_lin']"
            )
        
        target_modules = detected_modules
        logger.info(f"✓ Detected attention modules: {target_modules}")
    else:
        logger.info(f"Using specified target modules: {target_modules}")
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    
    # Apply LoRA to the transformer
    logger.info(f"Applying LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")
    transformer = get_peft_model(transformer, lora_config)
    
    # Replace the transformer in the SentenceTransformer
    base_model[0].auto_model = transformer
    
    return base_model, target_modules


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count trainable and total parameters.
    
    Args:
        model: Model to analyze.
        
    Returns:
        Tuple of (trainable_params, total_params).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def assert_trainable_ratio(
    model: torch.nn.Module, max_ratio: float = 0.01, raise_error: bool = True
) -> float:
    """Assert that trainable parameter ratio is below threshold.
    
    Args:
        model: Model to check.
        max_ratio: Maximum allowed ratio of trainable/total params.
        raise_error: Whether to raise error if ratio exceeds threshold.
        
    Returns:
        Actual trainable ratio.
        
    Raises:
        ValueError: If ratio exceeds threshold and raise_error=True.
    """
    trainable, total = count_parameters(model)
    ratio = trainable / total if total > 0 else 0.0
    
    logger.info(
        f"Parameter count: {trainable:,} trainable / {total:,} total "
        f"({ratio:.2%})"
    )
    
    if ratio > max_ratio:
        msg = (
            f"❌ Trainable ratio {ratio:.2%} exceeds maximum {max_ratio:.2%}!\n"
            f"This likely means LoRA is not properly configured.\n"
            f"Expected <1% for efficient training."
        )
        if raise_error:
            raise ValueError(msg)
        else:
            logger.warning(msg)
    else:
        logger.info(f"✓ Trainable ratio {ratio:.2%} is within bounds (<{max_ratio:.2%})")
    
    return ratio


def save_lora_adapter(model: SentenceTransformer, output_dir: Path) -> None:
    """Save only the LoRA adapter weights.
    
    Args:
        model: SentenceTransformer with LoRA adapters.
        output_dir: Directory to save adapter.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the PEFT model (wrapped transformer)
    transformer = model[0].auto_model
    
    # Save adapter
    transformer.save_pretrained(output_dir)
    logger.info(f"✓ Saved LoRA adapter to: {output_dir}")


def load_lora_adapter(
    base_model_name: str, adapter_dir: Path
) -> SentenceTransformer:
    """Load base model and attach saved LoRA adapter.
    
    Args:
        base_model_name: HuggingFace model identifier.
        adapter_dir: Directory containing saved adapter.
        
    Returns:
        SentenceTransformer with loaded LoRA adapter.
    """
    from peft import PeftModel
    
    logger.info(f"Loading base model: {base_model_name}")
    base_model = SentenceTransformer(base_model_name)
    
    # Get the underlying transformer
    transformer = base_model[0].auto_model
    
    # Load adapter
    logger.info(f"Loading LoRA adapter from: {adapter_dir}")
    transformer = PeftModel.from_pretrained(transformer, adapter_dir)
    
    # Replace transformer
    base_model[0].auto_model = transformer
    
    logger.info("✓ Loaded model with LoRA adapter")
    return base_model
