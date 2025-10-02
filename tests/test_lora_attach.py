"""Smoke tests for LoRA model attachment."""

import pytest
import torch


def test_find_attention_modules():
    """Test that attention modules are auto-detected."""
    from temporal_lora.models.lora_model import find_attention_modules
    from sentence_transformers import SentenceTransformer
    
    # Load base model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    transformer = model[0].auto_model
    
    # Find attention modules
    attention_modules = find_attention_modules(transformer)
    
    # Should find at least 3 modules (Q, K, V)
    assert len(attention_modules) >= 3, f"Expected ≥3 attention modules, found {len(attention_modules)}"
    
    # Common patterns
    expected_patterns = ["query", "key", "value", "q_proj", "k_proj", "v_proj"]
    found_any = any(pattern in module for module in attention_modules for pattern in expected_patterns)
    assert found_any, f"No recognized attention patterns found in: {attention_modules}"
    
    print(f"✓ Detected {len(attention_modules)} attention modules: {attention_modules}")


def test_create_lora_model():
    """Test LoRA model creation."""
    from temporal_lora.models.lora_model import create_lora_model
    
    # Create model with small LoRA rank for speed
    model, target_modules = create_lora_model(
        base_model_name="sentence-transformers/all-MiniLM-L6-v2",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    
    # Check target modules detected
    assert len(target_modules) >= 3, f"Expected ≥3 target modules, got {len(target_modules)}"
    
    # Check model can encode
    embeddings = model.encode(["test sentence"], convert_to_tensor=True)
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] > 0  # Has embedding dimension
    
    print(f"✓ Model created with target modules: {target_modules}")
    print(f"✓ Embedding shape: {embeddings.shape}")


def test_trainable_parameter_ratio():
    """Test that trainable parameter ratio is < 1%."""
    from temporal_lora.models.lora_model import create_lora_model, assert_trainable_ratio
    
    # Create model
    model, _ = create_lora_model(
        base_model_name="sentence-transformers/all-MiniLM-L6-v2",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    
    # Check ratio
    ratio = assert_trainable_ratio(model, max_ratio=0.01, raise_error=True)
    
    # Should be well below 1%
    assert ratio < 0.01, f"Trainable ratio {ratio:.4%} exceeds 1%"
    assert ratio > 0, "Trainable ratio should be > 0"
    
    print(f"✓ Trainable parameter ratio: {ratio:.4%}")


def test_save_and_load_adapter(tmp_path):
    """Test saving and loading LoRA adapter."""
    from temporal_lora.models.lora_model import (
        create_lora_model,
        save_lora_adapter,
        load_lora_adapter,
    )
    
    # Create model
    model, _ = create_lora_model(
        base_model_name="sentence-transformers/all-MiniLM-L6-v2",
        lora_r=8,
    )
    
    # Encode with original model
    test_text = "This is a test sentence."
    embedding_before = model.encode([test_text], convert_to_tensor=True)
    
    # Save adapter
    adapter_dir = tmp_path / "test_adapter"
    save_lora_adapter(model, adapter_dir)
    
    # Check files saved
    assert adapter_dir.exists()
    assert (adapter_dir / "adapter_config.json").exists()
    assert (adapter_dir / "adapter_model.bin").exists() or (
        adapter_dir / "adapter_model.safetensors"
    ).exists()
    
    # Load adapter into fresh model
    loaded_model = load_lora_adapter(
        "sentence-transformers/all-MiniLM-L6-v2",
        adapter_dir,
    )
    
    # Encode with loaded model
    embedding_after = loaded_model.encode([test_text], convert_to_tensor=True)
    
    # Embeddings should be identical (or very close due to numerical precision)
    diff = torch.abs(embedding_before - embedding_after).max().item()
    assert diff < 1e-5, f"Embeddings differ by {diff}"
    
    print(f"✓ Adapter saved and loaded successfully")
    print(f"✓ Embedding difference: {diff:.2e}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_on_gpu():
    """Test that model can be moved to GPU."""
    from temporal_lora.models.lora_model import create_lora_model
    
    model, _ = create_lora_model(
        base_model_name="sentence-transformers/all-MiniLM-L6-v2",
        lora_r=8,
    )
    
    # Move to GPU
    model.to("cuda")
    
    # Encode
    embeddings = model.encode(["test"], convert_to_tensor=True)
    
    # Should be on GPU
    assert embeddings.device.type == "cuda"
    
    print(f"✓ Model successfully running on GPU")
