"""Tests for trainer modes (LoRA, full fine-tune, sequential fine-tune)."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from sentence_transformers import InputExample

from temporal_lora.train.trainer import UnifiedTrainer, log_model_info, check_cuda_oom_risk


class TestTrainerModes:
    """Test different training modes."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup
        if temp_path.exists():
            shutil.rmtree(temp_path)
    
    @pytest.fixture
    def tiny_train_examples(self):
        """Create tiny synthetic training data."""
        examples = [
            InputExample(texts=["This is text 1", "This is similar to text 1"], label=1.0),
            InputExample(texts=["Another example here", "More text for training"], label=1.0),
            InputExample(texts=["Third training pair", "Yet another text sample"], label=1.0),
        ]
        return examples
    
    def test_lora_mode_creates_adapters(self, temp_dir, tiny_train_examples):
        """Test that LoRA mode creates adapter files."""
        trainer = UnifiedTrainer(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            mode="lora",
            lora_r=8,
            epochs=1,
            batch_size=2,
            seed=42,
        )
        
        output_dir = temp_dir / "lora_test"
        metrics, model = trainer.train_bucket(
            bucket_name="test_bucket",
            train_examples=tiny_train_examples,
            output_dir=output_dir,
        )
        
        # Check that adapter files exist
        assert output_dir.exists()
        assert (output_dir / "adapter_config.json").exists()
        assert (output_dir / "adapter_model.safetensors").exists() or \
               (output_dir / "adapter_model.bin").exists()
        
        # Check metrics
        assert metrics["mode"] == "lora"
        assert metrics["train_examples"] == len(tiny_train_examples)
    
    def test_full_ft_mode_creates_checkpoint(self, temp_dir, tiny_train_examples):
        """Test that full_ft mode creates full model checkpoint."""
        trainer = UnifiedTrainer(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            mode="full_ft",
            epochs=1,
            batch_size=2,
            seed=42,
        )
        
        output_dir = temp_dir / "full_ft_test"
        metrics, model = trainer.train_bucket(
            bucket_name="test_bucket",
            train_examples=tiny_train_examples,
            output_dir=output_dir,
        )
        
        # Check that full model files exist
        assert output_dir.exists()
        # Should have sentence-transformers model files
        assert any(output_dir.glob("*.safetensors")) or \
               any(output_dir.glob("*.bin")) or \
               any(output_dir.glob("pytorch_model.bin"))
        
        # Check metrics
        assert metrics["mode"] == "full_ft"
        assert metrics["train_examples"] == len(tiny_train_examples)
    
    def test_seq_ft_mode_continues_training(self, temp_dir, tiny_train_examples):
        """Test that seq_ft mode can continue from existing model."""
        trainer = UnifiedTrainer(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            mode="seq_ft",
            epochs=1,
            batch_size=2,
            seed=42,
        )
        
        # Train first bucket
        output_dir1 = temp_dir / "seq_ft_step1"
        metrics1, model1 = trainer.train_bucket(
            bucket_name="bucket1",
            train_examples=tiny_train_examples,
            output_dir=output_dir1,
            existing_model=None,  # Start fresh
        )
        
        # Train second bucket continuing from first
        output_dir2 = temp_dir / "seq_ft_step2"
        metrics2, model2 = trainer.train_bucket(
            bucket_name="bucket2",
            train_examples=tiny_train_examples,
            output_dir=output_dir2,
            existing_model=model1,  # Continue training
        )
        
        # Both should succeed
        assert output_dir1.exists()
        assert output_dir2.exists()
        assert metrics1["mode"] == "seq_ft"
        assert metrics2["mode"] == "seq_ft"
        
        # Model 2 should be the same object as model 1 (continued training)
        assert model2 is model1
    
    def test_lora_mode_has_low_trainable_ratio(self, tiny_train_examples):
        """Test that LoRA mode has <1% trainable parameters."""
        trainer = UnifiedTrainer(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            mode="lora",
            lora_r=16,
            epochs=1,
            batch_size=2,
            seed=42,
        )
        
        model = trainer.create_model()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_ratio = trainable_params / total_params if total_params > 0 else 0
        
        # LoRA should have <1% trainable parameters
        assert trainable_ratio < 0.01, f"LoRA trainable ratio too high: {trainable_ratio:.4%}"
    
    def test_full_ft_mode_has_all_parameters_trainable(self, tiny_train_examples):
        """Test that full_ft mode has all parameters trainable."""
        trainer = UnifiedTrainer(
            base_model_name="sentence-transformers/all-MiniLM-L6-v2",
            mode="full_ft",
            epochs=1,
            batch_size=2,
            seed=42,
        )
        
        model = trainer.create_model()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_ratio = trainable_params / total_params if total_params > 0 else 0
        
        # Full FT should have ~100% trainable parameters
        assert trainable_ratio > 0.99, f"Full FT trainable ratio too low: {trainable_ratio:.4%}"
    
    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown training mode"):
            trainer = UnifiedTrainer(
                base_model_name="sentence-transformers/all-MiniLM-L6-v2",
                mode="invalid_mode",
                epochs=1,
                batch_size=2,
                seed=42,
            )
            trainer.create_model()


class TestTrainerLogging:
    """Test trainer logging and safety features."""
    
    def test_log_model_info_runs_without_error(self):
        """Test that log_model_info doesn't crash."""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Should not raise
        log_model_info(model, "lora")
        log_model_info(model, "full_ft")
        log_model_info(model, "seq_ft")
    
    def test_check_cuda_oom_risk_runs_without_error(self):
        """Test that OOM check doesn't crash."""
        # Should not raise
        check_cuda_oom_risk(batch_size=32, model_size_mb=100, mode="lora")
        check_cuda_oom_risk(batch_size=32, model_size_mb=100, mode="full_ft")
        check_cuda_oom_risk(batch_size=32, model_size_mb=100, mode="seq_ft")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
