"""Test CLI argument parsing and config validation."""

import pytest
from pathlib import Path
from tide_lite.utils.config import TIDEConfig, load_config


class TestConfig:
    """Test configuration loading and validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TIDEConfig()
        
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.hidden_dim == 384
        assert config.time_dims == 32
        assert config.max_seq_len == 128
        assert config.batch_size == 32
        assert config.freeze_encoder == True
    
    def test_config_override(self):
        """Test configuration override."""
        config = TIDEConfig(
            batch_size=64,
            learning_rate=1e-3,
            num_epochs=10,
        )
        
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3
        assert config.num_epochs == 10
    
    def test_config_validation(self):
        """Test configuration validation."""
        # These should work
        config = TIDEConfig(
            mlp_dropout=0.5,
            consistency_weight=0.2,
        )
        assert config.mlp_dropout == 0.5
        assert config.consistency_weight == 0.2
        
        # Test pooling strategy validation
        config = TIDEConfig(pooling_strategy="mean")
        assert config.pooling_strategy == "mean"
        
        config = TIDEConfig(pooling_strategy="cls")
        assert config.pooling_strategy == "cls"


class TestCLIParsing:
    """Test CLI argument parsing."""
    
    def test_parse_train_args(self):
        """Test training argument parsing."""
        from tide_lite.cli.train import setup_argument_parser
        
        parser = setup_argument_parser()
        
        # Test default dry-run
        args = parser.parse_args([])
        assert args.run == False  # Default is dry-run
        
        # Test with --run flag
        args = parser.parse_args(["--run"])
        assert args.run == True
        
        # Test config override
        args = parser.parse_args([
            "--batch-size", "64",
            "--learning-rate", "1e-3",
            "--num-epochs", "5",
        ])
        assert args.batch_size == 64
        assert args.learning_rate == 1e-3
        assert args.num_epochs == 5
    
    def test_parse_eval_args(self):
        """Test evaluation argument parsing."""
        from tide_lite.cli.eval_stsb import setup_argument_parser
        
        parser = setup_argument_parser()
        
        # Test required model argument
        args = parser.parse_args(["--model", "test_model.pt"])
        assert args.model == "test_model.pt"
        assert args.run == False  # Default is dry-run
        
        # Test model type
        args = parser.parse_args([
            "--model", "minilm",
            "--type", "baseline",
        ])
        assert args.model == "minilm"
        assert args.type == "baseline"
    
    def test_orchestrator_subcommands(self):
        """Test orchestrator subcommand parsing."""
        from tide_lite.cli.tide import setup_parser
        
        parser = setup_parser()
        
        # Test train subcommand
        args = parser.parse_args(["train", "--output-dir", "results"])
        assert args.command == "train"
        assert args.output_dir == Path("results")
        
        # Test bench-all subcommand
        args = parser.parse_args(["bench-all", "--model", "minilm", "--type", "baseline"])
        assert args.command == "bench-all"
        assert args.model == "minilm"
        assert args.type == "baseline"
        
        # Test ablation subcommand
        args = parser.parse_args([
            "ablation",
            "--time-mlp-hidden", "64,128",
            "--consistency-weight", "0.1,0.2",
        ])
        assert args.command == "ablation"
        assert args.time_mlp_hidden == "64,128"
        assert args.consistency_weight == "0.1,0.2"


class TestDatasetLoading:
    """Test dataset head loading."""
    
    def test_load_stsb_head(self):
        """Test loading first few samples of STS-B."""
        from tide_lite.data.datasets import load_stsb
        
        cfg = {"max_samples": 10, "cache_dir": "./data", "seed": 42}
        datasets = load_stsb(cfg)
        
        # Should have train, validation, test splits
        assert "train" in datasets
        assert "validation" in datasets
        assert "test" in datasets
        
        # Check we can access first sample
        if len(datasets["train"]) > 0:
            sample = datasets["train"][0]
            assert "sentence1" in sample
            assert "sentence2" in sample
            assert "score" in sample
    
    def test_load_quora_head(self):
        """Test loading first few samples of Quora."""
        from tide_lite.data.datasets import load_quora
        
        cfg = {"max_samples": 10, "cache_dir": "./data", "seed": 42}
        corpus, queries, qrels = load_quora(cfg)
        
        # Should return three components
        assert corpus is not None
        assert queries is not None
        assert qrels is not None
        
        # Check structure if data available
        if len(corpus) > 0:
            doc = corpus[0]
            assert "doc_id" in doc
            assert "text" in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
