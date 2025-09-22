"""Unit tests for CLI argument parsers."""

import argparse
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.tide_lite.cli.train_cli import (
    create_train_parser,
    load_yaml_config,
    merge_configs,
    parse_train_args,
)
from src.tide_lite.cli.eval_stsb_cli import create_eval_stsb_parser
from src.tide_lite.cli.eval_quora_cli import create_eval_quora_parser
from src.tide_lite.cli.eval_temporal_cli import create_eval_temporal_parser
from src.tide_lite.cli.aggregate_cli import create_aggregate_parser
from src.tide_lite.cli.plots_cli import create_plots_parser
from src.tide_lite.cli.report_cli import create_report_parser


class TestTrainCLI:
    """Test training CLI parser."""
    
    def test_create_train_parser(self):
        """Test train parser creation."""
        parser = create_train_parser()
        
        # Check that essential arguments exist
        args = parser.parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/output"
        ])
        
        assert args.data_dir == "/tmp/data"
        assert args.output_dir == "/tmp/output"
    
    def test_train_parser_with_config(self):
        """Test train parser with config file."""
        parser = create_train_parser()
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode='w', delete=False) as f:
            yaml.dump({"batch_size": 64, "lr": 0.001}, f)
            config_path = f.name
        
        try:
            args = parser.parse_args([
                "--config", config_path,
                "--data-dir", "/tmp/data",
                "--output-dir", "/tmp/output"
            ])
            assert args.config == config_path
        finally:
            Path(config_path).unlink()
    
    def test_train_parser_overrides(self):
        """Test CLI argument overrides."""
        parser = create_train_parser()
        
        args = parser.parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/output",
            "--batch-size", "32",
            "--lr", "0.0001",
            "--epochs", "10",
            "--encoder-name", "bert-base-uncased"
        ])
        
        assert args.batch_size == 32
        assert args.lr == 0.0001
        assert args.epochs == 10
        assert args.encoder_name == "bert-base-uncased"
    
    def test_train_parser_boolean_flags(self):
        """Test boolean flags in train parser."""
        parser = create_train_parser()
        
        # Test with flags
        args = parser.parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/output",
            "--freeze-encoder",
            "--use-wandb"
        ])
        assert args.freeze_encoder is True
        assert args.use_wandb is True
        
        # Test without flags (defaults)
        args = parser.parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/output"
        ])
        # Check defaults if any


class TestEvalCLIs:
    """Test evaluation CLI parsers."""
    
    def test_eval_stsb_parser(self):
        """Test STS-B evaluation parser."""
        parser = create_eval_stsb_parser()
        
        args = parser.parse_args([
            "--checkpoint", "/tmp/model.pt",
            "--data-dir", "/tmp/data"
        ])
        
        assert args.checkpoint == "/tmp/model.pt"
        assert args.data_dir == "/tmp/data"
    
    def test_eval_quora_parser(self):
        """Test Quora evaluation parser."""
        parser = create_eval_quora_parser()
        
        args = parser.parse_args([
            "--checkpoint", "/tmp/model.pt",
            "--data-dir", "/tmp/data",
            "--batch-size", "64"
        ])
        
        assert args.checkpoint == "/tmp/model.pt"
        assert args.data_dir == "/tmp/data"
        assert args.batch_size == 64
    
    def test_eval_temporal_parser(self):
        """Test temporal evaluation parser."""
        parser = create_eval_temporal_parser()
        
        args = parser.parse_args([
            "--checkpoint", "/tmp/model.pt",
            "--data-dir", "/tmp/data",
            "--temporal-split", "week"
        ])
        
        assert args.checkpoint == "/tmp/model.pt"
        assert args.data_dir == "/tmp/data"
        assert args.temporal_split == "week"


class TestAggregateCLI:
    """Test aggregation CLI parser."""
    
    def test_aggregate_parser(self):
        """Test aggregate parser."""
        parser = create_aggregate_parser()
        
        args = parser.parse_args([
            "--results-dir", "/tmp/results",
            "--output", "/tmp/aggregated.json"
        ])
        
        assert args.results_dir == "/tmp/results"
        assert args.output == "/tmp/aggregated.json"
    
    def test_aggregate_parser_pattern(self):
        """Test aggregate parser with pattern."""
        parser = create_aggregate_parser()
        
        args = parser.parse_args([
            "--results-dir", "/tmp/results",
            "--pattern", "*.json",
            "--output", "/tmp/aggregated.json"
        ])
        
        assert args.pattern == "*.json"


class TestPlotsCLI:
    """Test plots CLI parser."""
    
    def test_plots_parser(self):
        """Test plots parser."""
        parser = create_plots_parser()
        
        args = parser.parse_args([
            "--results", "/tmp/results.json",
            "--output-dir", "/tmp/plots"
        ])
        
        assert args.results == "/tmp/results.json"
        assert args.output_dir == "/tmp/plots"
    
    def test_plots_parser_types(self):
        """Test plots parser with plot types."""
        parser = create_plots_parser()
        
        args = parser.parse_args([
            "--results", "/tmp/results.json",
            "--output-dir", "/tmp/plots",
            "--plot-types", "loss", "accuracy"
        ])
        
        assert "loss" in args.plot_types
        assert "accuracy" in args.plot_types


class TestReportCLI:
    """Test report CLI parser."""
    
    def test_report_parser(self):
        """Test report parser."""
        parser = create_report_parser()
        
        args = parser.parse_args([
            "--experiment-dir", "/tmp/experiment",
            "--output", "/tmp/report.md"
        ])
        
        assert args.experiment_dir == "/tmp/experiment"
        assert args.output == "/tmp/report.md"
    
    def test_report_parser_format(self):
        """Test report parser with format."""
        parser = create_report_parser()
        
        args = parser.parse_args([
            "--experiment-dir", "/tmp/experiment",
            "--output", "/tmp/report.html",
            "--format", "html"
        ])
        
        assert args.format == "html"


class TestConfigUtilities:
    """Test configuration utilities."""
    
    def test_load_yaml_config(self):
        """Test YAML config loading."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode='w', delete=False) as f:
            config_data = {
                "model": {
                    "encoder_name": "bert-base",
                    "hidden_dim": 768
                },
                "training": {
                    "batch_size": 32,
                    "lr": 0.001
                }
            }
            yaml.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            loaded_config = load_yaml_config(config_path)
            assert loaded_config == config_data
        finally:
            config_path.unlink()
    
    def test_load_yaml_config_not_found(self):
        """Test error when config file not found."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config(Path("/nonexistent/config.yaml"))
    
    def test_merge_configs(self):
        """Test configuration merging."""
        base_config = {
            "model": {"hidden_dim": 384, "dropout": 0.1},
            "training": {"batch_size": 32, "lr": 0.001}
        }
        
        overrides = {
            "model": {"hidden_dim": 768},  # Override
            "training": {"epochs": 10}  # New key
        }
        
        merged = merge_configs(base_config, overrides)
        
        assert merged["model"]["hidden_dim"] == 768  # Overridden
        assert merged["model"]["dropout"] == 0.1  # Preserved
        assert merged["training"]["batch_size"] == 32  # Preserved
        assert merged["training"]["epochs"] == 10  # New
    
    def test_parse_train_args_integration(self):
        """Test full argument parsing integration."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode='w', delete=False) as f:
            yaml.dump({"batch_size": 64, "lr": 0.001}, f)
            config_path = f.name
        
        try:
            with patch('sys.argv', ['train.py', 
                                   '--config', config_path,
                                   '--data-dir', '/tmp/data',
                                   '--output-dir', '/tmp/output',
                                   '--batch-size', '128']):  # Override
                args = parse_train_args()
                
                # Config value should be overridden by CLI
                assert args.batch_size == 128
                assert args.lr == 0.001  # From config
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
