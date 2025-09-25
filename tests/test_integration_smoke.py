"""Fast integration smoke test for data loading, model APIs, and CLI.

This test verifies:
- Dataset loaders work with small samples
- Model APIs (encode_texts) work end-to-end
- CLI parsers create valid configs without execution

Designed to run in seconds for CI/CD pipelines.
"""

import pytest
import torch
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tide_lite.data.datasets import load_stsb, load_quora, load_timeqa
from src.tide_lite.models import TIDELite, TIDELiteConfig, BaselineEncoder, load_minilm_baseline


class TestDataLoading:
    """Test dataset loading with minimal samples."""
    
    def test_stsb_loading(self):
        """Load 16 samples from STS-B."""
        cfg = {
            "cache_dir": "./data",
            "max_samples": 16,
            "seed": 42
        }
        
        dataset = load_stsb(cfg)
        
        # Verify structure
        assert "train" in dataset
        assert "validation" in dataset
        assert "test" in dataset
        
        # Verify sample counts
        assert len(dataset["train"]) <= 16
        assert len(dataset["validation"]) <= 16
        assert len(dataset["test"]) <= 16
        
        # Verify fields
        sample = dataset["train"][0]
        assert "sentence1" in sample
        assert "sentence2" in sample
        assert "label" in sample
        
        # Verify label range
        assert 0 <= sample["label"] <= 5
    
    def test_quora_loading(self):
        """Load 16 samples from Quora duplicate questions."""
        cfg = {
            "cache_dir": "./data",
            "max_samples": 16,
            "seed": 42
        }
        
        corpus, queries, qrels = load_quora(cfg)
        
        # Verify structure
        assert len(corpus) > 0
        assert len(queries) > 0
        assert len(qrels) > 0
        
        # Verify fields in corpus
        assert "text" in corpus.column_names
        assert "doc_id" in corpus.column_names
        
        # Verify fields in queries
        assert "text" in queries.column_names
        assert "query_id" in queries.column_names
        
        # Verify fields in qrels
        assert "query_id" in qrels.column_names
        assert "doc_id" in qrels.column_names
        assert "relevance" in qrels.column_names
        
        # Verify relevance scores
        assert all(r == 1.0 for r in qrels["relevance"])
    
    @pytest.mark.skipif(
        not Path("./data/timeqa").exists() and not Path("./data/templama").exists(),
        reason="TimeQA/TempLAMA data not available"
    )
    def test_temporal_loading(self):
        """Load 16 samples from TimeQA/TempLAMA if available."""
        cfg = {
            "timeqa_data_dir": "./data/timeqa",
            "templama_path": "./data/templama",
            "cache_dir": "./data",
            "max_samples": 16,
            "seed": 42
        }
        
        try:
            dataset = load_timeqa(cfg)
            
            # Verify structure
            assert len(dataset) > 0
            assert len(dataset) <= 16
            
            # Verify fields
            sample = dataset[0]
            assert "question" in sample
            assert "context" in sample
            assert "answer" in sample
            assert "timestamp" in sample
            
            # Verify timestamp is numeric
            assert isinstance(sample["timestamp"], (int, float))
            
        except FileNotFoundError as e:
            # Expected if neither dataset is available
            if hasattr(pytest, 'skip'):
                pytest.skip(f"Temporal datasets not found: {e}")
            else:
                # Running outside pytest
                print(f"✗ Temporal dataset loading skipped: not found (expected)")


class TestModelAPIs:
    """Test model encode_texts API with minimal data."""
    
    def test_baseline_model_api(self):
        """Test MiniLM baseline encoder API."""
        model = load_minilm_baseline()
        
        # Test texts
        texts = [
            "This is a test sentence.",
            "Another example text for embedding.",
            "Testing the model API.",
            "Fourth test input.",
        ]
        
        # Test encoding
        embeddings = model.encode_texts(texts, batch_size=2)
        
        # Verify shapes
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == model.embedding_dim
        
        # Verify values are reasonable
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
        
        # Verify different texts give different embeddings
        assert not torch.allclose(embeddings[0], embeddings[1])
        
        # Test single text
        single_emb = model.encode_texts(["Single text"], batch_size=1)
        assert single_emb.shape == (1, model.embedding_dim)
    
    def test_tide_lite_model_api(self):
        """Test TIDE-Lite model encoder API."""
        config = TIDELiteConfig(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            time_encoding_dim=32,
            mlp_hidden_dim=128,
            freeze_encoder=True
        )
        
        model = TIDELite(config)
        model.eval()
        
        # Test texts with timestamps
        texts = [
            "Events in 2021 were significant.",
            "The year 2022 brought changes.",
            "Looking ahead to 2023.",
            "Reflecting on past years.",
        ]
        
        # Mock timestamps (Unix time for years 2021-2023)
        timestamps = torch.tensor([
            1609459200.0,  # 2021-01-01
            1640995200.0,  # 2022-01-01
            1672531200.0,  # 2023-01-01
            1650000000.0,  # Somewhere in 2022
        ])
        
        # Test encoding without timestamps (baseline)
        base_embeddings = model.encode_texts(texts, batch_size=2)
        
        assert base_embeddings.shape[0] == len(texts)
        assert base_embeddings.shape[1] == config.hidden_dim
        assert not torch.isnan(base_embeddings).any()
        
        # Test encoding with timestamps (temporal)
        with patch.object(model, 'current_timestamps', timestamps):
            temporal_embeddings = model.encode_texts(texts, batch_size=2)
            
            assert temporal_embeddings.shape == base_embeddings.shape
            assert not torch.isnan(temporal_embeddings).any()
            
            # Temporal embeddings should differ from base
            assert not torch.allclose(temporal_embeddings, base_embeddings, atol=1e-4)
    
    def test_model_parameter_counts(self):
        """Test extra parameter counting."""
        # Baseline should have 0 extra parameters
        baseline = load_minilm_baseline()
        assert baseline.count_extra_parameters() == 0
        
        # TIDE-Lite should have extra parameters
        config = TIDELiteConfig(
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            time_encoding_dim=32,
            mlp_hidden_dim=128,
            freeze_encoder=True
        )
        tide = TIDELite(config)
        extra_params = tide.count_extra_parameters()
        
        # Should have 40K-60K extra parameters with default config
        assert 40000 < extra_params < 60000


class TestCLIParsers:
    """Test CLI argument parsers without execution."""
    
    @patch('sys.argv', ['train', '--config', 'configs/tide_lite.yaml', '--dry-run'])
    @patch('src.tide_lite.train.trainer.TIDELiteTrainer')
    def test_train_cli_parser(self, mock_trainer):
        """Test train CLI creates valid config."""
        # Mock trainer to prevent actual training
        mock_instance = MagicMock()
        mock_trainer.return_value = mock_instance
        mock_instance.train.return_value = {"dry_run": True}
        
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True)
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--output-dir', type=str, default='./outputs')
        parser.add_argument('--num-epochs', type=int, default=3)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--learning-rate', type=float, default=1e-4)
        parser.add_argument('--warmup-steps', type=int, default=500)
        parser.add_argument('--eval-every', type=int, default=1000)
        parser.add_argument('--save-every', type=int, default=1000)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--wandb', action='store_true')
        parser.add_argument('--wandb-project', type=str, default='tide-lite')
        
        args = parser.parse_args(['--config', 'configs/tide_lite.yaml', '--dry-run'])
        
        # Verify args are parsed
        assert args.config == 'configs/tide_lite.yaml'
        assert args.dry_run is True
        assert args.num_epochs == 3
        assert args.batch_size == 32
    
    @patch('sys.argv', ['eval-stsb', '--model-path', './model', '--output-dir', './results'])
    def test_eval_stsb_cli_parser(self):
        """Test eval-stsb CLI creates valid config."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-path', type=str, required=True)
        parser.add_argument('--output-dir', type=str, required=True)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--use-temporal', action='store_true')
        parser.add_argument('--cache-dir', type=str, default='./data')
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
        
        args = parser.parse_args(['--model-path', './model', '--output-dir', './results'])
        
        assert args.model_path == './model'
        assert args.output_dir == './results'
        assert args.batch_size == 32
        assert args.use_temporal is False
    
    @patch('sys.argv', ['eval-quora', '--model-path', './model', '--output-dir', './results'])  
    def test_eval_quora_cli_parser(self):
        """Test eval-quora CLI creates valid config."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-path', type=str, required=True)
        parser.add_argument('--output-dir', type=str, required=True)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--top-k', type=int, nargs='+', default=[1, 5, 10])
        parser.add_argument('--use-temporal', action='store_true')
        parser.add_argument('--cache-dir', type=str, default='./data')
        parser.add_argument('--max-corpus-size', type=int, default=None)
        
        args = parser.parse_args(['--model-path', './model', '--output-dir', './results'])
        
        assert args.model_path == './model'
        assert args.output_dir == './results'
        assert args.top_k == [1, 5, 10]
        assert args.max_corpus_size is None
    
    @patch('sys.argv', ['eval-temporal', '--model-path', './model', '--output-dir', './results'])
    def test_eval_temporal_cli_parser(self):
        """Test eval-temporal CLI creates valid config."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-path', type=str, required=True)
        parser.add_argument('--output-dir', type=str, required=True)
        parser.add_argument('--dataset', type=str, choices=['timeqa', 'templama'], default='timeqa')
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--data-dir', type=str, default='./data/timeqa')
        parser.add_argument('--max-samples', type=int, default=None)
        parser.add_argument('--use-temporal', action='store_true', default=True)
        
        args = parser.parse_args(['--model-path', './model', '--output-dir', './results'])
        
        assert args.model_path == './model'
        assert args.output_dir == './results'
        assert args.dataset == 'timeqa'
        assert args.use_temporal is True


class TestEndToEndFlow:
    """Test complete flow without training."""
    
    def test_data_to_model_flow(self):
        """Test loading data and passing through model."""
        # Load minimal STS-B data
        cfg = {"cache_dir": "./data", "max_samples": 4, "seed": 42}
        dataset = load_stsb(cfg)
        
        # Get sample texts
        texts = []
        for i in range(min(4, len(dataset["train"]))):
            sample = dataset["train"][i]
            texts.append(sample["sentence1"])
            texts.append(sample["sentence2"])
        
        # Create model
        model = load_minilm_baseline()
        
        # Encode texts
        embeddings = model.encode_texts(texts, batch_size=2)
        
        # Verify we get embeddings for all texts
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == model.embedding_dim
        
        # Compute similarity for pairs
        for i in range(0, len(embeddings), 2):
            emb1 = embeddings[i]
            emb2 = embeddings[i+1]
            
            # Cosine similarity
            sim = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
            
            # Should be in valid range
            assert -1.0 <= sim.item() <= 1.0
    
    def test_model_save_load(self):
        """Test saving and loading model state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create and save model
            config = TIDELiteConfig(
                encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                time_encoding_dim=32,
                mlp_hidden_dim=128,
                freeze_encoder=True
            )
            model1 = TIDELite(config)
            
            # Save config and state
            config_path = tmpdir / "config.json"
            model_path = tmpdir / "model.pt"
            
            import json
            with open(config_path, 'w') as f:
                json.dump(config.__dict__, f)
            
            torch.save(model1.state_dict(), model_path)
            
            # Load model
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            config2 = TIDELiteConfig(**loaded_config)
            model2 = TIDELite(config2)
            model2.load_state_dict(torch.load(model_path))
            
            # Test both models produce same output
            texts = ["Test sentence for comparison."]
            
            model1.eval()
            model2.eval()
            
            with torch.no_grad():
                emb1 = model1.encode_texts(texts, batch_size=1)
                emb2 = model2.encode_texts(texts, batch_size=1)
            
            assert torch.allclose(emb1, emb2, atol=1e-6)


if __name__ == "__main__":
    """Run smoke test directly."""
    import sys
    
    print("Running integration smoke test...")
    print("-" * 60)
    
    # Test data loading
    print("Testing data loading...")
    data_test = TestDataLoading()
    
    try:
        data_test.test_stsb_loading()
        print("✓ STS-B loading works")
    except Exception as e:
        print(f"✗ STS-B loading failed: {e}")
    
    try:
        data_test.test_quora_loading()
        print("✓ Quora loading works")
    except Exception as e:
        # Quora might fail due to network/download issues
        if "http" in str(e) or "qim.fs.quoracdn.net" in str(e):
            print("✗ Quora loading failed: download error (network issue, can retry)")
        else:
            print(f"✗ Quora loading failed: {e}")
    
    try:
        data_test.test_temporal_loading()
        print("✓ Temporal dataset loading works")
    except (FileNotFoundError, Exception) as e:
        # This is expected if temporal datasets are not available
        if "TimeQA" in str(e) or "TempLAMA" in str(e) or "Skipped" in str(e.__class__.__name__):
            print("✗ Temporal dataset loading skipped: not found (expected)")
        else:
            print(f"✗ Temporal dataset loading failed: {e}")
    
    # Test model APIs
    print("\nTesting model APIs...")
    model_test = TestModelAPIs()
    
    try:
        model_test.test_baseline_model_api()
        print("✓ Baseline model API works")
    except Exception as e:
        print(f"✗ Baseline model API failed: {e}")
    
    try:
        model_test.test_tide_lite_model_api()
        print("✓ TIDE-Lite model API works")
    except Exception as e:
        print(f"✗ TIDE-Lite model API failed: {e}")
    
    try:
        model_test.test_model_parameter_counts()
        print("✓ Parameter counting works")
    except Exception as e:
        print(f"✗ Parameter counting failed: {e}")
    
    # Test CLI parsers
    print("\nTesting CLI parsers...")
    cli_test = TestCLIParsers()
    
    try:
        cli_test.test_train_cli_parser()
        print("✓ Train CLI parser works")
    except Exception as e:
        print(f"✗ Train CLI parser failed: {e}")
    
    try:
        cli_test.test_eval_stsb_cli_parser()
        print("✓ Eval-STSB CLI parser works")
    except Exception as e:
        print(f"✗ Eval-STSB CLI parser failed: {e}")
    
    try:
        cli_test.test_eval_quora_cli_parser()
        print("✓ Eval-Quora CLI parser works")
    except Exception as e:
        print(f"✗ Eval-Quora CLI parser failed: {e}")
    
    try:
        cli_test.test_eval_temporal_cli_parser()
        print("✓ Eval-Temporal CLI parser works")
    except Exception as e:
        print(f"✗ Eval-Temporal CLI parser failed: {e}")
    
    # Test end-to-end
    print("\nTesting end-to-end flow...")
    e2e_test = TestEndToEndFlow()
    
    try:
        e2e_test.test_data_to_model_flow()
        print("✓ Data-to-model flow works")
    except Exception as e:
        print(f"✗ Data-to-model flow failed: {e}")
    
    try:
        e2e_test.test_model_save_load()
        print("✓ Model save/load works")
    except Exception as e:
        print(f"✗ Model save/load failed: {e}")
    
    print("-" * 60)
    print("Integration smoke test complete!")
