"""Main experiment runner for dynamic embeddings."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import yaml
import argparse
import logging
from typing import Dict, List
import sys

sys.path.append('..')

from src.models.matryoshka_embedding import MatryoshkaEmbedding
from src.models.temporal_embedding import TemporalEmbedding
from src.models.contextual_embedding import ContextualEmbedding
from src.adapters.lora_adapter import MultiTaskLoRAAdapter
from src.training.trainer import DynamicEmbeddingTrainer
from src.training.loss_functions import ContrastiveLoss, MatryoshkaLoss
from src.evaluation.benchmarks import BenchmarkSuite
from src.data.data_loader import create_dataloaders
from src.utils.config_manager import ConfigManager
from src.visualization.plotter import create_comparison_plots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Main experiment runner."""

    def __init__(self, config_path: str):
        """Initialize experiment runner.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup directories
        self.experiment_dir = Path(self.config['experiment']['output_dir'])
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Experiment initialized: {self.config['experiment']['name']}")
        logger.info(f"Using device: {self.device}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def create_models(self) -> Dict[str, nn.Module]:
        """Create all model variants for comparison."""
        models = {}

        # Matryoshka model
        if self.config['models']['matryoshka']['enabled']:
            models['matryoshka'] = MatryoshkaEmbedding(
                **self.config['models']['matryoshka']['params']
            )

        # Temporal model
        if self.config['models']['temporal']['enabled']:
            models['temporal'] = TemporalEmbedding(
                **self.config['models']['temporal']['params']
            )

        # Contextual model
        if self.config['models']['contextual']['enabled']:
            models['contextual'] = ContextualEmbedding(
                **self.config['models']['contextual']['params']
            )

        # Add LoRA adapters if specified
        if self.config.get('use_lora', False):
            for name, model in models.items():
                models[f'{name}_lora'] = MultiTaskLoRAAdapter(
                    model,
                    self.config['lora']['task_configs']
                )

        logger.info(f"Created {len(models)} model variants")
        return models

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare datasets for training and evaluation."""
        # Create synthetic data for demonstration
        vocab_size = self.config['data']['vocab_size']
        train_size = self.config['data']['train_size']
        val_size = self.config['data']['val_size']
        test_size = self.config['data']['test_size']
        seq_len = self.config['data']['max_seq_length']
        batch_size = self.config['training']['batch_size']

        # Generate synthetic data
        def create_synthetic_dataset(size: int) -> TensorDataset:
            inputs = torch.randint(1, vocab_size, (size, seq_len))
            labels = torch.randint(0, 2, (size,))
            timestamps = torch.arange(size).float().unsqueeze(1).expand(size, seq_len)

            return TensorDataset(inputs, labels, timestamps)

        train_dataset = create_synthetic_dataset(train_size)
        val_dataset = create_synthetic_dataset(val_size)
        test_dataset = create_synthetic_dataset(test_size)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        logger.info(f"Data prepared: {train_size} train, {val_size} val, {test_size} test samples")

        return train_loader, val_loader, test_loader

    def train_models(
            self,
            models: Dict[str, nn.Module],
            train_loader: DataLoader,
            val_loader: DataLoader
    ):
        """Train all model variants."""
        for model_name, model in models.items():
            logger.info(f"\nTraining {model_name}...")

            # Create trainer
            trainer = DynamicEmbeddingTrainer(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                learning_rate=self.config['training']['learning_rate'],
                warmup_steps=self.config['training']['warmup_steps'],
                max_steps=self.config['training']['max_steps'],
                device=str(self.device),
                checkpoint_dir=str(self.experiment_dir / 'checkpoints' / model_name)
            )

            # Select loss function based on model type
            if 'matryoshka' in model_name:
                loss_fn = MatryoshkaLoss()
            else:
                loss_fn = ContrastiveLoss()

            # Train
            trainer.train(
                num_epochs=self.config['training']['num_epochs'],
                loss_fn=loss_fn,
                val_frequency=self.config['training']['val_frequency'],
                checkpoint_frequency=self.config['training']['checkpoint_frequency']
            )

            # Save final model
            model_path = self.experiment_dir / 'models' / f'{model_name}_final.pt'
            model_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_path)

            logger.info(f"Training completed for {model_name}")

    def evaluate_models(
            self,
            models: Dict[str, nn.Module],
            test_loader: DataLoader
    ) -> Dict:
        """Run comprehensive evaluation."""
        logger.info("\nRunning benchmark suite...")

        # Create benchmark suite
        benchmark = BenchmarkSuite(
            models=models,
            device=str(self.device),
            results_dir=str(self.experiment_dir / 'benchmarks')
        )

        # Prepare test data for benchmarks
        test_inputs = []
        test_labels = []

        for batch in test_loader:
            test_inputs.append(batch[0])
            test_labels.append(batch[1])

        test_inputs = torch.cat(test_inputs)
        test_labels = torch.cat(test_labels)

        # Run benchmarks
        results = {}

        # 1. Retrieval benchmark
        queries = test_inputs[:100]
        documents = test_inputs[100:600]
        relevance = torch.randint(0, 2, (100, 500))

        results['retrieval'] = benchmark.run_retrieval_benchmark(
            queries, documents, relevance
        )

        # 2. Classification benchmark
        results['classification'] = benchmark.run_classification_benchmark(
            test_loader, test_loader, num_classes=2
        )

        # 3. Speed benchmark
        input_sizes = [(8, 128), (16, 256), (32, 512)]
        results['speed'] = benchmark.run_speed_benchmark(input_sizes)

        # 4. Memory benchmark
        results['memory'] = benchmark.run_memory_benchmark(input_sizes)

        # Save results
        benchmark.save_results()

        # Generate comparison
        comparison_df = benchmark.compare_models()
        logger.info("\nModel Comparison:")
        logger.info(comparison_df.to_string())

        # Generate plots
        benchmark.plot_results()

        return results

    def run_ablation_study(self, models: Dict[str, nn.Module]):
        """Run ablation studies."""
        logger.info("\nRunning ablation studies...")

        ablation_results = {}

        # Test different embedding dimensions for Matryoshka
        if 'matryoshka' in models:
            model = models['matryoshka']
            dimensions = [64, 128, 256, 512, 768]

            dim_results = {}
            for dim in dimensions:
                # Create test input
                test_input = torch.randint(1, 1000, (32, 128)).to(self.device)

                # Get embeddings at specific dimension
                with torch.no_grad():
                    embeddings = model(test_input, target_dim=dim)

                # Measure quality (simplified - cosine similarity preservation)
                full_embeds = model(test_input, target_dim=768)
                truncated = full_embeds[..., :dim]
                similarity = torch.nn.functional.cosine_similarity(
                    embeddings.view(-1), truncated.view(-1), dim=0
                )

                dim_results[dim] = similarity.item()

            ablation_results['dimension_analysis'] = dim_results

        # Save ablation results
        import json
        with open(self.experiment_dir / 'ablation_results.json', 'w') as f:
            json.dump(ablation_results, f, indent=2)

        logger.info(f"Ablation results: {ablation_results}")

    def run(self):
        """Run complete experiment."""
        logger.info("=" * 50)
        logger.info(f"Starting experiment: {self.config['experiment']['name']}")
        logger.info("=" * 50)

        # 1. Create models
        models = self.create_models()

        # 2. Prepare data
        train_loader, val_loader, test_loader = self.prepare_data()

        # 3. Train models
        if self.config['experiment']['train']:
            self.train_models(models, train_loader, val_loader)

        # 4. Evaluate models
        if self.config['experiment']['evaluate']:
            results = self.evaluate_models(models, test_loader)

        # 5. Run ablation studies
        if self.config['experiment']['ablation']:
            self.run_ablation_study(models)

        logger.info("=" * 50)
        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved to: {self.experiment_dir}")
        logger.info("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run dynamic embeddings experiment')
    parser.add_argument(
        '--config',
        type=str,
        default='config/experiment_config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Run experiment
    runner = ExperimentRunner(args.config)
    runner.run()


if __name__ == '__main__':
    main()