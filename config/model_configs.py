"""
Configuration module for Dynamic Embeddings
Based on research papers and best practices
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class ExpertConfig:
    """Configuration for expert models"""

    # Expert models - only truly specialized ones that add value
    experts: Dict[str, Dict] = field(default_factory=lambda: {
        'general': {
            'model': 'BAAI/bge-base-en-v1.5',  # SOTA general
            'type': 'sentence-transformer',
            'dim': 768,
            'description': 'General-purpose SOTA embedding model'
        },
        'scientific': {
            'model': 'allenai/specter2',
            'type': 'sentence-transformer',
            'dim': 768,
            'description': 'Scientific papers and citations'
        },
        'code': {
            'model': 'microsoft/unixcoder-base',  # Better than codebert
            'type': 'transformer',
            'dim': 768,
            'description': 'Code understanding and retrieval'
        },
        'qa': {
            'model': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            'type': 'sentence-transformer',
            'dim': 768,
            'description': 'Question-answering optimized'
        },
        'instruct': {
            'model': 'intfloat/e5-base-v2',
            'type': 'sentence-transformer',
            'dim': 768,
            'description': 'Instruction-following embeddings'
        }
    })

    # Model dimensions
    hidden_dim: int = 768
    projection_dim: int = 512
    router_dim: int = 384

    # Number of experts
    @property
    def num_experts(self) -> int:
        return len(self.experts)


@dataclass
class RouterConfig:
    """Configuration for MoE router"""

    # Router model (lightweight)
    encoder_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    encoder_dim: int = 384

    # Router architecture
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1

    # Routing strategy
    top_k: int = 2  # Number of experts to use (sparsity)
    temperature: float = 1.0
    use_load_balancing: bool = True
    load_balance_alpha: float = 0.01

    # Noise for exploration
    routing_noise: float = 0.1
    noise_epsilon: float = 1e-2


@dataclass
class FusionConfig:
    """Configuration for AdapterFusion"""

    # Attention mechanism
    num_heads: int = 8
    dropout: float = 0.1

    # Number of fusion layers
    num_layers: int = 3

    # Gating mechanism
    use_gating: bool = True
    gate_temperature: float = 1.0


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Learning rates
    router_lr: float = 1e-4
    fusion_lr: float = 5e-5
    overall_lr: float = 2e-5

    # Batch sizes
    batch_size: int = 32
    gradient_accumulation_steps: int = 4

    # Training stages (curriculum learning)
    router_epochs: int = 3
    fusion_epochs: int = 3
    finetune_epochs: int = 10

    # Regularization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 500

    # Loss weights
    contrastive_weight: float = 1.0
    load_balance_weight: float = 0.01
    diversity_weight: float = 0.01

    # Temperature for contrastive learning
    temperature: float = 0.07

    # Early stopping
    patience: int = 3
    min_delta: float = 0.001

    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 3

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True  # Use FP16 for efficiency


@dataclass
class DataConfig:
    """Data configuration"""

    # Maximum sequence length
    max_seq_length: int = 256  # Reduced for efficiency

    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5

    # Validation split
    validation_split: float = 0.1
    test_split: float = 0.1

    # Dataset paths
    cache_dir: str = './cache'
    output_dir: str = './outputs'


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""

    # MTEB tasks to evaluate on (focus on where dynamic helps)
    mteb_tasks: List[str] = field(default_factory=lambda: [
        # Retrieval (most important)
        'MSMARCO',
        'NQ',
        'HotpotQA',
        'FEVER',
        'SciFact',
        'TRECCOVID',

        # Multi-domain classification
        'Banking77Classification',
        'MassiveIntentClassification',
        'AmazonCounterfactualClassification',

        # Cross-domain similarity
        'STS12',
        'STS22',
        'STSBenchmark',

        # Clustering
        'RedditClustering',
        'TwentyNewsgroupsClustering',
    ])

    # Baseline models for comparison
    baseline_models: List[str] = field(default_factory=lambda: [
        'sentence-transformers/all-mpnet-base-v2',
        'sentence-transformers/all-MiniLM-L6-v2',
        'BAAI/bge-base-en-v1.5',
        'intfloat/e5-base-v2',
    ])

    # Evaluation strategy
    eval_batch_size: int = 64
    use_cached_embeddings: bool = True


@dataclass
class CompleteConfig:
    """Complete configuration combining all components"""

    expert: ExpertConfig = field(default_factory=ExpertConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Experiment settings
    experiment_name: str = 'dynamic_embeddings_v2'
    seed: int = 42

    def save(self, path: str):
        """Save configuration to file"""
        import json
        from pathlib import Path

        config_dict = {
            'expert': self.expert.__dict__,
            'router': self.router.__dict__,
            'fusion': self.fusion.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'evaluation': {
                k: v for k, v in self.evaluation.__dict__.items()
                if k != 'mteb_tasks' and k != 'baseline_models'
            },
            'evaluation_tasks': self.evaluation.mteb_tasks,
            'baseline_models': self.evaluation.baseline_models,
            'experiment_name': self.experiment_name,
            'seed': self.seed
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from file"""
        import json

        with open(path, 'r') as f:
            config_dict = json.load(f)

        config = cls()

        # Load each component
        for key, value in config_dict.get('expert', {}).items():
            setattr(config.expert, key, value)
        for key, value in config_dict.get('router', {}).items():
            setattr(config.router, key, value)
        for key, value in config_dict.get('fusion', {}).items():
            setattr(config.fusion, key, value)
        for key, value in config_dict.get('training', {}).items():
            setattr(config.training, key, value)
        for key, value in config_dict.get('data', {}).items():
            setattr(config.data, key, value)
        for key, value in config_dict.get('evaluation', {}).items():
            setattr(config.evaluation, key, value)

        config.evaluation.mteb_tasks = config_dict.get('evaluation_tasks', [])
        config.evaluation.baseline_models = config_dict.get('baseline_models', [])
        config.experiment_name = config_dict.get('experiment_name', 'dynamic_embeddings_v2')
        config.seed = config_dict.get('seed', 42)

        return config


def get_task_specific_config(task: str) -> CompleteConfig:
    """Get configuration optimized for specific task"""

    config = CompleteConfig()

    if task == 'retrieval':
        # For retrieval, use more experts and higher temperature
        config.router.top_k = 3
        config.router.temperature = 1.2
        config.training.contrastive_weight = 1.5

    elif task == 'classification':
        # For classification, use fewer experts with lower temperature
        config.router.top_k = 2
        config.router.temperature = 0.5
        config.training.batch_size = 64

    elif task == 'similarity':
        # For similarity, consistent routing is important
        config.router.top_k = 2
        config.router.temperature = 0.3
        config.fusion.num_layers = 2

    elif task == 'clustering':
        # For clustering, diversity is important
        config.router.top_k = 4
        config.training.diversity_weight = 0.05

    return config