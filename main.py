"""
Main entry point for Dynamic Embeddings v2
Complete refactored system with modular architecture
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import numpy as np
from typing import List, Dict, Optional
import json
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from config.model_config import CompleteConfig, get_task_specific_config
from src.models.dynamic_model import DynamicEmbeddingModel
from src.training.trainer import DynamicEmbeddingTrainer
from src.training.dataset import DynamicEmbeddingDataset
from src.evaluation.mteb_evaluator import MTEBEvaluator
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def quick_demo():
    """Quick demonstration of the system capabilities"""
    logger.info("="*70)
    logger.info("QUICK DEMO - Dynamic Embeddings v2")
    logger.info("="*70)

    # Load configuration
    config = CompleteConfig()
    logger.info(f"\nConfiguration:")
    logger.info(f"  Experts: {list(config.expert.experts.keys())}")
    logger.info(f"  Device: {config.training.device}")
    logger.info(f"  Top-K routing: {config.router.top_k}")

    # Initialize model
    logger.info("\nInitializing model...")
    model = DynamicEmbeddingModel(config)

    # Test texts covering different domains
    test_texts = [
        # Scientific
        "Recent advances in quantum computing demonstrate exponential speedup for factorization",
        # Medical
        "The COVID-19 mRNA vaccines use lipid nanoparticles to deliver genetic instructions",
        # Financial
        "Federal Reserve raises interest rates to combat inflation pressures",
        # Code
        "def quicksort(arr): return sorted(arr) if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
        # Q&A
        "What is the capital of France and what is its population?",
        # Mixed domain
        "AI-driven drug discovery reduces pharmaceutical development costs by 50%",
    ]

    logger.info(f"\nProcessing {len(test_texts)} test texts...")

    # Encode texts
    embeddings = model.encode(test_texts, show_progress=True)
    logger.info(f"Generated embeddings: shape {embeddings.shape}")

    # Analyze routing patterns
    logger.info("\n" + "="*50)
    logger.info("ROUTING ANALYSIS")
    logger.info("="*50)

    routing_analysis = model.analyze_routing(test_texts)

    for i, text in enumerate(test_texts):
        logger.info(f"\nText {i+1}: {text[:60]}...")
        logger.info(f"  Entropy: {routing_analysis['entropy'][i]:.3f}")

        # Show top experts
        top_experts = routing_analysis['top_experts']
        expert_names = list(config.expert.experts.keys())
        logger.info("  Top experts:")
        for j in range(min(3, len(expert_names))):
            expert_idx = top_experts['indices'][i][j]
            weight = top_experts['weights'][i][j]
            logger.info(f"    - {expert_names[expert_idx]}: {weight:.3f}")

    # Calculate similarity matrix
    logger.info("\n" + "="*50)
    logger.info("SEMANTIC SIMILARITY ANALYSIS")
    logger.info("="*50)

    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)

    # Find most similar pairs
    similarities = []
    for i in range(len(test_texts)):
        for j in range(i+1, len(test_texts)):
            similarities.append((sim_matrix[i,j], i, j))

    similarities.sort(reverse=True)
    logger.info("\nMost similar text pairs:")
    for sim, i, j in similarities[:3]:
        logger.info(f"  Similarity: {sim:.3f}")
        logger.info(f"    Text {i+1}: {test_texts[i][:50]}...")
        logger.info(f"    Text {j+1}: {test_texts[j][:50]}...")

    return model, embeddings


def train_model(args):
    """Full training pipeline"""
    logger.info("="*70)
    logger.info("TRAINING DYNAMIC EMBEDDINGS MODEL")
    logger.info("="*70)

    # Load configuration
    if args.config:
        config = CompleteConfig.load(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = get_task_specific_config(args.task) if args.task else CompleteConfig()
        logger.info(f"Using {'task-specific' if args.task else 'default'} configuration")

    # Override with command-line arguments
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.overall_lr = args.learning_rate
    if args.epochs:
        config.training.finetune_epochs = args.epochs

    # Set seed
    set_seed(config.seed)

    # Create output directory
    output_dir = Path(args.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save configuration
    config.save(output_dir / "config.json")

    # Initialize model
    logger.info("\nInitializing model...")
    model = DynamicEmbeddingModel(config)

    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        model.load(args.checkpoint)

    # Load training data
    logger.info("\nLoading training data...")
    from datasets import load_dataset

    # Example: Load multiple datasets for diverse training
    all_texts = []
    all_labels = []

    # AG News (news domain)
    try:
        dataset = load_dataset('ag_news', split='train[:5000]')
        all_texts.extend(dataset['text'])
        all_labels.extend(dataset['label'])
        logger.info(f"  Loaded {len(dataset)} samples from AG News")
    except Exception as e:
        logger.warning(f"  Failed to load AG News: {e}")

    # IMDB (reviews domain)
    try:
        dataset = load_dataset('imdb', split='train[:5000]')
        all_texts.extend(dataset['text'])
        all_labels.extend([l + 4 for l in dataset['label']])  # Offset labels
        logger.info(f"  Loaded {len(dataset)} samples from IMDB")
    except Exception as e:
        logger.warning(f"  Failed to load IMDB: {e}")

    if not all_texts:
        # Fallback to synthetic data
        logger.warning("Using synthetic training data")
        all_texts = [
            "Scientific paper about machine learning",
            "Medical research on vaccines",
            "Financial news about markets",
            "Code snippet in Python",
            "Question about geography",
        ] * 100
        all_labels = [0, 1, 2, 3, 4] * 100

    logger.info(f"Total training samples: {len(all_texts)}")

    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=config.seed
    )

    # Create datasets and loaders
    train_dataset = DynamicEmbeddingDataset(
        train_texts, train_labels, augment=True
    )
    val_dataset = DynamicEmbeddingDataset(
        val_texts, val_labels, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size * 2,
        shuffle=False
    )

    # Initialize trainer
    logger.info("\nInitializing trainer...")
    trainer = DynamicEmbeddingTrainer(model, config, use_wandb=args.wandb)

    # Train
    logger.info("\nStarting training...")
    trainer.train(train_loader, val_loader)

    # Save final model
    final_path = output_dir / "final_model.pt"
    model.save(final_path)
    logger.info(f"Model saved to {final_path}")

    return model, output_dir


def evaluate_model(args):
    """Evaluate a trained model"""
    logger.info("="*70)
    logger.info("EVALUATING DYNAMIC EMBEDDINGS MODEL")
    logger.info("="*70)

    # Load configuration and model
    if args.config:
        config = CompleteConfig.load(args.config)
    else:
        config = CompleteConfig()

    # Initialize model
    logger.info("Loading model...")
    model = DynamicEmbeddingModel(config)

    if args.checkpoint:
        model.load(args.checkpoint)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    else:
        logger.warning("No checkpoint provided, using untrained model")

    # Initialize evaluator
    evaluator = MTEBEvaluator(model, config)

    # Run evaluation
    if args.mteb_tasks:
        tasks = args.mteb_tasks
    else:
        # Default tasks focusing on where dynamic embeddings excel
        tasks = [
            'Banking77Classification',  # Multi-domain classification
            'MSMARCO',  # Retrieval
            'STSBenchmark',  # Similarity
        ]

    results = {}
    for task in tasks:
        logger.info(f"\nEvaluating on {task}...")
        try:
            task_results = evaluator.evaluate_task(task)
            results[task] = task_results
            logger.info(f"Results: {task_results}")
        except Exception as e:
            logger.error(f"Failed to evaluate {task}: {e}")
            results[task] = {"error": str(e)}

    # Special evaluation: Domain shift
    logger.info("\n" + "="*50)
    logger.info("DOMAIN SHIFT EVALUATION")
    logger.info("="*50)

    domain_shift_results = evaluator.evaluate_domain_shift()
    results['domain_shift'] = domain_shift_results

    logger.info("\nDomain Shift Results:")
    for model_name, scores in domain_shift_results.items():
        logger.info(f"  {model_name}: {scores}")

    # Save results
    if args.output_dir:
        output_path = Path(args.output_dir) / "evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")

    return results


def inference_mode(args):
    """Use model for inference on custom texts"""
    logger.info("="*70)
    logger.info("INFERENCE MODE")
    logger.info("="*70)

    # Load model
    config = CompleteConfig.load(args.config) if args.config else CompleteConfig()
    model = DynamicEmbeddingModel(config)

    if args.checkpoint:
        model.load(args.checkpoint)
        logger.info(f"Loaded model from {args.checkpoint}")

    # Process input texts
    if args.input_file:
        with open(args.input_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(texts)} texts from {args.input_file}")
    else:
        # Interactive mode
        logger.info("Enter texts (empty line to finish):")
        texts = []
        while True:
            text = input("> ").strip()
            if not text:
                break
            texts.append(text)

    if not texts:
        logger.warning("No texts provided")
        return

    # Generate embeddings
    logger.info(f"\nGenerating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress=True)

    # Analyze routing
    routing_analysis = model.analyze_routing(texts)

    # Display results
    logger.info("\n" + "="*50)
    logger.info("RESULTS")
    logger.info("="*50)

    for i, text in enumerate(texts):
        logger.info(f"\nText {i+1}: {text[:100]}...")
        logger.info(f"  Embedding shape: {embeddings[i].shape}")
        logger.info(f"  Embedding norm: {np.linalg.norm(embeddings[i]):.3f}")
        logger.info(f"  Routing entropy: {routing_analysis['entropy'][i]:.3f}")

        # Top experts
        expert_names = list(config.expert.experts.keys())
        top_indices = routing_analysis['top_experts']['indices'][i]
        top_weights = routing_analysis['top_experts']['weights'][i]

        logger.info("  Top experts:")
        for idx, weight in zip(top_indices[:3], top_weights[:3]):
            logger.info(f"    - {expert_names[idx]}: {weight:.3f}")

    # Save embeddings if requested
    if args.output_dir:
        output_path = Path(args.output_dir) / "embeddings.npy"
        np.save(output_path, embeddings)
        logger.info(f"\nEmbeddings saved to {output_path}")

        # Save routing analysis
        analysis_path = Path(args.output_dir) / "routing_analysis.json"
        # Convert numpy arrays to lists for JSON serialization
        json_analysis = {
            'texts': texts,
            'entropy': routing_analysis['entropy'].tolist(),
            'dominant_expert': routing_analysis['dominant_expert'].tolist(),
            'expert_usage': routing_analysis['expert_usage'].tolist()
        }
        with open(analysis_path, 'w') as f:
            json.dump(json_analysis, f, indent=2)
        logger.info(f"Routing analysis saved to {analysis_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Dynamic Embeddings v2 - Research-based implementation"
    )

    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Demo mode
    demo_parser = subparsers.add_parser('demo', help='Quick demonstration')

    # Training mode
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, help='Path to config file')
    train_parser.add_argument('--task', type=str,
                            choices=['retrieval', 'classification', 'similarity'],
                            help='Task-specific configuration')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--checkpoint', type=str, help='Load from checkpoint')
    train_parser.add_argument('--output-dir', type=str, default='outputs',
                            help='Output directory')
    train_parser.add_argument('--wandb', action='store_true',
                            help='Use Weights & Biases logging')

    # Evaluation mode
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--config', type=str, help='Path to config file')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                           help='Path to model checkpoint')
    eval_parser.add_argument('--mteb-tasks', nargs='+',
                           help='MTEB tasks to evaluate on')
    eval_parser.add_argument('--output-dir', type=str, help='Save results to')

    # Inference mode
    infer_parser = subparsers.add_parser('inference', help='Run inference')
    infer_parser.add_argument('--config', type=str, help='Path to config file')
    infer_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')
    infer_parser.add_argument('--input-file', type=str,
                            help='File with texts (one per line)')
    infer_parser.add_argument('--output-dir', type=str,
                            help='Directory to save embeddings')

    args = parser.parse_args()

    # Execute based on mode
    if args.mode == 'demo':
        model, embeddings = quick_demo()

    elif args.mode == 'train':
        model, output_dir = train_model(args)
        logger.info(f"\nTraining complete! Model saved to {output_dir}")

    elif args.mode == 'evaluate':
        results = evaluate_model(args)
        logger.info("\nEvaluation complete!")

    elif args.mode == 'inference':
        inference_mode(args)

    else:
        # Default to demo if no mode specified
        logger.info("No mode specified, running demo...")
        model, embeddings = quick_demo()
        logger.info("\nTip: Use 'python main.py --help' to see all options")

    logger.info("\n" + "="*70)
    logger.info("Done!")
    logger.info("="*70)


if __name__ == "__main__":
    main()