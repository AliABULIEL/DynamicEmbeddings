# DEMRL++: Implementation Plan

## Overview
DEMRL++ (Dynamic Embeddings with Matryoshka Representation Learning) combines Curriculum Matryoshka learning with Adaptive LoRA for efficient, continually-adaptable text embeddings.

## Core Contributions

### 1. Curriculum-MRL (C-MRL)
- **Multi-resolution training**: Target dimensions {128, 256, 512, 768}
- **Curriculum schedule**: p_t(d) biases toward small dims early, transitions to uniform
- **Cross-dimension consistency**: Regularization to frozen 768-dim view with stop-gradient
- **Direct optimization**: Task loss at sampled dimension d, not just nested truncation

### 2. AdapterCache (LRU)
- **LoRA modules**: Parameter-efficient adaptation keyed by task/domain signature
- **LRU cache**: Capacity K ∈ {0, 2, 4, 8} with lazy-loading on cache miss
- **Few-shot warmup**: Optional N-step inner loop for new adapter initialization
- **Performance tracking**: Hit-rate monitoring and miss latency logging

### 3. Quality-Efficiency Frontier
- **Low-dim performance**: Target +1-2 Spearman improvement at 128-256d vs vanilla MRL
- **Efficiency metrics**: Latency (ms/query), throughput (qps), embedding size (MB)
- **Competitive at scale**: Match 512-768d baselines at 256d with lower resource usage

## Datasets

### STS-B (Semantic Textual Similarity)
- **Source**: HuggingFace glue/stsb
- **Task**: Sentence similarity scoring
- **Metric**: Spearman correlation
- **Split**: Train/Dev/Test (standard)

### Retrieval_small (Quora-based)
- **Source**: HuggingFace quora dataset
- **Corpus**: ~10k documents
- **Queries**: ~1k queries
- **Relevance**: Defined by duplicate pairs
- **Index**: FAISS (GPU if available, else CPU)
- **Metrics**: nDCG@10, Recall@10

## Baselines

### HuggingFace Models
1. **all-MiniLM-L6-v2** (384d) - Lightweight, efficient
2. **all-mpnet-base-v2** (768d) - Strong general-purpose
3. **intfloat/e5-base-v2** - Modern contrastive
4. **BAAI/bge-base-en-v1.5** - SOTA Chinese+English
5. **jinaai/jina-embeddings-v3** (optional) - Multi-lingual with LoRA

### Comparison Axes
- **Dimension sweep**: {128, 256, 512, 768} where applicable
- **Quality metrics**: Spearman (STS-B), nDCG@10 (retrieval)
- **Efficiency metrics**: Latency@batch=32, throughput, model size

## Training Configuration

### Curriculum Schedule
```python
# Early epochs: bias toward small dims
epoch_ratio = epoch / max_epochs
p_128 = 0.6 * (1 - epoch_ratio) + 0.25
p_256 = 0.3 * (1 - epoch_ratio) + 0.25
p_512 = 0.1 + 0.25 * epoch_ratio
p_768 = 0.25 * epoch_ratio
```

### Hyperparameters
- **Backbone**: MPNet-base or similar via sentence-transformers
- **Batch size**: 32-64 with gradient accumulation (4-8 steps)
- **Learning rate**: 3e-3 with cosine annealing
- **Mixed precision**: FP16/AMP for GPU training
- **Early stopping**: Patience=5 on validation loss
- **Checkpointing**: Every 100 iterations

### Loss Components
```python
L_total = L_task(z_d) + λ_d * L_consistency(z_d, sg(z_768))
λ_d = (1 - d/768) * base_lambda  # Stronger regularization for smaller dims
```

## Evaluation Protocol

### Dimension Analysis
- **Sweep**: Test all baselines and DEMRL++ at {128, 256, 512, 768}
- **Plots**: Score vs dimension, latency vs dimension
- **Frontier**: Pareto-optimal configurations

### AdapterCache Ablation
- **Cache sizes**: K ∈ {0, 2, 4}
- **Domain shift**: A→B→C→A simulation
- **Metrics**: Hit-rate, miss latency, quality retention

### Efficiency Profiling
- **Memory**: Peak GPU usage, adapter storage overhead
- **Speed**: ms/query@batch=32, queries per second
- **Size**: Embedding dimensions, model parameters

## Acceptance Criteria

### Quality Thresholds
1. **STS-B@768d**: Spearman ≥ 0.80
2. **Low-dim gain**: +1-2 Spearman over vanilla MRL at 128-256d
3. **Efficiency win**: 256d DEMRL++ matches 512-768d baseline performance

### Performance Requirements
1. **Training time**: < 45 min on Colab Pro (T4/V100/A100)
2. **Inference**: < 10ms/query for 256d embeddings
3. **Memory**: < 4GB GPU memory for training

### Robustness Checks
1. **Reproducibility**: 3 seeds with mean±std reporting
2. **Cache stability**: ≥98% quality retention under domain shift
3. **Convergence**: Loss decrease within 5 epochs

## Implementation Milestones

### Phase 1: Infrastructure (Tasks A-C)
- Project scaffold and configuration
- Data loaders for STS-B and retrieval_small
- Deterministic splits and preprocessing

### Phase 2: Core Models (Tasks D-E)
- Encoder with MRL heads
- C-MRL training loop with curriculum
- AdapterCache with LRU policy

### Phase 3: Evaluation (Tasks F-G)
- Baseline runners with profiling
- Comprehensive benchmarking suite
- Visualization and analysis tools

### Phase 4: Deployment (Tasks H-J)
- Unit tests and smoke tests
- Colab notebook with full pipeline
- Documentation and reproducibility artifacts

## Risk Mitigation

### Technical Risks
- **Memory overflow**: Gradient accumulation + mixed precision
- **Cache thrashing**: Bounded cache size + eviction policy
- **Convergence issues**: Learning rate scheduling + early stopping

### Evaluation Risks
- **Cherry-picking**: Fixed baselines + standard benchmarks
- **Overfitting**: Train/dev/test splits + cross-validation
- **Variance**: Multiple seeds + confidence intervals

## Expected Outcomes

### Research Contributions
1. **C-MRL**: Novel curriculum for multi-resolution embeddings
2. **AdapterCache**: System for online domain adaptation
3. **Efficiency analysis**: Comprehensive quality-efficiency trade-offs

### Practical Impact
- **Deployment-ready**: Colab-compatible with < 4GB memory
- **Flexible dimensions**: 128-768d from single model
- **Online adaptation**: No retraining for new domains

## Success Metrics Summary

| Metric | Target | Validation |
|--------|--------|------------|
| STS-B Spearman@768d | ≥ 0.80 | Standard test set |
| Low-dim improvement | +1-2 pts | vs vanilla MRL |
| 256d efficiency | Match 512d baseline | nDCG@10 + latency |
| Training time | < 45 min | Colab Pro GPU |
| Memory usage | < 4GB | Peak GPU allocation |
| Cache hit-rate | ≥ 90% | Under domain shift |
