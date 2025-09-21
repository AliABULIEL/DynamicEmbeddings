# DEMRL++ Implementation Plan

## Project Overview
**DEMRL++: Dynamic Embeddings with Matryoshka Representation Learning and Adaptive LoRA**

A research implementation demonstrating efficient, continually-adaptable text embeddings through:
- Curriculum Matryoshka Representation Learning (C-MRL) 
- Adaptive LoRA with LRU cache (AdapterCache)
- Multi-resolution embeddings at {128, 256, 512, 768} dimensions

## Method

### 1. Curriculum-MRL (C-MRL)
- **Per-batch dimension sampling**: Sample target dimension d ~ p_t(d) with annealed schedule
  - Early training: Bias toward smaller dimensions (128, 256)
  - Mid training: Uniform sampling across all dimensions
  - Late training: Focus on larger dimensions for refinement
- **Direct optimization**: Train task loss directly at sampled dimension d
- **Cross-dimension consistency**: Regularize lower dims to align with frozen 768-d representation
  - L_cons = Σ_d λ_d * (1 - cosine(pad(z_d), stop_grad(z_768)))
  - λ_d weights decay with dimension size

### 2. AdapterCache (LRU)
- **LoRA modules**: Parameter-efficient adapters keyed by task/domain signatures
- **LRU eviction**: Capacity K ∈ {2, 4, 8} with lazy loading on cache miss
- **Few-shot warmup**: Optional inner loop (≤10 steps) for new adapter initialization
- **Instrumentation**: Track hit rate, miss latency, memory overhead

### 3. Multi-Resolution Architecture
- **Base encoder**: MPNet-base or similar transformer (768-d output)
- **Truncated views**: z_d = z[:d] for d ∈ {128, 256, 512, 768}
- **Projection heads**: Learnable projection + LayerNorm per dimension
- **Single model**: Serves all dimension requirements via intelligent truncation

## Datasets

### STS-B (Semantic Textual Similarity)
- **Source**: HuggingFace glue/stsb
- **Train/Val/Test**: 5,749 / 1,500 / 1,379 sentence pairs
- **Metric**: Spearman correlation
- **Target**: ≥0.80 @ 768d, +1-2 points over vanilla MRL @ 128-256d

### Retrieval_small (Document Retrieval)
- **Source**: HuggingFace Quora duplicate questions
- **Corpus size**: ~10,000 documents
- **Query size**: ~1,000 queries
- **Index**: FAISS (GPU if available, CPU fallback)
- **Metrics**: nDCG@10 (primary), Recall@10 (optional)
- **Target**: Match 512-768d baseline performance @ 256d

## Baselines

### HuggingFace Models
1. **all-MiniLM-L6-v2** (384d)
   - 22M parameters, optimized for efficiency
   - Strong performance/size trade-off
   
2. **all-mpnet-base-v2** (768d)
   - 110M parameters, SOTA on many tasks
   - Primary comparison for full dimension
   
3. **intfloat/e5-base-v2** (768d)
   - Contrastive pre-training
   - Strong on retrieval tasks
   
4. **BAAI/bge-base-en-v1.5** (768d)
   - Recent SOTA, retrieval-optimized
   - Important for demonstrating competitiveness

5. **jinaai/jina-embeddings-v3** (optional)
   - Task-specific LoRA adapters
   - Comparison point for adapter approach

## Metrics

### Quality Metrics
- **STS-B**: Spearman correlation coefficient
- **Retrieval**: nDCG@10, Recall@10
- **Dimension sweep**: Performance at {128, 256, 512, 768}d
- **Adaptation**: Performance retention under A→B→C→A domain shift

### Efficiency Metrics
- **Latency**: ms/query @ batch_size=32
- **Throughput**: queries per second (qps)
- **Memory**: Embedding size in MB
- **Disk**: Model checkpoint size
- **Training**: Time to convergence, GPU memory usage

### AdapterCache Metrics
- **Hit rate**: % queries served from cache
- **Miss latency**: Time to load/warm new adapter
- **Memory overhead**: KB per cached adapter
- **Capacity analysis**: Performance vs K ∈ {0, 2, 4}

## Acceptance Criteria

### Primary Requirements
✓ **STS-B Performance**: ≥0.80 Spearman @ 768d
✓ **Low-dim improvement**: +1-2 Spearman points over vanilla MRL @ 128-256d
✓ **Retrieval parity**: nDCG@10 @ 256d matches 512-768d baseline
✓ **Efficiency gain**: Lower latency and memory @ 256d vs 768d baselines

### Secondary Requirements
✓ **Adaptation robustness**: ≥98% quality retention with AdapterCache under domain shift
✓ **Compute overhead**: <1% extra compute with adapter switching
✓ **Convergence**: Training completes within Colab Pro time limits
✓ **Reproducibility**: Fixed seeds, deterministic results

## Technical Stack

### Core Libraries
- **PyTorch**: 2.2-2.3 (GPU acceleration, AMP training)
- **Transformers**: Latest stable (model loading)
- **Sentence-Transformers**: Baseline models
- **PEFT**: LoRA implementation
- **FAISS**: Efficient similarity search
- **Accelerate**: Multi-GPU support (future)

### Development Tools
- **Config**: PyYAML for experiment management
- **Logging**: Built-in Python logging + W&B (optional)
- **Testing**: pytest for unit tests
- **Formatting**: black, ruff for code quality
- **Visualization**: matplotlib for plots

## Execution Timeline

### Phase 1: Foundation (Tasks A-C)
- Plan documentation
- Repository scaffolding
- Data pipeline implementation

### Phase 2: Core Model (Tasks D-E)
- C-MRL architecture
- AdapterCache system
- Training loop with curriculum

### Phase 3: Evaluation (Tasks F-G)
- Baseline runners
- Benchmark suite
- Visualization tools

### Phase 4: Deployment (Tasks H-J)
- Test coverage
- Colab notebook
- Documentation and reproducibility

## Risk Mitigation

### Technical Risks
- **Memory overflow**: Gradient accumulation, mixed precision, batch size tuning
- **Convergence issues**: Learning rate scheduling, warm-up, gradient clipping
- **Cache thrashing**: Bounded capacity, intelligent eviction policy

### Operational Risks
- **Colab timeout**: Checkpointing, session recovery
- **Data availability**: Local caching, fallback sources
- **Dependency conflicts**: Pinned versions, virtual environments

## Success Metrics

### Short-term (v1.0)
- Working implementation passing all acceptance criteria
- Clean, documented codebase
- Reproducible results
- Functional Colab demo

### Long-term (Future)
- Publication-ready results
- Community adoption
- Extension to temporal/streaming scenarios
- Integration with production systems
