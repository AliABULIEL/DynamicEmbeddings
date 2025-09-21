# DEMRL++ Implementation Plan

## Overview
**DEMRL++**: Dynamic Embeddings with Curriculum Matryoshka Representation Learning and Adaptive LoRA

A research framework addressing the quality-efficiency-adaptability trade-off in text embeddings through:
1. **Curriculum-MRL (C-MRL)**: Multi-resolution embeddings with dimension-aware curriculum
2. **AdapterCache**: LRU-based LoRA adapter management for online domain adaptation

## Method

### 1. Curriculum Matryoshka Learning (C-MRL)
- **Target dimensions**: {128, 256, 512, 768}
- **Per-batch sampling**: Sample dimension d ~ p_t(d) with annealed schedule
  - Early training: Bias toward small dimensions (p_128=0.6, p_256=0.3, p_512=0.1, p_768=0.0)
  - Mid training: Gradual shift to uniform (p_128=p_256=p_512=p_768=0.25)
- **Loss components**:
  - Task loss at sampled dimension d
  - Cross-dimension consistency: L_cons = λ_d * (1 - cos(z_d, sg(z_768)))
  - Stop-gradient on z_768 to prevent collapse
  - Dimension-aware weighting: λ_d = (1 - d/768) * base_weight

### 2. AdapterCache with LRU
- **Architecture**: LoRA adapters (rank 8, alpha 16) per task/domain
- **Cache management**:
  - LRU eviction with capacity K ∈ {2, 4}
  - Lazy loading on cache miss
  - Track hit rate and miss latency
- **Few-shot warmup**: Optional N-step SGD (N ≤ 10) for new adapters
- **Signatures**: Task/domain/language identifiers for adapter selection

### 3. Implementation Details
- **Backbone**: MPNet-base (768d) or MiniLM (384d for testing)
- **Projection heads**: Linear + LayerNorm for each target dimension
- **Training**: Mixed precision (FP16), gradient accumulation, cosine annealing
- **Efficiency**: Batch size tuning, FAISS indexing, embedding truncation

## Datasets

### STS-B (Semantic Textual Similarity)
- **Source**: HuggingFace `glue/stsb`
- **Split**: Train (5,749), Dev (1,500), Test (1,379)
- **Metric**: Spearman correlation
- **Target**: ≥0.80 @ 768d, improvement at low dims

### Retrieval Small (from Quora Duplicates)
- **Source**: HuggingFace `quora` dataset
- **Corpus**: ~10,000 unique questions
- **Queries**: ~1,000 held-out questions
- **Relevance**: Duplicate pairs define positive matches
- **Index**: FAISS (GPU if available, else CPU)
- **Metrics**: nDCG@10, Recall@10

## Baselines

### HuggingFace Models
1. **all-MiniLM-L6-v2** (384d)
   - Lightweight, 22M params
   - Strong efficiency baseline
   
2. **all-mpnet-base-v2** (768d)
   - Our primary comparison
   - State-of-art general purpose
   
3. **intfloat/e5-base-v2** (768d)
   - Contrastive pre-training
   - Strong retrieval performance
   
4. **BAAI/bge-base-en-v1.5** (768d)
   - Recent SOTA on MTEB
   - Multi-task trained

### Vanilla MRL Baseline
- Standard Matryoshka without curriculum
- Same architecture as DEMRL++ minus C-MRL schedule

## Metrics

### Quality Metrics
- **STS-B Spearman**: Correlation on test set
- **Retrieval nDCG@10**: Ranking quality
- **Retrieval Recall@10**: Coverage of relevant items

### Efficiency Metrics
- **Latency**: ms/query @ batch=32
- **Throughput**: queries/second
- **Embedding size**: MB for 10K vectors
- **Memory footprint**: Peak GPU/RAM usage

### Adaptation Metrics
- **Cache hit rate**: % queries served from cache
- **Miss latency**: Time to load/warm adapter
- **Domain shift robustness**: Performance on A→B→C→A

## Acceptance Criteria

### Primary Goals
1. ✅ **STS-B Performance**: ≥0.80 Spearman @ 768d
2. ✅ **Low-dim improvement**: +1-2 Spearman over vanilla MRL @ 128-256d
3. ✅ **Efficiency win**: @ 256d, match 512-768d baseline with lower latency/size

### Secondary Goals
4. **AdapterCache**: ≥98% quality retention with K∈{2,4}
5. **Latency bound**: <1% overhead from adapter switching
6. **Memory efficiency**: <2GB GPU for inference

## Experimental Protocol

### Training Phases
1. **Smoke test** (2-5 min CPU): Verify loss decrease
2. **Short GPU run**: Full dim sweep {128, 256, 512, 768}
3. **Full training**: 10 epochs with early stopping

### Evaluation Protocol
1. **Baseline evaluation**: All HF models on both tasks
2. **DEMRL++ sweep**: Test each dimension
3. **Ablations**:
   - C-MRL vs vanilla MRL
   - With/without consistency loss
   - AdapterCache K ∈ {0, 2, 4}
4. **Domain shift**: Synthetic A→B→C→A sequence

### Reproducibility
- Fixed seeds (42, 43, 44) for 3-run averaging
- Log environment: CUDA version, GPU model, library versions
- Save configs and checkpoints for all experiments
- Generate outputs/run.json with full manifest

## Expected Outcomes

### Quantitative
- **STS-B**: 0.82-0.85 Spearman @ 768d, 0.75-0.78 @ 256d, 0.70-0.73 @ 128d
- **Retrieval**: 0.40-0.45 nDCG@10 @ 256d matching 768d baselines
- **Latency**: 2-3x speedup @ 256d vs 768d
- **Memory**: 3-4x reduction @ 256d

### Qualitative
- Clear quality-efficiency frontiers in plots
- Smooth curriculum learning curves
- Effective adapter specialization logs

## Timeline
1. **Setup** (30 min): Scaffold, configs, dependencies
2. **Data** (1 hr): Loaders, FAISS indexing
3. **Models** (2 hr): Encoder, MRL heads, adapters
4. **Training** (2 hr): Loop, losses, metrics
5. **Evaluation** (2 hr): Baselines, benchmarks, plots
6. **Testing** (1 hr): Unit tests, smoke runs
7. **Documentation** (1 hr): README, Colab notebook

Total: ~10 hours implementation + experiments

## Risk Mitigation
- **OOM on GPU**: Gradient accumulation, smaller batches
- **Slow convergence**: Increase LR, adjust curriculum
- **Cache thrashing**: Limit K, increase few-shot steps
- **FAISS issues**: Fallback to exact search
- **Colab timeout**: Checkpoint frequently, use Drive

## Success Indicators
- ✅ Loss convergence in training
- ✅ Consistent improvements at low dims
- ✅ Reproducible results across seeds
- ✅ Clean efficiency-quality trade-offs
- ✅ Working Colab demo < 45 minutes
