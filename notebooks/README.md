# Notebooks

This directory contains Jupyter notebooks for exploratory analysis and demonstrations.

## Available Notebooks

### `demo.ipynb` (Coming Soon)
Interactive demonstration of temporal LoRA retrieval on a small subset of arXiv abstracts.

**Contents:**
- Load pre-trained temporal adapters
- Encode queries and documents from different time periods
- Visualize embedding spaces with UMAP
- Compare retrieval performance across scenarios
- Interactive term drift exploration

**Usage:**
```bash
jupyter notebook demo.ipynb
```

**Note:** Notebooks use small data subsets (≤1000 samples) for fast execution. For full-scale experiments, use the CLI commands.

---

## Development Guidelines

1. **Keep notebooks small**: Focus on visualization and exploration, not training
2. **Use cached data**: Load pre-computed embeddings and indexes when possible
3. **Document cells**: Add markdown explanations for each analysis step
4. **Fixed seeds**: Always set random seeds for reproducibility
5. **Clear outputs**: Clear all outputs before committing

---

## Colab Support

To run notebooks in Google Colab:

1. Upload the notebook to Colab
2. Install dependencies:
   ```python
   !pip install -q temporal-lora
   ```
3. Mount Google Drive if needed:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

---

## Visualization Examples

Planned visualizations:
- **Embedding space evolution**: UMAP projections colored by time bucket
- **Cross-period similarity**: Heatmaps showing query-document similarities
- **Term drift trajectories**: Track specific terms across time buckets
- **Attention patterns**: Visualize LoRA adapter attention weights

---

## Contact

For notebook issues or requests, open a GitHub issue.
