# Data Sources and Licenses

## Primary Dataset: arXiv CS/ML Abstracts

### Hugging Face Dataset
- **Name**: `arxiv-metadata-oai-snapshot`
- **Link**: [https://huggingface.co/datasets/arxiv-metadata](https://huggingface.co/datasets/arxiv-metadata)
- **Subset**: Computer Science (cs.*) and Machine Learning (stat.ML) categories
- **Fields Used**: `title`, `abstract`, `year` (derived from `update_date` or `published`)
- **License**: [CC0 1.0 Universal (Public Domain)](https://creativecommons.org/publicdomain/zero/1.0/)
- **Size**: ~2M+ papers (filtered to CS/ML)

### Access Instructions
```python
from datasets import load_dataset

# Load full arXiv metadata
dataset = load_dataset("arxiv-metadata", split="train")

# Filter to CS/ML categories
cs_ml = dataset.filter(lambda x: any(cat.startswith("cs.") or cat == "stat.ML" 
                                      for cat in x["categories"].split()))
```

### Preprocessing
Our pipeline:
1. Filters papers with `cs.*` or `stat.ML` categories
2. Extracts year from `update_date` (YYYY-MM-DD format)
3. Removes papers with missing `title` or `abstract`
4. Stratifies by time buckets (≤2018, 2019-2024)
5. Samples up to `max_per_bucket` papers per bucket (default: 6000)

## CSV Fallback Option

If Hugging Face Datasets are unavailable (e.g., restricted networks), we provide a CSV with the same schema:

### Schema
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `title` | str | Paper title | "Attention Is All You Need" |
| `abstract` | str | Paper abstract (first 512 chars) | "The dominant sequence transduction models..." |
| `year` | int | Publication year | 2017 |

### Sample CSV
```csv
title,abstract,year
"BERT: Pre-training of Deep Bidirectional Transformers","We introduce a new language representation model...",2019
"Temporal Embeddings for Dynamic Graphs","We propose a method for learning time-aware...",2020
```

### Generating CSV from HF Dataset
```bash
python -m temporal_lora.cli prepare-data --output_format csv --output_path data/arxiv_cs_ml.csv
```

## License Summary

### Code (This Repository)
- **License**: MIT License
- **File**: [LICENSE](LICENSE)
- **Summary**: Permissive; allows commercial use, modification, distribution

### Data (arXiv Abstracts)
- **License**: CC0 1.0 Universal (Public Domain Dedication)
- **Attribution**: While not legally required, we acknowledge arXiv.org as the source
- **Citation**: See [arXiv documentation](https://arxiv.org/help/bulk_data)

### Third-Party Models
- **sentence-transformers/all-MiniLM-L6-v2**: Apache 2.0
- **PEFT (Hugging Face)**: Apache 2.0
- **FAISS**: MIT License

## Ethical Considerations

### Data Selection Bias
- arXiv is English-dominant and skewed toward certain subfields
- CS/ML papers may over-represent institutional research
- Temporal trends reflect publication patterns, not all research activity

### Privacy
- Abstracts are publicly available; no PII concerns
- Author names not used in this project

### Intended Use
This dataset is for **research purposes only**:
- Evaluating temporal embedding methods
- Studying semantic drift in scientific language
- Benchmarking information retrieval systems

**Not intended for**:
- Production search engines without validation
- Decision-making about research quality or impact
- Automated paper summarization without human review

## Updates and Versioning

- **Dataset Version**: Snapshot as of January 2025
- **Code Version**: See [CITATION.cff](CITATION.cff)
- **Updates**: arXiv metadata updates daily; re-run `prepare-data` for latest papers

## Contact

For questions about data usage or licensing, please open an issue in the repository.
