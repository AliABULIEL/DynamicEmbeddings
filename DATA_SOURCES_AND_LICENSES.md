# Data Sources and Licenses

This document details all data sources used in the Temporal LoRA project, their licenses, and fallback options.

---

## Primary Data Source

### arXiv CS/ML Abstracts

**Source**: [Hugging Face Dataset](https://huggingface.co/datasets/arxiv-cs-ml)

**Description**: Academic paper abstracts from arXiv Computer Science and Machine Learning categories.

**License**: **CC0 1.0 Universal (Public Domain)**
- You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.
- See full license: https://creativecommons.org/publicdomain/zero/1.0/

**Coverage**:
- **Years**: 2010-2025 (continuously updated)
- **Categories**: cs.AI, cs.CL, cs.CV, cs.LG, cs.NE, stat.ML
- **Size**: ~500k papers (varies over time)

**Schema**:
```json
{
  "paper_id": "2301.12345",
  "title": "Attention Is All You Need",
  "abstract": "The dominant sequence transduction models...",
  "year": 2023,
  "categories": ["cs.LG", "cs.AI"],
  "authors": ["Author A", "Author B"],
  "published": "2023-01-15"
}
```

**Required Fields** (used by our pipeline):
- `paper_id` (str): Unique identifier
- `title` (str): Paper title
- `abstract` (str): Abstract text
- `year` (int): Publication year

**Access**:
```python
from datasets import load_dataset

dataset = load_dataset("arxiv-cs-ml")
# Or specify year range
dataset = load_dataset("arxiv-cs-ml", split="train[2010:2025]")
```

**Citation**:
```bibtex
@misc{arxiv_dataset,
  title={arXiv Dataset},
  author={arXiv},
  year={2025},
  url={https://arxiv.org}
}
```

---

## Fallback: CSV Format

If the Hugging Face dataset is unavailable or you want to use custom data, provide a CSV with the following schema:

### CSV Schema

**Required Columns**:
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `paper_id` | str | Unique paper identifier | `"2301.12345"` |
| `title` | str | Paper title | `"Attention Is All You Need"` |
| `abstract` | str | Paper abstract (full text) | `"The dominant sequence..."` |
| `year` | int | Publication year | `2023` |

**Optional Columns** (ignored by pipeline):
- `authors`, `categories`, `published`, `url`, etc.

**Example CSV**:
```csv
paper_id,title,abstract,year
2301.12345,"Attention Is All You Need","The dominant sequence transduction models are based on...",2023
2301.67890,"BERT: Pre-training of Deep Bidirectional Transformers","We introduce a new language representation model...",2018
```

**Usage**:
```bash
# Place CSV in data/raw/arxiv_custom.csv
python -m temporal_lora.cli prepare-data --source csv --csv_path data/raw/arxiv_custom.csv
```

---

## Data Processing Pipeline

Our pipeline performs:

1. **Loading**: HF datasets or CSV
2. **Filtering**:
   - Remove papers with missing `abstract` or `title`
   - Remove abstracts shorter than 50 characters
   - Remove years outside 2010-2025
3. **Bucketing**: Assign papers to time buckets (default: ≤2018, 2019-2024)
4. **Sampling**: Max N samples per bucket (default: 6000) for balanced training
5. **Splitting**: 80% train, 10% val, 10% test (stratified by bucket)
6. **Caching**: Save processed data to `data/processed/`

**Output Files**:
```
data/processed/
├── train.jsonl        # Training samples
├── val.jsonl          # Validation samples
├── test.jsonl         # Test samples
├── metadata.json      # Stats (counts per bucket, splits)
└── bucket_config.json # Time bucket definitions
```

---

## Privacy & Ethics

### No Personal Data
- All papers are publicly available research publications
- Author names (if present) are not used in embeddings or retrieval
- No email addresses, affiliations, or personal identifiers are processed

### Fair Use
- Academic abstracts are factual summaries (not creative works)
- Used for non-commercial research and education
- Transformative use: creating temporal embeddings, not redistributing abstracts

### Bias Considerations
- arXiv skews toward English-language, Western academic publications
- CS/ML categories over-represent certain research areas
- Temporal distribution reflects submission trends (exponential growth in ML papers)

### Responsible Use
- This project is for **research and educational purposes**
- Do not use embeddings to:
  - Identify paper authors without consent
  - Make hiring/funding decisions without human review
  - Generate fake paper abstracts or plagiarize content

---

## Custom Data Sources

You can use any corpus with the required schema:

### Supported Formats
- **HuggingFace datasets**: Any dataset with `title`, `abstract`, `year` columns
- **CSV**: See schema above
- **JSONL**: One JSON object per line with required fields

### Preparing Custom Data

```python
import pandas as pd

# Example: Convert your corpus to the required format
df = pd.DataFrame({
    'paper_id': ['id1', 'id2', ...],
    'title': ['Title 1', 'Title 2', ...],
    'abstract': ['Abstract 1', 'Abstract 2', ...],
    'year': [2020, 2021, ...]
})

df.to_csv('data/raw/custom_corpus.csv', index=False)
```

Then run:
```bash
python -m temporal_lora.cli prepare-data \
  --source csv \
  --csv_path data/raw/custom_corpus.csv \
  --max_per_bucket 5000
```

---

## License Compliance

### Our Code (MIT License)
- You can use, modify, and distribute our code
- Attribution appreciated but not required
- See [LICENSE](LICENSE) for details

### arXiv Data (CC0 Public Domain)
- No restrictions on use
- No attribution legally required (but ethically encouraged)
- Our pipeline includes `DATA_SOURCES_AND_LICENSES.md` in all exports

### Dependencies
All Python packages are open-source (MIT, Apache 2.0, BSD):
- `sentence-transformers`: Apache 2.0
- `peft`: Apache 2.0
- `faiss`: MIT
- `transformers`: Apache 2.0
- See `requirements.txt` for full list

---

## Questions?

For data-related questions:
- **arXiv access**: See https://info.arxiv.org/help/api/index.html
- **Custom data**: Open an issue with your use case
- **License concerns**: Email aliab@example.com

---

## Changelog

- **2025-01-15**: Initial data documentation
- **TBD**: Add support for PubMed/bioRxiv biomedical papers
