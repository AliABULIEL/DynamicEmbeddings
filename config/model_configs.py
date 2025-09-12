"""
Model configurations and HuggingFace model identifiers
"""

# Domain-specific models
DOMAIN_MODELS = {
    'scientific': {
        'model_name': 'allenai/scibert_scivocab_uncased',
        'type': 'transformer',  # or 'sentence-transformer'
        'description': 'SciBERT for scientific text'
    },
    'news': {
        'model_name': 'sentence-transformers/all-mpnet-base-v2',
        'type': 'sentence-transformer',
        'description': 'General model as news proxy'
    },
    'medical': {
        'model_name': 'dmis-lab/biobert-v1.1',
        'type': 'transformer',
        'description': 'BioBERT for biomedical text'
    },
    'legal': {
        'model_name': 'nlpaueb/legal-bert-base-uncased',
        'type': 'transformer',
        'description': 'Legal-BERT for legal documents'
    },
    'social': {
        'model_name': 'vinai/bertweet-base',
        'type': 'transformer',
        'description': 'BERTweet for social media'
    }
}

# Zero-shot classification models
ZERO_SHOT_MODELS = {
    'default': 'facebook/bart-large-mnli',
    'backup': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
}

# Baseline models for comparison
BASELINE_MODELS = {
    'bert-base': 'sentence-transformers/bert-base-nli-mean-tokens',
    'roberta': 'sentence-transformers/roberta-base-nli-mean-tokens',
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2'
}