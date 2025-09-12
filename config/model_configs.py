"""
Updated model configurations with better models
"""

# Domain-specific models (updated with better alternatives where available)
DOMAIN_MODELS = {
    'scientific': {
        'model_name': 'allenai/scibert_scivocab_uncased',
        'type': 'transformer',
        'description': 'SciBERT for scientific text',
        'alternative': 'allenai/specter2'  # Better alternative
    },
    'news': {
        'model_name': 'sentence-transformers/all-mpnet-base-v2',
        'type': 'sentence-transformer',
        'description': 'MPNet as news proxy (best performing)'
    },
    'medical': {
        'model_name': 'dmis-lab/biobert-v1.1',
        'type': 'transformer',
        'description': 'BioBERT for biomedical text',
        'alternative': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    },
    'legal': {
        'model_name': 'nlpaueb/legal-bert-base-uncased',
        'type': 'transformer',
        'description': 'Legal-BERT for legal documents',
        'alternative': 'lexlms/legal-xlm-roberta-base'
    },
    'social': {
        'model_name': 'vinai/bertweet-base',
        'type': 'transformer',
        'description': 'BERTweet for social media',
        'use_with_caution': True  # Problematic with non-social text
    }
}

# Zero-shot classification models
ZERO_SHOT_MODELS = {
    'default': 'facebook/bart-large-mnli',
    'backup': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
    'large': 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli'  # More accurate
}

# Baseline models for comparison (updated)
BASELINE_MODELS = {
    'bert-base': 'sentence-transformers/bert-base-nli-mean-tokens',
    'roberta': 'sentence-transformers/roberta-base-nli-mean-tokens',
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'distilbert': 'sentence-transformers/distilbert-base-nli-mean-tokens'
}

# Advanced models for stronger baselines
ADVANCED_MODELS = {
    'e5-large': 'intfloat/e5-large-v2',
    'gte-large': 'thenlper/gte-large',
    'bge-large': 'BAAI/bge-large-en-v1.5',
    'instructor': 'hkunlp/instructor-large'
}