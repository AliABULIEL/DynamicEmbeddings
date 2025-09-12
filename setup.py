"""
Setup script for the project
"""
from setuptools import setup, find_packages

setup(
    name="domain-embedding-composition",
    version="1.0.0",
    author="Your Name",
    description="Dynamic Domain-Based Embedding Composition for NLP Tasks",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.2",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "datasets>=2.14.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.66.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)