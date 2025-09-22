"""Setup script for TIDE-Lite."""

from setuptools import setup, find_packages

setup(
    name="tide-lite",
    version="0.1.0",
    description="Temporally-Indexed Dynamic Embeddings",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.1",
        "transformers>=4.44",
        "datasets>=2.20",
        "sentence-transformers>=3.0",
        "faiss-cpu>=1.7.4",
        "scikit-learn>=1.4",
        "numpy>=1.26",
        "pandas>=2.0",
        "tqdm>=4.66",
        "matplotlib>=3.8",
        "peft>=0.12",
        "scipy>=1.10",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "tide=tide_lite.cli.tide:main",
            "tide-train=tide_lite.cli.train_cli:main",
            "tide-eval-stsb=tide_lite.cli.eval_stsb_cli:main",
            "tide-eval-quora=tide_lite.cli.eval_quora_cli:main",
            "tide-eval-temporal=tide_lite.cli.eval_temporal_cli:main",
            "tide-aggregate=tide_lite.cli.aggregate_cli:main",
            "tide-plots=tide_lite.cli.plots_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
