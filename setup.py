"""Setup configuration for Dynamic Embeddings package."""

from setuptools import setup, find_packages

setup(
    name="dynamic-embeddings",
    version="1.0.0",
    description="State-of-the-art dynamic embeddings implementation",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#")
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
    },
)