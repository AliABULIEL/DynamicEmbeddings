"""
Download and prepare arXiv dataset with REAL data from arXiv API.

Supports both real API and synthetic fallback for testing.
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd
from datetime import datetime
import time
import sys


def download_real_arxiv(
    categories: List[str] = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE"],
    max_papers: int = 50000,
    start_year: int = 2010,
    end_year: int = 2024,
    output_dir: Optional[str] = None,
    batch_size: int = 1000,
) -> Path:
    """
    Download REAL arXiv papers using the official arXiv API.
    
    Args:
        categories: arXiv categories to download (cs.AI, cs.LG, etc.)
        max_papers: Maximum number of papers to download
        start_year: Start year for date filter
        end_year: End year for date filter
        output_dir: Output directory (defaults to data/raw/)
        batch_size: Papers per API request
        
    Returns:
        Path to saved CSV file
    """
    try:
        import arxiv
    except ImportError:
        print("❌ arxiv package not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "arxiv"])
        import arxiv
    
    print("="*60)
    print("📥 Downloading REAL arXiv Data")
    print("="*60)
    print(f"Categories: {', '.join(categories)}")
    print(f"Date range: {start_year}-{end_year}")
    print(f"Target: {max_papers:,} papers")
    print(f"Batch size: {batch_size}")
    print("="*60 + "\n")
    
    # Build query
    category_query = " OR ".join([f"cat:{cat}" for cat in categories])
    
    # Configure client
    client = arxiv.Client(
        page_size=batch_size,
        delay_seconds=3.0,  # Rate limiting
        num_retries=3
    )
    
    # Create search
    search = arxiv.Search(
        query=category_query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers_data = []
    skipped = 0
    
    print("🔍 Fetching papers from arXiv API...")
    print("(This may take a while - arXiv rate limits to ~1 request/3 seconds)\n")
    
    start_time = time.time()
    
    for i, paper in enumerate(client.results(search), 1):
        # Extract year from published date
        year = paper.published.year
        
        # Filter by year range
        if year < start_year or year > end_year:
            skipped += 1
            continue
        
        # Extract paper info
        papers_data.append({
            "paper_id": paper.entry_id.split("/")[-1],  # arXiv ID
            "title": paper.title,
            "abstract": paper.summary,
            "year": year,
            "published_date": paper.published.strftime("%Y-%m-%d"),
            "categories": ", ".join(paper.categories),
            "authors": ", ".join([author.name for author in paper.authors[:5]]),  # First 5 authors
        })
        
        # Progress update
        if i % 100 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (max_papers - i) / rate if rate > 0 else 0
            print(f"  Progress: {len(papers_data):,}/{max_papers:,} papers "
                  f"({i:,} fetched, {skipped:,} skipped) | "
                  f"Rate: {rate:.1f} papers/sec | "
                  f"ETA: {eta/60:.1f} min")
        
        # Stop if we have enough papers
        if len(papers_data) >= max_papers:
            break
    
    elapsed = time.time() - start_time
    
    if len(papers_data) == 0:
        print("\n❌ No papers found matching criteria!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(papers_data)
    
    # Sort by year then date
    df = df.sort_values(["year", "published_date"]).reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print(f"✅ Downloaded {len(df):,} papers in {elapsed/60:.1f} minutes")
    print(f"📅 Year range: {df['year'].min()} - {df['year'].max()}")
    
    # Year distribution
    print(f"\n📈 Year distribution:")
    year_dist = df['year'].value_counts().sort_index()
    for year, count in year_dist.items():
        bar = "█" * int(count / year_dist.max() * 40)
        print(f"   {year}: {count:>5,} {bar}")
    
    # Category distribution
    print(f"\n📚 Top categories:")
    all_cats = []
    for cats in df['categories'].str.split(", "):
        all_cats.extend(cats)
    cat_series = pd.Series(all_cats)
    for cat, count in cat_series.value_counts().head(10).items():
        print(f"   {cat}: {count:,}")
    
    print(f"{'='*60}\n")
    
    # Save to CSV
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "arxiv_data.csv"
    
    df.to_csv(output_file, index=False)
    print(f"💾 Data saved to: {output_file}\n")
    
    return output_file


def download_synthetic(
    max_papers: int = 30000,
    output_dir: Optional[str] = None
) -> Path:
    """
    Generate synthetic arXiv data with temporal vocabulary shifts.
    Fallback for testing without API access.
    
    Args:
        max_papers: Number of synthetic papers to generate
        output_dir: Output directory
        
    Returns:
        Path to saved CSV file
    """
    print("="*60)
    print("🧪 Generating Synthetic Data")
    print("="*60)
    print(f"Papers: {max_papers:,}")
    print("="*60 + "\n")
    
    # Define vocabulary evolution to simulate semantic drift
    old_terms = [
        "neural networks", "support vector machines", "decision trees",
        "k-means clustering", "naive bayes", "ensemble methods",
        "logistic regression", "random forests", "hidden markov models"
    ]
    
    mid_terms = [
        "deep learning", "convolutional networks", "recurrent networks",
        "word embeddings", "attention mechanisms", "transfer learning",
        "residual networks", "batch normalization", "dropout"
    ]
    
    new_terms = [
        "transformers", "BERT", "GPT", "large language models",
        "diffusion models", "vision transformers", "self-supervised learning",
        "few-shot learning", "prompt engineering", "in-context learning",
        "retrieval augmented generation", "reinforcement learning from human feedback"
    ]
    
    data = []
    
    # Realistic growth pattern (exponential)
    years = list(range(2010, 2025))
    year_weights = [1.0 * (1.15 ** i) for i in range(len(years))]
    total_weight = sum(year_weights)
    
    print("📝 Generating papers with temporal vocabulary shifts...")
    
    for year, weight in zip(years, year_weights):
        papers_this_year = int((weight / total_weight) * max_papers)
        
        # Select vocabulary based on year
        if year < 2014:
            vocab = old_terms
            era = "classical ML"
        elif year < 2018:
            vocab = old_terms[-3:] + mid_terms
            era = "deep learning"
        else:
            vocab = mid_terms[-3:] + new_terms
            era = "modern AI"
        
        for i in range(papers_this_year):
            topic = vocab[i % len(vocab)]
            
            # Generate contextually appropriate abstract
            if year < 2014:
                abstract = (
                    f"This paper presents novel approaches to {topic} using classical machine learning techniques. "
                    f"We develop algorithms that demonstrate improved performance on benchmark datasets. "
                    f"Experimental results show that our {topic}-based method achieves competitive accuracy "
                    f"while maintaining computational efficiency. Our contributions advance the field of {topic} "
                    f"through rigorous mathematical analysis and extensive empirical validation."
                )
            elif year < 2018:
                abstract = (
                    f"We introduce a deep learning framework for {topic} that leverages modern neural architectures. "
                    f"Our approach combines {topic} with end-to-end training and achieves state-of-the-art results "
                    f"on multiple benchmarks. Through careful architectural design and optimization strategies, "
                    f"we demonstrate that {topic} can be scaled effectively. Ablation studies confirm the importance "
                    f"of our proposed components in achieving superior performance."
                )
            else:
                abstract = (
                    f"We present a novel {topic} method that scales to large datasets through efficient pretraining. "
                    f"Building on recent advances in {topic}, our approach demonstrates remarkable generalization "
                    f"and few-shot learning capabilities. We achieve new benchmarks across diverse tasks, showing "
                    f"that {topic} benefits from scale and appropriate inductive biases. Our findings contribute "
                    f"to understanding how {topic} can be deployed in practical applications at scale."
                )
            
            data.append({
                "paper_id": f"synthetic_{year}_{i:05d}",
                "title": f"Advances in {topic.title()} ({year})",
                "abstract": abstract,
                "year": year,
                "published_date": f"{year}-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "categories": f"cs.AI, cs.LG",
                "authors": f"Author {i % 10 + 1}, Collaborator {(i + 1) % 10 + 1}"
            })
    
    df = pd.DataFrame(data[:max_papers])
    df = df.sort_values(["year", "published_date"]).reset_index(drop=True)
    
    print(f"✅ Generated {len(df):,} synthetic papers")
    print(f"📅 Year range: {df['year'].min()} - {df['year'].max()}")
    
    # Year distribution
    print(f"\n📈 Year distribution:")
    year_dist = df['year'].value_counts().sort_index()
    for year, count in year_dist.items():
        bar = "█" * int(count / year_dist.max() * 40)
        print(f"   {year}: {count:>5,} {bar}")
    
    # Vocabulary distribution
    print(f"\n📚 Vocabulary distribution (top 10):")
    vocab_counts = {}
    for _, row in df.iterrows():
        for term in old_terms + mid_terms + new_terms:
            if term.lower() in row['title'].lower() or term.lower() in row['abstract'].lower():
                vocab_counts[term] = vocab_counts.get(term, 0) + 1
    
    for term, count in sorted(vocab_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"   {term}: {count:,}")
    
    print()
    
    # Save to CSV
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "arxiv_data.csv"
    
    df.to_csv(output_file, index=False)
    print(f"💾 Data saved to: {output_file}\n")
    
    return output_file


def download_and_prepare(
    max_papers: int = 30000,
    output_dir: Optional[str] = None,
    use_real_api: bool = True,
    categories: Optional[List[str]] = None,
    start_year: int = 2010,
    end_year: int = 2024,
) -> Path:
    """
    Download arXiv data - tries real API first, falls back to synthetic.
    
    Args:
        max_papers: Maximum number of papers
        output_dir: Output directory (default: data/raw/)
        use_real_api: Try to use real arXiv API
        categories: arXiv categories for real API
        start_year: Start year filter
        end_year: End year filter
        
    Returns:
        Path to saved CSV file
    """
    if use_real_api:
        try:
            print("🌐 Attempting to download REAL arXiv data...\n")
            
            if categories is None:
                categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE"]
            
            return download_real_arxiv(
                categories=categories,
                max_papers=max_papers,
                start_year=start_year,
                end_year=end_year,
                output_dir=output_dir,
                batch_size=1000,
            )
        except Exception as e:
            print(f"\n⚠️  Real API failed: {e}")
            print("📝 Falling back to synthetic data...\n")
            time.sleep(1)
    
    return download_synthetic(max_papers=max_papers, output_dir=output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download arXiv data (real or synthetic)"
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=50000,
        help="Maximum number of papers"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/raw/)"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic data generation"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="cs.AI,cs.LG,cs.CL,cs.CV,cs.NE",
        help="Comma-separated arXiv categories"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="Start year"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="End year"
    )
    
    args = parser.parse_args()
    
    categories = [c.strip() for c in args.categories.split(",")]
    
    output_file = download_and_prepare(
        max_papers=args.max_papers,
        output_dir=args.output_dir,
        use_real_api=not args.synthetic,
        categories=categories,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    
    if output_file:
        print("\n" + "="*60)
        print("🎉 SUCCESS!")
        print("="*60)
        print(f"📁 Data ready at: {output_file}")
        print("\n🚀 Next steps:")
        print("  1. python -m temporal_lora.cli prepare-data")
        print("  2. python -m temporal_lora.cli train-adapters --mode lora")
        print("="*60)
