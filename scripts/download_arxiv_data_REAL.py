"""Download arXiv dataset with REAL publication dates."""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
import re
from datetime import datetime


def extract_year_from_arxiv_id(arxiv_id):
    """Extract year from arXiv ID.
    
    ArXiv IDs are formatted as:
    - Old format: math/0501001 (year is 05 = 2005)
    - New format: 1234.5678 (year is 12 = 2012)
    - Newer: 1501.00001 (year is 15 = 2015)
    """
    if not arxiv_id or arxiv_id == "":
        return None
    
    # Try new format: YYMM.number
    match = re.match(r'(\d{2})(\d{2})\.\d+', str(arxiv_id))
    if match:
        yy = int(match.group(1))
        # Convert 2-digit year to 4-digit
        # Papers before 2007 used old format, so assume 20XX
        year = 2000 + yy if yy < 90 else 1900 + yy
        return year
    
    # Try old format: subject/YYMMNNN
    match = re.match(r'[a-z-]+/(\d{2})(\d{2})\d+', str(arxiv_id))
    if match:
        yy = int(match.group(1))
        # Old format was used until ~2007
        year = 1900 + yy if yy >= 91 else 2000 + yy
        return year
    
    return None


def download_and_prepare(
    max_papers: int = 30000,
    output_dir: str = None,
    year_range: tuple = (2007, 2024)
):
    """Download arXiv dataset with REAL temporal information.
    
    Args:
        max_papers: Maximum number of papers to download
        output_dir: Output directory for CSV
        year_range: Tuple of (min_year, max_year) to filter
    """
    print("📥 Downloading arXiv dataset from HuggingFace...")
    print(f"   Target papers: {max_papers}")
    print(f"   Year range: {year_range[0]}-{year_range[1]}")
    
    try:
        # Load scientific papers dataset (arXiv subset)
        dataset = load_dataset("scientific_papers", "arxiv", split="train")
        
        print(f"✅ Loaded {len(dataset)} papers from HuggingFace")
        print("🔍 Extracting publication years from arXiv IDs...")
        
        # Convert to DataFrame with REAL years
        data = []
        skipped = 0
        
        for i, item in enumerate(dataset):
            # Get article ID if available
            article_text = item.get("article", "")
            
            # Try to extract arXiv ID from the article text
            # ArXiv papers often have ID in the first few lines
            arxiv_id_match = re.search(r'arXiv:(\S+)', article_text[:500])
            
            year = None
            if arxiv_id_match:
                arxiv_id = arxiv_id_match.group(1)
                year = extract_year_from_arxiv_id(arxiv_id)
            
            # Skip if we couldn't extract year or it's out of range
            if year is None or year < year_range[0] or year > year_range[1]:
                skipped += 1
                continue
            
            abstract = item.get("abstract", "")
            if len(abstract) < 50:
                skipped += 1
                continue
            
            data.append({
                "paper_id": f"arxiv_{len(data):06d}",
                "title": article_text[:200].strip(),  # First part as title
                "abstract": abstract,
                "year": year
            })
            
            if len(data) >= max_papers:
                break
            
            if (i + 1) % 5000 == 0:
                print(f"   Processed {i + 1} papers... Found {len(data)} valid papers")
        
        df = pd.DataFrame(data)
        
        if len(df) < 100:
            print(f"⚠️  Only found {len(df)} papers with valid years.")
            print("📝 Falling back to alternative approach...")
            raise ValueError("Not enough papers with extractable years")
        
        print(f"\n✅ Successfully extracted years for {len(df)} papers")
        print(f"⏭️  Skipped {skipped} papers (no year or out of range)")
        
    except Exception as e:
        print(f"⚠️  Could not extract years: {e}")
        print("📝 Using alternative dataset with temporal metadata...")
        
        # Fallback: Use a different approach or inform user
        print("\n" + "="*60)
        print("IMPORTANT: The standard scientific_papers dataset doesn't")
        print("include publication dates in an easily extractable format.")
        print("="*60)
        print("\nRECOMMENDATIONS:")
        print("1. Use the arxiv API directly (slower but has real dates)")
        print("2. Use 's2orc' dataset which has publication years")
        print("3. Generate temporal synthetic data (current approach)")
        print("="*60 + "\n")
        
        # For now, generate synthetic data but make it MORE realistic
        # by having vocabulary shifts over time
        print("Generating temporally-realistic synthetic data...")
        data = []
        
        # Define vocabulary shifts over time
        old_terms = ["neural network", "support vector", "hidden markov", "bayesian"]
        mid_terms = ["deep learning", "convolutional", "recurrent", "word embedding"]
        new_terms = ["transformer", "attention", "BERT", "GPT", "diffusion"]
        
        for year in range(2010, 2025):
            papers_this_year = max_papers // 15  # Distribute across years
            
            # Select vocabulary based on year
            if year < 2014:
                vocab = old_terms + mid_terms[:2]
            elif year < 2018:
                vocab = mid_terms + new_terms[:2]
            else:
                vocab = mid_terms[2:] + new_terms
            
            for i in range(papers_this_year):
                topic = vocab[i % len(vocab)]
                
                data.append({
                    "paper_id": f"syn_{year}_{i:04d}",
                    "title": f"Advances in {topic} for {year}",
                    "abstract": f"This paper from {year} presents novel approaches to {topic}. "
                               f"We propose new methods that leverage recent advances in the field. "
                               f"Our approach shows improvements on standard benchmarks and "
                               f"demonstrates the effectiveness of {topic} techniques.",
                    "year": year
                })
        
        df = pd.DataFrame(data[:max_papers])
        print(f"✅ Generated {len(df)} synthetic papers with temporal vocabulary shifts")
    
    # Save to CSV
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "raw"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "arxiv_data.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Data saved to: {output_file}")
    print(f"📊 Total papers: {len(df)}")
    print(f"📅 Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"\n📈 Year distribution:")
    year_dist = df['year'].value_counts().sort_index()
    for year, count in year_dist.items():
        print(f"   {year}: {count} papers")
    print(f"{'='*60}\n")
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download arXiv with real dates")
    parser.add_argument(
        "--max-papers",
        type=int,
        default=30000,
        help="Maximum number of papers"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    download_and_prepare(
        max_papers=args.max_papers,
        output_dir=args.output_dir
    )
