#!/usr/bin/env python3
"""
Standalone script to download arXiv data.
Can be run directly: python download_arxiv_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from temporal_lora.data.download_arxiv import download_and_prepare

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download arXiv data with real API or synthetic fallback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 50K real papers (RECOMMENDED for training)
  python download_arxiv_data.py --max-papers 50000
  
  # Download 100K papers from specific categories
  python download_arxiv_data.py --max-papers 100000 --categories cs.AI,cs.LG,cs.CL
  
  # Download papers from 2015-2024 only
  python download_arxiv_data.py --start-year 2015 --end-year 2024
  
  # Generate synthetic data (for testing)
  python download_arxiv_data.py --synthetic --max-papers 10000
  
  # Custom output directory
  python download_arxiv_data.py --output-dir /path/to/output --max-papers 50000
        """
    )
    
    parser.add_argument(
        "--max-papers",
        type=int,
        default=50000,
        help="Maximum number of papers to download (default: 50000)"
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
        help="Force synthetic data generation (skip real API)"
    )
    
    parser.add_argument(
        "--categories",
        type=str,
        default="cs.AI,cs.LG,cs.CL,cs.CV,cs.NE",
        help="Comma-separated arXiv categories (default: cs.AI,cs.LG,cs.CL,cs.CV,cs.NE)"
    )
    
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="Start year for date filter (default: 2010)"
    )
    
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="End year for date filter (default: 2024)"
    )
    
    args = parser.parse_args()
    
    # Parse categories
    categories = [c.strip() for c in args.categories.split(",")]
    
    print("\n" + "="*60)
    print("🚀 arXiv Data Downloader")
    print("="*60)
    print(f"Mode: {'SYNTHETIC' if args.synthetic else 'REAL API'}")
    print(f"Target: {args.max_papers:,} papers")
    if not args.synthetic:
        print(f"Categories: {', '.join(categories)}")
        print(f"Years: {args.start_year}-{args.end_year}")
    print("="*60 + "\n")
    
    # Download
    output_file = download_and_prepare(
        max_papers=args.max_papers,
        output_dir=args.output_dir,
        use_real_api=not args.synthetic,
        categories=categories,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    
    if output_file and output_file.exists():
        print("\n" + "="*60)
        print("✅ DOWNLOAD COMPLETE!")
        print("="*60)
        print(f"📁 File: {output_file}")
        print(f"📊 Size: {output_file.stat().st_size / (1024*1024):.1f} MB")
        print("\n🚀 Next Steps:")
        print("  1. Preprocess data:")
        print("     python -m temporal_lora.cli prepare-data")
        print("  2. Train LoRA adapters:")
        print("     python -m temporal_lora.cli train-adapters --mode lora --epochs 2")
        print("  3. Build indexes:")
        print("     python -m temporal_lora.cli build-indexes")
        print("  4. Evaluate:")
        print("     python -m temporal_lora.cli evaluate")
        print("="*60)
    else:
        print("\n❌ Download failed!")
        sys.exit(1)
