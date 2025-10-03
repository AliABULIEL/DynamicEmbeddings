#!/bin/bash
# Quick test script for download functionality

echo "🧪 Testing arXiv Download Script"
echo "================================"
echo ""

# Test 1: Synthetic data (fast)
echo "Test 1: Synthetic data generation (1000 papers)"
echo "------------------------------------------------"
python3 download_arxiv_data.py --synthetic --max-papers 1000
if [ $? -eq 0 ]; then
    echo "✅ Test 1 PASSED"
else
    echo "❌ Test 1 FAILED"
    exit 1
fi

echo ""
echo "================================"
echo ""

# Test 2: Check output file
echo "Test 2: Verify output file"
echo "------------------------------------------------"
if [ -f "data/raw/arxiv_data.csv" ]; then
    echo "✅ Output file exists: data/raw/arxiv_data.csv"
    echo ""
    echo "File stats:"
    wc -l data/raw/arxiv_data.csv
    du -h data/raw/arxiv_data.csv
    echo ""
    echo "First 5 rows:"
    head -n 5 data/raw/arxiv_data.csv
    echo ""
    echo "✅ Test 2 PASSED"
else
    echo "❌ Test 2 FAILED: Output file not found"
    exit 1
fi

echo ""
echo "================================"
echo "✅ ALL TESTS PASSED!"
echo "================================"
echo ""
echo "🎉 Download script is working!"
echo ""
echo "Next: Try real data download:"
echo "  python download_arxiv_data.py --max-papers 10000"
echo ""
