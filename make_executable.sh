#!/bin/bash
# Make all test scripts executable
chmod +x run_quick_test.sh
chmod +x run_smoke_test.sh
chmod +x run_full_flow.sh

echo "✅ All scripts are now executable!"
echo ""
echo "Available test flows:"
echo "  1. Quick component test (30 sec): ./run_quick_test.sh"
echo "  2. Smoke test (1-2 min):          ./run_smoke_test.sh"
echo "  3. Full training (2-6 hr):        ./run_full_flow.sh"
