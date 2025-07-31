#!/bin/bash

echo "=== BIP39 GPU Solver - GPU Functionality Test ==="
echo ""

# Test 1: Basic GPU test
echo "1. Running testgpu function..."
cargo test testgpu -- --nocapture
echo ""

# Test 2: Check GPU information
echo "2. Checking GPU/OpenCL information..."
if command -v clinfo &> /dev/null; then
    echo "OpenCL devices detected:"
    clinfo -l
else
    echo "clinfo not available. Installing..."
    sudo apt-get install -y clinfo
    echo "OpenCL devices detected:"
    clinfo -l 2>/dev/null || echo "No OpenCL devices found"
fi
echo ""

# Test 3: Run solver with test config (small batch)
echo "3. Running solver with test configuration (small batch)..."
if [ -f "test_config.json" ]; then
    timeout 60 ./target/release/bip39-solver-gpu --config test_config.json || echo "Test completed (timeout expected)"
else
    echo "test_config.json not found"
fi
echo ""

# Test 4: Test all binaries exist
echo "4. Checking all binaries..."
ls -la target/release/bip39-*
echo ""

echo "=== GPU Test Complete ==="
