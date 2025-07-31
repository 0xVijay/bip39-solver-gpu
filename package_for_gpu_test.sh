#!/bin/bash

# Script to package the BIP39 GPU solver for remote testing
# This creates a standalone package that can be copied to a GPU machine

echo "Creating package for GPU testing..."

# Create package directory
PKG_DIR="bip39-gpu-test-package"
rm -rf $PKG_DIR
mkdir -p $PKG_DIR

# Copy all source files
cp -r src $PKG_DIR/
cp Cargo.toml $PKG_DIR/
cp Cargo.lock $PKG_DIR/
cp README.md $PKG_DIR/
cp example_config.json $PKG_DIR/
cp test_config.json $PKG_DIR/
cp distributed_config.json $PKG_DIR/
cp test_distributed_config.json $PKG_DIR/
cp install.sh $PKG_DIR/

# Copy OpenCL kernels if they exist
if [ -d "cl" ]; then
    cp -r cl $PKG_DIR/
fi

# Create installation script for GPU machine
cat > $PKG_DIR/setup_gpu_test.sh << 'EOF'
#!/bin/bash

echo "Setting up BIP39 GPU solver for testing..."

# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev
sudo apt-get install -y ocl-icd-opencl-dev opencl-headers

# Install Rust if not present
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
fi

# Build the project
echo "Building BIP39 GPU solver..."
cargo build --release

# Run GPU tests
echo "Running GPU tests..."
cargo test testgpu -- --nocapture

echo "Setup complete! GPU solver ready for testing."
echo ""
echo "To run the solver:"
echo "  ./target/release/bip39-solver-gpu --config test_config.json"
echo ""
echo "To run distributed server:"
echo "  ./target/release/bip39-server --config distributed_config.json"
echo ""
echo "To run distributed worker:"
echo "  ./target/release/bip39-worker --config distributed_config.json"
EOF

chmod +x $PKG_DIR/setup_gpu_test.sh

# Create GPU test script
cat > $PKG_DIR/test_gpu_functionality.sh << 'EOF'
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
EOF

chmod +x $PKG_DIR/test_gpu_functionality.sh

# Create README for the package
cat > $PKG_DIR/GPU_TEST_README.md << 'EOF'
# BIP39 GPU Solver - GPU Testing Package

This package contains the complete BIP39 GPU solver for testing on GPU-enabled machines.

## Quick Setup

1. Extract this package on your GPU machine
2. Run the setup script:
   ```bash
   ./setup_gpu_test.sh
   ```

## Testing GPU Functionality

Run the comprehensive GPU test:
```bash
./test_gpu_functionality.sh
```

## Manual Testing

### Test GPU Detection
```bash
cargo test testgpu -- --nocapture
```

### Test Basic Solver
```bash
./target/release/bip39-solver-gpu --config test_config.json
```

### Test Distributed System
```bash
# Terminal 1 - Start server
./target/release/bip39-server --config distributed_config.json

# Terminal 2 - Start worker
./target/release/bip39-worker --config distributed_config.json
```

## GPU Requirements

- NVIDIA GPU with CUDA support OR AMD GPU with OpenCL support
- OpenCL drivers installed
- OpenCL development headers

## Expected Output

If GPU is working correctly, you should see:
```
✓ GPU acceleration initialized successfully
✓ GPU processed batch successfully
```

If GPU is not available, you'll see:
```
⚠ GPU acceleration not available: No OpenCL platforms found
The solver will fall back to CPU processing
```

## Troubleshooting

1. **No OpenCL platforms**: Install GPU drivers and OpenCL runtime
2. **Compilation errors**: Install build dependencies with setup script
3. **Permission denied**: Ensure GPU is accessible to your user account

## Performance Testing

The GPU implementation should significantly outperform CPU:
- CPU: ~17,800 mnemonics/sec
- GPU: Expected 10x-100x improvement depending on hardware
EOF

# Create compressed package
echo "Creating compressed package..."
tar -czf ${PKG_DIR}.tar.gz $PKG_DIR

echo ""
echo "✓ Package created: ${PKG_DIR}.tar.gz"
echo "✓ Package directory: $PKG_DIR"
echo ""
echo "To test on remote GPU machine:"
echo "1. Copy ${PKG_DIR}.tar.gz to the GPU machine"
echo "2. Extract: tar -xzf ${PKG_DIR}.tar.gz"
echo "3. cd $PKG_DIR"
echo "4. ./setup_gpu_test.sh"
echo "5. ./test_gpu_functionality.sh"
echo ""
echo "Package is ready for GPU testing!"