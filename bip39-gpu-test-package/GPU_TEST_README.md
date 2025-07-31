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
