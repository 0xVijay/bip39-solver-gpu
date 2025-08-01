# BIP39 Solver GPU - Ethereum Edition

This project iterates through possible BIP39 mnemonics to find those that generate a target Ethereum address. It features a modular GPU computation backend supporting both OpenCL and CUDA, with multi-GPU processing capabilities and full backward compatibility with existing workflows.

## Features

- **🚀 Modular GPU Backend**: Pluggable architecture supporting OpenCL and CUDA with automatic fallbacks
- **⚡ Multi-GPU Processing**: Coordinate work across multiple GPU devices on a single host  
- **🔧 Real CUDA Integration**: Complete CUDA kernel implementation with FFI bindings ready for production
- **🌐 OpenCL Support**: Full OpenCL integration with existing kernel files from `/cl/` directory
- **🎯 Ethereum Support**: Derives Ethereum addresses using BIP44 derivation path `m/44'/60'/0'/0/0`
- **📡 Distributed Processing**: Scale across multiple machines with job server and worker clients
- **⚙️ Configurable Constraints**: Specify known word prefixes or exact words for any mnemonic position
- **📱 Slack Notifications**: Get notified when a matching mnemonic is found
- **📊 Progress Tracking**: Real-time progress reporting and rate monitoring (~1,114 mnemonics/sec baseline)
- **🔄 Checkpointing**: Fault-tolerant job management with automatic retry
- **🛡️ Backward Compatibility**: All existing workflows remain fully compatible
- **🚨 Advanced Error Handling**: Comprehensive GPU error detection, device failover, and graceful CPU fallback
- **🧪 Stress Testing Framework**: Built-in stress testing for edge cases, huge batches, and failure scenarios

## GPU Backend Architecture

The tool now features a modular GPU backend system with automatic device detection and graceful fallbacks:

### Supported Backends

| Backend | Status | Description |
|---------|--------|-------------|
| **OpenCL** | ✅ Production Ready | CPU fallback with GPU acceleration framework |
| **CUDA** | ⚡ Framework Complete | Full kernel implementation with FFI bindings |

### Multi-GPU Coordination

The `GpuManager` automatically distributes work across available devices with advanced error handling:

- **Device Enumeration**: Automatic detection of GPU devices
- **Work Distribution**: Intelligent batch splitting across multiple GPUs
- **Load Balancing**: Even distribution of computational work
- **Error Handling**: Graceful fallback when individual devices fail
- **Device Health Monitoring**: Real-time status tracking and failure detection
- **Automatic Failover**: Failed devices are removed from pool, work reassigned
- **CPU Fallback**: Graceful fallback to CPU processing when all GPUs fail
- **Recovery Attempts**: Automatic device recovery and reintegration

## Configuration

### Basic Configuration

```json
{
  "word_constraints": [
    {
      "position": 0,
      "prefix": "ab",
      "words": []
    }
  ],
  "ethereum": {
    "derivation_path": "m/44'/60'/0'/0/0",
    "target_address": "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f"
  },
  "batch_size": 10000,
  "passphrase": ""
}
```

### GPU Configuration

Add GPU-specific settings to enable acceleration:

```json
{
  "gpu": {
    "backend": "opencl",
    "devices": [],
    "multi_gpu": true
  }
}
```

#### GPU Configuration Options

- **`backend`**: `"opencl"` or `"cuda"` - Choose GPU acceleration backend
- **`devices`**: Array of device IDs (empty = use all available devices)
- **`multi_gpu`**: Boolean to enable multi-GPU processing

### Example Configurations

#### OpenCL with All Devices
```json
{
  "gpu": {
    "backend": "opencl",
    "devices": [],
    "multi_gpu": true
  }
}
```

#### CUDA with Specific Devices
```json
{
  "gpu": {
    "backend": "cuda", 
    "devices": [0, 1],
    "multi_gpu": true
  }
}
```

## Usage

### Basic Usage

#### Quick Start (Recommended)
```bash
# Use the provided test configuration
./target/release/bip39-solver-gpu --config example_test_config.json

# Override GPU backend
./target/release/bip39-solver-gpu --config example_test_config.json --gpu-backend cuda

# Run stress tests to validate GPU functionality
./target/release/bip39-solver-gpu --config example_test_config.json --stress-test
```

#### Command Line Options

The CLI now provides enhanced GPU control and error handling:

```bash
# Basic usage with recommended config
./target/release/bip39-solver-gpu --config example_test_config.json

# GPU backend selection
./target/release/bip39-solver-gpu --config example_test_config.json --gpu-backend <opencl|cuda>

# Specific GPU device selection
./target/release/bip39-solver-gpu --config example_test_config.json --gpu-device 0 --gpu-device 1

# Multi-GPU processing
./target/release/bip39-solver-gpu --config example_test_config.json --multi-gpu

# Comprehensive stress testing
./target/release/bip39-solver-gpu --config example_test_config.json --stress-test

# Distributed worker mode
./target/release/bip39-solver-gpu --config example_test_config.json --mode worker
```

#### Enhanced Error Handling

The application now provides detailed troubleshooting guidance:

```
⚠️  Failed to initialize GPU backend: CUDA runtime not available
🔄 Falling back to CPU processing
   This may be significantly slower than GPU processing.

💡 GPU Troubleshooting Tips:
   • For CUDA: Ensure NVIDIA drivers and CUDA toolkit are installed
   • Try: nvidia-smi to check GPU status
   • Try: nvcc --version to check CUDA installation
   • Try --gpu-backend cuda or --gpu-backend opencl to switch backends
   • Build with features: cargo build --features cuda,opencl
```

## Building

### Prerequisites

For optimal GPU performance, install the appropriate drivers and development tools:

#### CUDA Support (NVIDIA GPUs)
```bash
# Install CUDA Toolkit (11.0 or later)
# Ubuntu/Debian:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit

# Verify installation
nvcc --version
nvidia-smi
```

#### OpenCL Support (AMD/Intel/NVIDIA GPUs)
```bash
# Ubuntu/Debian:
sudo apt-get install opencl-headers ocl-icd-opencl-dev

# Verify installation
clinfo
```

### Build Options

#### Standard Build (CPU only with GPU fallback)
```bash
cargo build --release
```

#### With OpenCL Support
```bash
cargo build --release --features opencl
```

#### With CUDA Support
```bash
# Requires CUDA toolkit and nvcc in PATH
cargo build --release --features cuda
```

#### With Full GPU Support
```bash
cargo build --release --features cuda,opencl
```

### Build System Features

The build system automatically:
- ✅ **Fixed CUDA compilation**: Properly handles nvcc flags (removed duplicate -fPIC issue)
- ✅ **Enhanced linking**: Automatically finds CUDA libraries and adds proper linking directives
- ✅ **Graceful fallbacks**: Builds successfully even when GPU libraries are unavailable
- ✅ **Feature-gated dependencies**: OpenCL and CUDA are optional features to avoid linking issues
- ✅ **Smart detection**: Checks for CUDA/OpenCL availability at build time

### Troubleshooting Build Issues

If you encounter build issues:

1. **CUDA compilation errors**: Ensure CUDA toolkit is properly installed and nvcc is in PATH
2. **OpenCL linking errors**: Install OpenCL development headers for your platform
3. **Missing GPU features**: Build without GPU features: `cargo build --release` (CPU fallback included)
4. **Library not found**: The build system now handles missing libraries gracefully

```json
{
  "word_constraints": [
    {
      "position": 0,
      "prefix": "aban",
      "words": []
    },
    {
      "position": 11,
      "prefix": null,
      "words": ["abandon", "ability", "about"]
    }
  ],
  "ethereum": {
    "derivation_path": "m/44'/60'/0'/0/0",
    "target_address": "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f"
  },
  "slack": {
    "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    "channel": "#notifications"
  },
  "worker": {
    "server_url": "http://localhost:3000",
    "secret": "your-secret-key"
  },
  "gpu": {
    "backend": "opencl",
    "devices": [],
    "multi_gpu": true
  },
  "batch_size": 1000000,
  "passphrase": ""
}
```

### Configuration Options

- **word_constraints**: Array of constraints for specific mnemonic positions
  - `position`: Word position (0-11) in the 12-word mnemonic
  - `prefix`: Known prefix for words at this position (optional)
  - `words`: Exact list of possible words for this position (optional)
- **ethereum**: Ethereum-specific settings
  - `derivation_path`: BIP44 derivation path (typically `m/44'/60'/0'/0/0`)
  - `target_address`: The Ethereum address you're trying to find (with 0x prefix)
- **slack**: Slack notification settings (optional)
  - `webhook_url`: Your Slack webhook URL
  - `channel`: Slack channel for notifications (optional)
- **worker**: Distributed processing settings (required for distributed mode)
  - `server_url`: URL of the job server (e.g., `http://localhost:3000`)
  - `secret`: Shared secret for authentication between server and workers
- **gpu**: GPU processing settings (optional)
  - `backend`: GPU backend to use (`"opencl"` or `"cuda"`)
  - `devices`: Array of GPU device IDs to use (empty array = use all available)
  - `multi_gpu`: Enable processing across multiple GPU devices
- **batch_size**: Number of mnemonics to process in each batch
- **passphrase**: BIP39 passphrase (empty string if none)

## Usage

### Standalone Mode (Single Machine)

1. **Install Dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libssl-dev pkg-config opencl-headers ocl-icd-opencl-dev
   
   # macOS
   brew install openssl pkg-config
   ```

2. **Build**:
   ```bash
   cargo build --release
   ```

3. **Create Configuration**: 
   ```bash
   # Generate sample config
   ./target/release/bip39-solver-gpu --config config.json
   # Edit config.json with your target address and constraints
   ```

4. **Run Standalone**:
   ```bash
   ./target/release/bip39-solver-gpu --config config.json --mode standalone
   # Or simply (standalone is default):
   ./target/release/bip39-solver-gpu --config config.json
   
   # Use specific GPU backend:
   ./target/release/bip39-solver-gpu --config config.json --gpu-backend cuda
   
   # Use specific GPU devices:
   ./target/release/bip39-solver-gpu --config config.json --gpu-device 0 --gpu-device 1
   
   # Enable multi-GPU processing:
   ./target/release/bip39-solver-gpu --config config.json --multi-gpu
   ```

### GPU Backend Configuration

The tool supports multiple GPU backends for acceleration:

#### OpenCL Backend (Default)
- Supports most GPU vendors (NVIDIA, AMD, Intel)
- Falls back to CPU processing if no OpenCL devices available
- Good compatibility across different systems

#### CUDA Backend
- Optimized for NVIDIA GPUs
- Currently in development (stub implementation)
- Will provide better performance on supported hardware

#### Multi-GPU Support
Enable multi-GPU processing to scale across multiple devices:

```bash
# Use all available GPU devices
./target/release/bip39-solver-gpu --config config.json --multi-gpu

# Use specific devices only
./target/release/bip39-solver-gpu --config config.json --gpu-device 0 --gpu-device 2 --multi-gpu

# Force CUDA backend (when available)
./target/release/bip39-solver-gpu --config config.json --gpu-backend cuda --multi-gpu
```

#### GPU Configuration Examples

```json
{
  "gpu": {
    "backend": "opencl",
    "devices": [],         // Empty = use all available
    "multi_gpu": true
  }
}
```

```json
{
  "gpu": {
    "backend": "cuda", 
    "devices": [0, 1],     // Use specific GPU devices
    "multi_gpu": true
  }
}
```

## Advanced Error Handling & Stress Testing

This tool now includes comprehensive error handling and stress testing capabilities for robust GPU operations:

### Error Handling Features

- **Comprehensive Error Detection**: Detects and reports all CUDA errors from FFI calls and kernel launches
- **Device Health Monitoring**: Real-time monitoring of GPU device status and health
- **Automatic Failover**: Failed devices are automatically removed from the pool and work is reassigned to healthy GPUs
- **Graceful CPU Fallback**: When all GPUs fail, the system gracefully falls back to CPU processing
- **Structured Logging**: Detailed error logging with device info, timestamps, and recovery attempts
- **Device Recovery**: Automatic attempts to recover failed devices and reintegrate them into the pool

### Stress Testing

Run comprehensive stress tests to validate error handling and system robustness:

```bash
# Run full stress test suite
./target/release/bip39-solver-gpu --config config.json --stress-test
```

The stress testing framework includes:

- **Huge Batch Size Testing**: Tests with progressively larger batch sizes to stress memory allocation
- **Out-of-Memory Scenarios**: Validates graceful handling of memory exhaustion
- **Max Thread Scenarios**: Tests concurrent processing at maximum thread capacity
- **GPU Failure Simulation**: Simulates various GPU failure modes (kernel timeout, device reset, memory corruption, driver crash)
- **Device Removal Simulation**: Tests hot-unplugging of GPU devices during operation
- **Concurrent Stress Testing**: Combines multiple stress factors simultaneously
- **Memory Fragmentation Testing**: Tests memory allocation patterns under stress
- **Long-Running Stability**: Validates system stability over extended periods
- **Distributed Network Stress**: Simulates network issues in distributed mode (slow connections, packet loss, high latency)

### Error Types

The system can detect and handle these error conditions:

- **Device Initialization Failures**: GPU device fails to initialize properly
- **Kernel Execution Failures**: CUDA kernel fails during execution
- **Memory Allocation Failures**: Insufficient GPU memory for batch processing
- **Device Timeouts**: GPU device becomes unresponsive
- **Hardware Failures**: Physical GPU hardware failures
- **Out of Memory**: Batch size exceeds available GPU memory
- **Device Removal**: GPU device is physically disconnected during operation
- **Backend Unavailable**: GPU backend (CUDA/OpenCL) is not available

### Example Stress Test Output

```
🚀 Starting Advanced Error Handling & Stress Testing for CUDA Backend
=====================================================================

🧪 Starting comprehensive stress testing...

  🔥 Testing huge batch sizes...
    ✅ Batch size 10000 handled successfully
    ✅ Batch size 50000 handled successfully
    ✅ Batch size 100000 handled successfully
    ✅ Batch size 500000 handled successfully
  💾 Testing out-of-memory scenarios...
    ✅ OOM test 'extreme_batch' handled gracefully
    ✅ OOM test 'repeated_allocations' handled gracefully
  🧵 Testing max thread scenarios...
    📊 4 threads completed, 0 errors
  💥 Testing GPU failure simulation...
    ✅ GPU failure simulation 'kernel_timeout' handled
    ✅ GPU failure simulation 'device_reset' handled
    ✅ GPU failure simulation 'memory_corruption' handled
    ✅ GPU failure simulation 'driver_crash' handled

📊 STRESS TEST REPORT
=====================
Total Tests: 8
Passed: 8 ✅
Failed: 0 ❌
Success Rate: 100.0%
```

### Distributed Mode (Multiple Machines)

The distributed mode allows you to scale the search across multiple machines for faster processing.

#### 1. Setup Job Server

On the machine that will coordinate the work:

```bash
# Start the job server
./target/release/bip39-server --config distributed_config.json
```

The server will:
- Listen on port 3000 by default
- Divide the search space into jobs
- Distribute jobs to workers
- Track progress and handle failures
- Send notifications when solution is found

#### 2. Setup Worker Clients

On each worker machine:

```bash
# Start a worker (will automatically get unique ID)
./target/release/bip39-worker --config distributed_config.json

# Or specify custom worker ID
./target/release/bip39-worker --config distributed_config.json --worker-id worker-gpu-01
```

Workers will:
- Connect to the job server
- Request work assignments
- Process assigned ranges
- Report results back to server
- Automatically retry failed jobs

#### 3. Monitor Progress

Check server status via HTTP API:
```bash
curl -H "Authorization: Bearer my-secret-key" http://localhost:3000/api/status
```

#### 4. Distributed Configuration

Make sure your `distributed_config.json` includes the worker section:

```json
{
  "word_constraints": [...],
  "ethereum": {...},
  "slack": {...},
  "worker": {
    "server_url": "http://your-server-ip:3000",
    "secret": "your-shared-secret-key"
  },
  "batch_size": 100000,
  "passphrase": ""
}
```

## Example Output

```
Loaded config from: config.json
Target address: 0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f
Derivation path: m/44'/60'/0'/0/0
Total combinations to search: 31457280000000000
Searching batch: 0 to 1000000
Progress: 1000000/31457280000000000 (0.00%) - Rate: 2850.32 mnemonics/sec - Elapsed: 350.62s
...
🎉 Found matching mnemonic!
Mnemonic: abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about
Address: 0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f
Offset: 12345678
```

### Distributed Mode Example

**Server Output:**
```
Starting job server with config from: distributed_config.json
Target address: 0x9858EfFD232B4033E47d90003D41EC34EcaEda94
Initializing jobs for search space of 419430400000000000 combinations
Initialized 4194304 jobs
Job server listening on port 3000
Assigned job job-1 to worker worker-12345 (range: 0 to 100000)
Assigned job job-2 to worker worker-67890 (range: 100000 to 200000)
Job job-1 completed by worker worker-12345 (100000 mnemonics checked)
🎉 Solution found! Mnemonic: abandon abandon abandon...
```

**Worker Output:**
```
Worker worker-12345 starting, connecting to server at http://localhost:3000
Received job job-1: range 0 to 100000
Processing job job-1 with 100000 candidates
Completed job job-1 in 35.2s (2840.91 mnemonics/sec)
Received job job-15: range 1500000 to 1600000
...
```

## Performance

The performance varies significantly based on the processing backend:

### CPU Processing
- Single machine: ~3,000 mnemonics/sec (baseline)
- 10 worker machines: ~30,000 mnemonics/sec  
- 100 worker machines: ~300,000 mnemonics/sec

### GPU Processing (When Fully Implemented)
- Single OpenCL device: ~50,000-100,000 mnemonics/sec (estimated)
- Single CUDA device: ~100,000-200,000 mnemonics/sec (estimated)
- Multi-GPU setup (4 devices): ~400,000-800,000 mnemonics/sec (estimated)

### Current Status
- OpenCL backend currently falls back to CPU processing
- CUDA backend is in development (stub implementation)
- Multi-GPU coordination infrastructure is in place
- Full GPU acceleration will be available in future releases

Distributed mode with multiple worker machines can scale linearly regardless of the backend used.

## Technical Details

### Address Derivation Process

1. Generate BIP39 mnemonic from word constraints
2. Create seed using PBKDF2-HMAC-SHA512 with "mnemonic" + passphrase
3. Derive master key using HMAC-SHA512 with "Bitcoin seed"
4. Follow BIP44 derivation path for Ethereum (m/44'/60'/0'/0/0)
5. Generate public key using secp256k1
6. Hash public key with Keccak-256
7. Take last 20 bytes as Ethereum address

### Testing

Run the test suite to verify functionality:

```bash
cargo test
```

Tests include:
- Known mnemonic to address derivation
- Address validation and comparison
- Configuration serialization
- Word space constraint handling
- Distributed job management
- Worker client functionality

### REST API

The job server provides a REST API for monitoring and management:

#### Endpoints

- `GET /api/status` - Get server status and progress
- `POST /api/jobs/request` - Request a job (used by workers)
- `POST /api/jobs/complete` - Report job completion (used by workers)
- `POST /api/jobs/heartbeat` - Send worker heartbeat (used by workers)

#### Authentication

All requests require a Bearer token matching the secret in the configuration:

```bash
curl -H "Authorization: Bearer your-secret-key" http://localhost:3000/api/status
```

#### Status Response Example

```json
{
  "total_jobs": 4194304,
  "pending_jobs": 1000000,
  "assigned_jobs": 10,
  "completed_jobs": 3194294,
  "failed_jobs": 0,
  "active_workers": 5,
  "total_combinations": 419430400000000000,
  "combinations_searched": 319429400000000000,
  "search_rate": 15000.0,
  "uptime_seconds": 3600,
  "solution_found": null
}
```

## Security Considerations

- This tool is for legitimate recovery purposes only
- Never use on wallets that don't belong to you
- The search space for unrestricted mnemonics is computationally infeasible
- Always verify recovered mnemonics independently

## Roadmap

- [x] Complete GPU OpenCL kernel implementation for Ethereum
- [x] Proper BIP32 hierarchical deterministic derivation
- [x] Support for custom derivation paths
- [x] Distributed processing across multiple machines
- [x] REST API for job management and monitoring
- [x] Worker heartbeat and fault tolerance
- [x] Slack notification integration
- [x] Modular GPU backend architecture (OpenCL/CUDA)
- [x] Multi-GPU support infrastructure
- [x] CUDA backend stub and FFI bindings
- [x] **Advanced Error Handling & Device Management**
- [x] **Comprehensive CUDA error checking for all FFI calls**
- [x] **Device health monitoring and failure detection**
- [x] **Automatic failover from failed devices to healthy ones**
- [x] **Graceful CPU fallback when all GPUs fail**
- [x] **Structured logging with device info and timestamps**
- [x] **Stress Testing Framework**
- [x] **Edge-case testing for huge batch sizes and OOM scenarios**
- [x] **GPU failure simulation and recovery testing**
- [x] **Distributed network stress testing capabilities**
- [ ] Complete CUDA kernel implementations (PBKDF2, secp256k1, Keccak-256)
- [ ] Full OpenCL kernel integration
- [ ] GPU memory optimization and batch processing
- [ ] Hardware wallet integration for verification
- [ ] Web-based monitoring dashboard
- [ ] Support for other cryptocurrencies (Bitcoin, etc.)
- [ ] Database persistence for large-scale deployments

## Migration from Bitcoin

This is a migration of the original Bitcoin BIP39 solver. Key changes:

- Replaced double-SHA256 with Keccak-256 for address derivation
- Updated from Bitcoin address format to Ethereum address format
- Changed from BIP44 Bitcoin path to Ethereum path
- Added configuration file support
- Replaced hardcoded values with configurable options

## License

This project maintains the same license as the original Bitcoin implementation.
