#!/bin/bash
set -e

echo "==== BIP39 Solver GPU: Automated Installer, Builder & Quick Test ===="

# Check for rust/cargo
if ! command -v cargo &> /dev/null; then
    echo "Rust/Cargo not found. Installing Rust toolchain..."
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    source $HOME/.cargo/env
    # Add Rust to PATH permanently
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.profile
else
    echo "Rust and Cargo found."
fi

echo "Installing system dependencies..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    
    # Install OpenCL development headers
    echo "Installing OpenCL development headers..."
    sudo apt-get install -y ocl-icd-opencl-dev clinfo
    
    # Install CUDA toolkit (for better GPU support)
    echo "Installing NVIDIA CUDA toolkit..."
    sudo apt-get install -y nvidia-cuda-toolkit nvidia-driver-470 || {
        echo "Warning: CUDA installation failed. Trying alternative method..."
        # Try downloading CUDA from NVIDIA (more reliable)
        wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run || {
            echo "Note: CUDA download failed. Will try to build with existing drivers."
        }
    }
    
    # Install build essentials
    sudo apt-get install -y build-essential pkg-config libssl-dev
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "On macOS - installing dependencies via Homebrew..."
    brew install pkg-config openssl
    echo "OpenCL is included with macOS. CUDA requires manual installation from NVIDIA."
fi

# Ensure Rust is in PATH for this session
export PATH="$HOME/.cargo/bin:$PATH"

echo "Building project with GPU features..."
cargo build --release --features cuda,opencl

echo "Running quick test with example_test_config.json:"
./target/release/bip39-solver-gpu --config example_test_config.json

echo ""
echo "==== Manual Server/Worker Launch Example ===="
echo "To launch a job server:"
echo "  ./target/release/bip39-server --config example_test_config.json"
echo "To launch a worker:"
echo "  ./target/release/bip39-worker --config example_test_config.json --worker"

echo ""
echo "==== GPU Status Check ===="
echo "Checking NVIDIA GPU status:"
nvidia-smi || echo "No NVIDIA GPU or drivers found"
echo ""
echo "Checking OpenCL devices:"
clinfo || echo "No OpenCL devices found"

echo ""
echo "==== Done! Check output above for results and next steps. ====" 