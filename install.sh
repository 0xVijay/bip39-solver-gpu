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
    
    # Install OpenCL development headers and runtime
    echo "Installing OpenCL development headers and runtime..."
    sudo apt-get install -y ocl-icd-opencl-dev opencl-headers clinfo mesa-opencl-icd intel-opencl-icd nvidia-opencl-icd-384 || {
        echo "Installing basic OpenCL packages..."
        sudo apt-get install -y ocl-icd-opencl-dev clinfo
    }
    
    # Install CUDA toolkit via package managers only
    echo "Installing NVIDIA CUDA toolkit via package managers..."
    
    # Install required packages for adding repositories
    sudo apt-get install -y software-properties-common ca-certificates gnupg lsb-release
    
    # Add NVIDIA official repository using package commands only
    echo "Adding NVIDIA CUDA repository..."
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/3bf863cc.pub || echo "CUDA GPG key fetch failed"
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/ /" || echo "CUDA repository add failed"
    
    # Update package lists
    sudo apt-get update
    
    # Install CUDA toolkit from repositories
    sudo apt-get install -y cuda-toolkit-12-3 cuda-drivers nvidia-cuda-toolkit || {
        echo "Warning: Latest CUDA packages failed. Trying fallback packages..."
        sudo apt-get install -y nvidia-cuda-toolkit cuda-toolkit || {
            echo "Note: CUDA installation failed. Will build with existing setup."
        }
    }
    
    # Install NVIDIA drivers with multiple fallback options
    echo "Installing NVIDIA drivers..."
    sudo apt-get install -y nvidia-driver-535 nvidia-driver-530 nvidia-driver-525 || {
        echo "Installing latest available NVIDIA driver..."
        sudo ubuntu-drivers install nvidia || {
            echo "Note: NVIDIA driver installation failed or drivers already present."
        }
    }
    
    # Install build essentials and development dependencies
    sudo apt-get install -y build-essential pkg-config libssl-dev cmake git curl wget
    
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