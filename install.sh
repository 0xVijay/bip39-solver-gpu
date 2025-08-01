#!/bin/bash
set -e

echo "==== BIP39 Solver GPU: Automated Installer, Builder & Quick Test ===="

# Function to check if package is installed
is_package_installed() {
    dpkg -l "$1" &> /dev/null
}

# Function to check if NVIDIA driver is already installed
check_nvidia_driver() {
    if nvidia-smi &> /dev/null; then
        echo "NVIDIA driver already installed and working:"
        nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1
        return 0
    fi
    return 1
}

# Function to safely install packages
safe_install() {
    local packages=("$@")
    local to_install=()
    
    for package in "${packages[@]}"; do
        if ! is_package_installed "$package"; then
            to_install+=("$package")
        else
            echo "Package $package already installed, skipping..."
        fi
    done
    
    if [ ${#to_install[@]} -gt 0 ]; then
        echo "Installing packages: ${to_install[*]}"
        sudo apt-get install -y "${to_install[@]}" || {
            echo "Warning: Some packages failed to install: ${to_install[*]}"
            return 1
        }
    else
        echo "All requested packages already installed."
    fi
}

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
    
    # Install basic build dependencies first
    echo "Installing build essentials and development dependencies..."
    safe_install build-essential pkg-config libssl-dev cmake git curl wget software-properties-common ca-certificates gnupg lsb-release
    
    # Install OpenCL development headers and runtime
    echo "Installing OpenCL development headers and runtime..."
    safe_install ocl-icd-opencl-dev opencl-headers clinfo || {
        echo "Installing basic OpenCL packages..."
        safe_install ocl-icd-opencl-dev clinfo
    }
    
    # Try to install OpenCL vendor implementations if not already present
    safe_install mesa-opencl-icd intel-opencl-icd || echo "Some OpenCL vendor implementations not available"
    
    # Check if NVIDIA drivers are already working
    if check_nvidia_driver; then
        echo "NVIDIA driver already working, skipping driver installation..."
    else
        echo "No working NVIDIA driver detected, attempting installation..."
        
        # Try to install NVIDIA drivers with fallback options
        echo "Installing NVIDIA drivers..."
        if ! safe_install nvidia-driver-535; then
            echo "nvidia-driver-535 failed, trying nvidia-driver-530..."
            if ! safe_install nvidia-driver-530; then
                echo "nvidia-driver-530 failed, trying nvidia-driver-525..."
                if ! safe_install nvidia-driver-525; then
                    echo "Manual driver packages failed, trying ubuntu-drivers..."
                    sudo ubuntu-drivers install nvidia || {
                        echo "Note: All NVIDIA driver installation methods failed."
                        echo "You may need to install drivers manually or they may already be installed."
                    }
                fi
            fi
        fi
    fi
    
    # Install CUDA toolkit if not already present
    if ! command -v nvcc &> /dev/null; then
        echo "NVCC not found, installing NVIDIA CUDA toolkit..."
        
        # Add NVIDIA official repository using package commands only
        if [ ! -f /etc/apt/sources.list.d/cuda*.list ]; then
            echo "Adding NVIDIA CUDA repository..."
            sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/3bf863cc.pub || echo "CUDA GPG key fetch failed"
            sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/ /" || echo "CUDA repository add failed"
            sudo apt-get update
        else
            echo "CUDA repository already configured."
        fi
        
        # Install CUDA toolkit from repositories
        if ! safe_install nvidia-cuda-toolkit; then
            echo "Trying alternative CUDA packages..."
            safe_install cuda-toolkit-12-3 cuda-drivers || {
                echo "Warning: CUDA toolkit installation failed. Building without CUDA support."
            }
        fi
    else
        echo "NVCC already available: $(nvcc --version | grep release)"
    fi
    
    # Try to install NVIDIA OpenCL implementation if driver is working
    if check_nvidia_driver; then
        safe_install nvidia-opencl-icd-384 || safe_install nvidia-opencl-dev || echo "NVIDIA OpenCL implementation not available"
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "On macOS - installing dependencies via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install pkg-config openssl
    else
        echo "Homebrew not found. Please install Homebrew first."
    fi
    echo "OpenCL is included with macOS. CUDA requires manual installation from NVIDIA."
fi

# Ensure Rust is in PATH for this session
export PATH="$HOME/.cargo/bin:$PATH"

echo ""
echo "==== Installation Summary ===="
echo "Checking final installation status..."

# Check CUDA installation
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA toolkit installed: $(nvcc --version | grep release)"
else
    echo "❌ CUDA toolkit not available"
fi

# Check NVIDIA driver
if check_nvidia_driver; then
    echo "✅ NVIDIA driver working"
else
    echo "❌ NVIDIA driver not working"
fi

# Check OpenCL
if command -v clinfo &> /dev/null; then
    echo "✅ OpenCL tools available"
    clinfo -l 2>/dev/null | head -5 || echo "❌ No OpenCL devices found"
else
    echo "❌ OpenCL tools not available"
fi

echo ""
echo "Building project with GPU features..."
if cargo build --release --features cuda,opencl; then
    echo "✅ Build successful"
    
    echo ""
    echo "Running quick test with example_test_config.json:"
    ./target/release/bip39-solver-gpu --config example_test_config.json || {
        echo "❌ Test failed. Check GPU driver installation."
    }
else
    echo "❌ Build failed. Check dependencies."
fi

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