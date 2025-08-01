#!/bin/bash
set -e

echo "==== BIP39 Solver GPU: Docker/VastAI Optimized Installer ===="

# Detect if we're in a Docker environment or VastAI instance
DOCKER_ENV=false
if [ -f /.dockerenv ] || [ -n "${VAST_INSTANCE_ID}" ] || [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    DOCKER_ENV=true
    echo "🐳 Docker/VastAI environment detected"
fi

# Function to check if package is installed
is_package_installed() {
    if command -v dpkg &> /dev/null; then
        dpkg -l "$1" &> /dev/null
    else
        rpm -q "$1" &> /dev/null 2>&1 || yum list installed "$1" &> /dev/null 2>&1 || false
    fi
}

# Function to check if NVIDIA driver is working
check_nvidia_driver() {
    if nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA driver working:"
        nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader,nounits | head -3
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
            echo "✓ Package $package already installed"
        fi
    done
    
    if [ ${#to_install[@]} -gt 0 ]; then
        echo "Installing packages: ${to_install[*]}"
        if command -v apt-get &> /dev/null; then
            apt-get update && apt-get install -y "${to_install[@]}" || {
                echo "⚠️  Some packages failed to install: ${to_install[*]}"
                return 1
            }
        elif command -v yum &> /dev/null; then
            yum install -y "${to_install[@]}" || {
                echo "⚠️  Some packages failed to install: ${to_install[*]}"
                return 1
            }
        else
            echo "❌ No supported package manager found"
            return 1
        fi
    fi
}

# Check for Rust/Cargo
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source ~/.cargo/env
    # Add Rust to PATH permanently
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.profile
else
    echo "✅ Rust and Cargo found: $(cargo --version)"
fi

# Ensure Rust is in PATH for this session
export PATH="$HOME/.cargo/bin:$PATH"

echo "Installing essential build dependencies..."
if [ "$DOCKER_ENV" = true ]; then
    # In Docker/VastAI environment, focus on missing packages only
    echo "🐳 Docker environment - installing minimal required packages"
    safe_install build-essential pkg-config libssl-dev cmake git curl wget
else
    # Full system installation
    echo "🖥️  Full system installation"
    if command -v apt-get &> /dev/null; then
        apt-get update
        safe_install build-essential pkg-config libssl-dev cmake git curl wget \
                    software-properties-common ca-certificates gnupg lsb-release
    fi
fi

# Check NVIDIA driver status
if check_nvidia_driver; then
    echo "🎯 NVIDIA GPU(s) detected and working"
else
    echo "⚠️  No working NVIDIA driver detected"
    if [ "$DOCKER_ENV" = false ]; then
        echo "Installing NVIDIA drivers..."
        if command -v ubuntu-drivers &> /dev/null; then
            ubuntu-drivers install nvidia || echo "⚠️  NVIDIA driver installation failed"
        else
            safe_install nvidia-driver-535 || safe_install nvidia-driver-530 || \
            echo "⚠️  NVIDIA driver installation failed"
        fi
    else
        echo "🐳 In Docker - assuming host has NVIDIA drivers configured"
    fi
fi

# Check for CUDA toolkit
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA toolkit found: $(nvcc --version | grep release)"
else
    echo "Installing CUDA toolkit..."
    if [ "$DOCKER_ENV" = true ]; then
        # In Docker, prioritize apt installation
        safe_install nvidia-cuda-toolkit || {
            # Try CUDA 12 packages if available
            safe_install cuda-toolkit-12-3 cuda-toolkit-12-2 cuda-toolkit-12-1 || {
                echo "⚠️  CUDA toolkit installation failed"
                echo "💡 Note: CUDA may already be available in /usr/local/cuda"
                # Check if CUDA is in common locations
                for cuda_path in /usr/local/cuda /opt/cuda /usr/cuda; do
                    if [ -f "$cuda_path/bin/nvcc" ]; then
                        echo "✅ Found CUDA at: $cuda_path"
                        export PATH="$cuda_path/bin:$PATH"
                        export CUDA_PATH="$cuda_path"
                        echo "export PATH=\"$cuda_path/bin:\$PATH\"" >> ~/.bashrc
                        echo "export CUDA_PATH=\"$cuda_path\"" >> ~/.bashrc
                        break
                    fi
                done
            }
        }
    else
        # Full system CUDA installation
        if command -v apt-get &> /dev/null; then
            if [ ! -f /etc/apt/sources.list.d/cuda*.list ]; then
                echo "Adding NVIDIA CUDA repository..."
                wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/3bf863cc.pub | apt-key add - || echo "⚠️  CUDA GPG key failed"
                echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/ /" > /etc/apt/sources.list.d/cuda.list
                apt-get update
            fi
            safe_install nvidia-cuda-toolkit cuda-toolkit-12-3 || echo "⚠️  CUDA installation failed"
        fi
    fi
fi

# Install OpenCL support
echo "Installing OpenCL support..."
if [ "$DOCKER_ENV" = true ]; then
    # Minimal OpenCL for Docker
    safe_install ocl-icd-opencl-dev opencl-headers clinfo || {
        echo "⚠️  OpenCL packages not available in this environment"
    }
else
    # Full OpenCL installation
    safe_install ocl-icd-opencl-dev opencl-headers clinfo \
                mesa-opencl-icd intel-opencl-icd || {
        echo "⚠️  Some OpenCL packages not available"
    }
fi

# Try to install NVIDIA OpenCL if NVIDIA driver is working
if check_nvidia_driver; then
    safe_install nvidia-opencl-icd || safe_install nvidia-opencl-dev || {
        echo "💡 NVIDIA OpenCL may already be included with driver"
    }
fi

echo ""
echo "==== Installation Summary ===="

# Check final status
echo "🔍 Checking component status..."

# Rust status
if command -v cargo &> /dev/null; then
    echo "✅ Rust: $(rustc --version)"
else
    echo "❌ Rust not available"
fi

# CUDA status
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA: $(nvcc --version | grep release)"
elif [ -f /usr/local/cuda/bin/nvcc ]; then
    echo "✅ CUDA found at: /usr/local/cuda/bin/nvcc"
    export PATH="/usr/local/cuda/bin:$PATH"
    export CUDA_PATH="/usr/local/cuda"
else
    echo "❌ CUDA toolkit not found"
fi

# NVIDIA driver status
if check_nvidia_driver; then
    echo "✅ NVIDIA driver working"
else
    echo "❌ NVIDIA driver not working"
fi

# OpenCL status  
if command -v clinfo &> /dev/null; then
    echo "✅ OpenCL tools available"
    echo "📋 OpenCL platforms:"
    clinfo -l 2>/dev/null | head -10 || echo "❌ No OpenCL devices found"
else
    echo "❌ OpenCL tools not available"
fi

echo ""
echo "🔨 Building project with GPU features..."
if cargo build --release --features cuda,opencl; then
    echo "✅ Build successful!"
    
    echo ""
    echo "🧪 Running quick test:"
    timeout 30 ./target/release/bip39-solver-gpu --config example_test_config.json || {
        echo "⚠️  Test failed or timed out. Check GPU driver installation."
        echo "💡 This may be normal if no target address is found quickly."
    }
else
    echo "❌ Build failed"
    echo ""
    echo "🔧 Troubleshooting tips:"
    echo "  • Ensure NVIDIA drivers are installed: nvidia-smi"
    echo "  • Ensure CUDA toolkit is available: nvcc --version"
    echo "  • Check OpenCL: clinfo"
    echo "  • Try building without features: cargo build --release"
fi

echo ""
echo "==== Quick Start Commands ===="
echo "🚀 Run solver:"
echo "  ./target/release/bip39-solver-gpu --config example_test_config.json"
echo ""
echo "🔧 Advanced options:"
echo "  ./target/release/bip39-solver-gpu --config example_test_config.json --gpu-backend cuda --multi-gpu"
echo "  ./target/release/bip39-solver-gpu --config example_test_config.json --gpu-backend opencl"
echo ""
echo "📊 Check GPU status:"
echo "  nvidia-smi    # NVIDIA GPUs"
echo "  clinfo        # OpenCL devices"

echo ""
echo "✨ Installation complete! Check status above for any issues."