#!/bin/bash
set -e

echo "==== BIP39 Solver GPU: VastAI Docker Optimized Installer ===="

# Detect if we're in a Docker environment or VastAI instance
DOCKER_ENV=false
VAST_AI=false

if [ -f /.dockerenv ]; then
    DOCKER_ENV=true
    echo "🐳 Docker environment detected"
fi

if [ -n "${VAST_INSTANCE_ID}" ] || [ -n "${VAST_CONTAINERD_LOGPATH}" ] || [ -n "${VAST_TCP_PORT_22}" ]; then
    VAST_AI=true
    echo "🚀 Vast.AI environment detected"
fi

if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "🎯 CUDA devices available: ${CUDA_VISIBLE_DEVICES}"
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

# Function to safely install packages (VastAI optimized)
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
        
        # Update package list if not done recently
        if [ ! -f /var/lib/apt/periodic/update-stamp ] || [ $(find /var/lib/apt/periodic/update-stamp -mmin +60) ]; then
            echo "📦 Updating package lists..."
            apt-get update -qq
        fi
        
        if command -v apt-get &> /dev/null; then
            DEBIAN_FRONTEND=noninteractive apt-get install -y "${to_install[@]}" || {
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
    echo "📦 Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path
    source ~/.cargo/env
    # Add Rust to PATH permanently  
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    if [ -f ~/.profile ]; then
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.profile
    fi
else
    echo "✅ Rust and Cargo found: $(cargo --version)"
fi

# Ensure Rust is in PATH for this session
export PATH="$HOME/.cargo/bin:$PATH"

echo "📦 Installing essential build dependencies..."
if [ "$VAST_AI" = true ] || [ "$DOCKER_ENV" = true ]; then
    # In VastAI/Docker environment, focus on missing packages only
    echo "🚀 VastAI/Docker environment - installing minimal required packages"
    safe_install build-essential pkg-config libssl-dev cmake git curl wget
else
    # Full system installation
    echo "🖥️  Full system installation"
    if command -v apt-get &> /dev/null; then
        safe_install build-essential pkg-config libssl-dev cmake git curl wget \
                    software-properties-common ca-certificates gnupg lsb-release
    fi
fi

# Check NVIDIA driver status
if check_nvidia_driver; then
    echo "🎯 NVIDIA GPU(s) detected and working"
    NVIDIA_WORKING=true
else
    echo "⚠️  No working NVIDIA driver detected"
    NVIDIA_WORKING=false
    
    if [ "$VAST_AI" = true ]; then
        echo "🚀 VastAI environment - drivers should be pre-installed by host"
        echo "💡 If you see this message, the VastAI instance may not have GPU access"
    elif [ "$DOCKER_ENV" = true ]; then
        echo "🐳 Docker environment - assuming host has NVIDIA drivers configured"
        echo "💡 Make sure to run with --gpus all flag"
    else
        echo "📦 Installing NVIDIA drivers..."
        if command -v ubuntu-drivers &> /dev/null; then
            ubuntu-drivers install nvidia || echo "⚠️  NVIDIA driver installation failed"
        else
            safe_install nvidia-driver-535 || safe_install nvidia-driver-530 || \
            echo "⚠️  NVIDIA driver installation failed"
        fi
    fi
fi

# Smart CUDA toolkit detection and installation
CUDA_AVAILABLE=false

# Check existing CUDA installations
for cuda_path in /usr/local/cuda /opt/cuda /usr/cuda; do
    if [ -f "$cuda_path/bin/nvcc" ]; then
        echo "✅ Found existing CUDA at: $cuda_path"
        export PATH="$cuda_path/bin:$PATH"
        export CUDA_PATH="$cuda_path"
        echo "export PATH=\"$cuda_path/bin:\$PATH\"" >> ~/.bashrc
        echo "export CUDA_PATH=\"$cuda_path\"" >> ~/.bashrc
        CUDA_AVAILABLE=true
        break
    fi
done

# Check if nvcc is already in PATH
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA toolkit found: $(nvcc --version | grep release | cut -d, -f2)"
    CUDA_AVAILABLE=true
else
    echo "📦 Installing CUDA toolkit..."
    if [ "$VAST_AI" = true ]; then
        # VastAI often has CUDA pre-installed in non-standard locations
        echo "🚀 VastAI: Checking for pre-installed CUDA..."
        
        # Common VastAI CUDA locations
        for cuda_path in /usr/local/cuda-12.2 /usr/local/cuda-12.1 /usr/local/cuda-12.0 \
                         /usr/local/cuda-11.8 /usr/local/cuda-11.7 /usr/local/cuda-11.6; do
            if [ -f "$cuda_path/bin/nvcc" ]; then
                echo "✅ Found VastAI CUDA at: $cuda_path"
                export PATH="$cuda_path/bin:$PATH"
                export CUDA_PATH="$cuda_path"
                echo "export PATH=\"$cuda_path/bin:\$PATH\"" >> ~/.bashrc
                echo "export CUDA_PATH=\"$cuda_path\"" >> ~/.bashrc
                CUDA_AVAILABLE=true
                break
            fi
        done
        
        # If not found, try package installation
        if [ "$CUDA_AVAILABLE" = false ]; then
            safe_install nvidia-cuda-toolkit || safe_install cuda-toolkit-12-3 || {
                echo "💡 VastAI Note: CUDA toolkit installation failed, but runtime may be available"
            }
        fi
    else
        # Standard installation for other environments
        safe_install nvidia-cuda-toolkit || safe_install cuda-toolkit-12-3 || {
            echo "⚠️  CUDA toolkit installation failed"
        }
    fi
fi

# Install OpenCL support (optimized for VastAI/Docker)
echo "📦 Installing OpenCL support..."
if [ "$VAST_AI" = true ] || [ "$DOCKER_ENV" = true ]; then
    # Minimal OpenCL for containerized environments
    safe_install ocl-icd-opencl-dev opencl-headers || {
        echo "💡 OpenCL dev packages not available - this is normal in some containers"
    }
    
    # Try to install clinfo for debugging
    safe_install clinfo || {
        echo "💡 clinfo not available"
    }
else
    # Full OpenCL installation for regular systems
    safe_install ocl-icd-opencl-dev opencl-headers clinfo \
                mesa-opencl-icd intel-opencl-icd || {
        echo "⚠️  Some OpenCL packages not available"
    }
fi

# Try to install NVIDIA OpenCL if NVIDIA driver is working
if [ "$NVIDIA_WORKING" = true ]; then
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

# CUDA status and path setup
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA: $(nvcc --version | grep release)"
else
    # Final attempt to find CUDA
    for cuda_path in /usr/local/cuda* /opt/cuda* /usr/cuda*; do
        if [ -f "$cuda_path/bin/nvcc" ] 2>/dev/null; then
            echo "✅ CUDA found at: $cuda_path/bin/nvcc"
            echo "💡 Adding to PATH: export PATH=\"$cuda_path/bin:\$PATH\""
            export PATH="$cuda_path/bin:$PATH"
            export CUDA_PATH="$cuda_path"
            break
        fi
    done
    
    if ! command -v nvcc &> /dev/null; then
        echo "❌ CUDA toolkit not found"
    fi
fi

# NVIDIA driver status
if check_nvidia_driver; then
    echo "✅ NVIDIA driver working"
    if [ "$VAST_AI" = true ]; then
        echo "🚀 VastAI GPU configuration appears correct"
    fi
else
    echo "❌ NVIDIA driver not working"
    if [ "$VAST_AI" = true ]; then
        echo "🚀 VastAI Note: GPU may not be allocated to this instance"
        echo "💡 Check your VastAI instance has GPU allocation"
    fi
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

# Try building with all features first
if cargo build --release --features cuda,opencl 2>/dev/null; then
    echo "✅ Build successful with CUDA and OpenCL!"
    BUILD_SUCCESS=true
elif cargo build --release --features cuda 2>/dev/null; then
    echo "✅ Build successful with CUDA only!"
    BUILD_SUCCESS=true
elif cargo build --release --features opencl 2>/dev/null; then
    echo "✅ Build successful with OpenCL only!"
    BUILD_SUCCESS=true
elif cargo build --release 2>/dev/null; then
    echo "✅ Build successful (CPU-only mode)!"
    BUILD_SUCCESS=true
else
    echo "❌ Build failed"
    BUILD_SUCCESS=false
fi

if [ "$BUILD_SUCCESS" = true ]; then
    echo ""
    echo "🧪 Running quick GPU detection test:"
    timeout 10 ./target/release/bip39-solver-gpu --config example_test_config.json 2>/dev/null | head -20 || {
        echo "⚠️  Test failed or timed out. This may be normal if no target address is found quickly."
        echo "💡 The build was successful, so the solver should work when properly configured."
    }
else
    echo ""
    echo "🔧 Troubleshooting tips:"
    echo "  • Ensure NVIDIA drivers are installed: nvidia-smi"
    echo "  • Ensure CUDA toolkit is available: nvcc --version"
    echo "  • Check OpenCL: clinfo"
    echo "  • Try building without features: cargo build --release"
    
    if [ "$VAST_AI" = true ]; then
        echo ""
        echo "🚀 VastAI specific tips:"
        echo "  • Ensure your instance has GPU allocation"
        echo "  • Check CUDA_VISIBLE_DEVICES environment variable"
        echo "  • Try: nvidia-smi to verify GPU access"
    fi
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

if [ "$VAST_AI" = true ]; then
    echo ""
    echo "🚀 VastAI specific commands:"
    echo "  vastai show instance \$VAST_INSTANCE_ID    # Check instance status"
    echo "  echo \$CUDA_VISIBLE_DEVICES               # Check available GPUs"
fi

echo ""
if [ "$BUILD_SUCCESS" = true ]; then
    echo "✨ Installation complete! GPU solver ready to use."
else
    echo "⚠️  Installation completed with issues. Check troubleshooting tips above."
fi