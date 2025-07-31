#!/usr/bin/env bash
set -e

echo "=== bip39-solver-gpu Automated Installer ==="

# Detect OS and install deps
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    OS=$(uname -s)
fi

echo "[*] Installing system packages..."

if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
    sudo apt-get update
    sudo apt-get install -y build-essential pkg-config libssl-dev ocl-icd-opencl-dev opencl-headers git curl
elif [[ "$OS" == "arch" ]]; then
    sudo pacman -Syu --needed base-devel opencl-headers ocl-icd git curl
else
    echo "[!] Unsupported OS: $OS"
    exit 1
fi

# Detect GPU and install drivers
if lspci | grep -i nvidia; then
    echo "[*] NVIDIA GPU detected. Checking for drivers..."
    if ! command -v nvidia-smi &>/dev/null; then
        echo "[*] Installing NVIDIA drivers..."
        sudo apt-get install -y nvidia-driver nvidia-cuda-toolkit
    fi
elif lspci | grep -i amd; then
    echo "[*] AMD GPU detected. Please ensure amdgpu-pro or ROCm drivers are installed."
    # On Ubuntu, ROCm installation is more involved; print instructions.
elif lspci | grep -i 'intel .*graphics'; then
    echo "[*] Intel GPU detected. Installing intel-opencl-icd..."
    sudo apt-get install -y intel-opencl-icd
else
    echo "[!] No supported GPU detected. Will fall back to CPU."
fi

# Install Rust toolchain if not present
if ! command -v cargo &>/dev/null; then
    echo "[*] Installing Rust (via rustup)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Clone if not already here
if [ ! -d "bip39-solver-gpu" ]; then
    echo "[*] Cloning bip39-solver-gpu..."
    git clone https://github.com/0xVijay/bip39-solver-gpu.git
    cd bip39-solver-gpu
fi

echo "[*] Building bip39-solver-gpu with Cargo..."
cargo build --release

echo "[*] Install complete. Usage example:"
echo "  ./target/release/solver --config config.json"

echo "=== Done ==="
