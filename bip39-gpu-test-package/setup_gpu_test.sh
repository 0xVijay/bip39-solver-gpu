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
