#!/bin/bash
set -e

echo "==== BIP39 Solver GPU: Automated Installer, Builder & Quick Test ===="

# Check for rust/cargo
if ! command -v cargo &> /dev/null; then
    echo "Rust/Cargo not found. Installing Rust toolchain..."
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    source $HOME/.cargo/env
else
    echo "Rust and Cargo found."
fi

echo "Ensuring OpenCL headers (for GPU support)..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    sudo apt-get install -y ocl-icd-opencl-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "On macOS, OpenCL is included."
fi

echo "Building project..."
cargo build --release

echo "Running standalone test with example_test_config.json:"
./target/release/bip39-solver-gpu --config example_test_config.json --mode standalone

echo ""
echo "==== Manual Server/Worker Launch Example ===="
echo "To launch a job server:"
echo "  ./target/release/bip39-server --config example_test_config.json"
echo "To launch a worker:"
echo "  ./target/release/bip39-worker --config example_test_config.json --worker-id test-gpu-worker"

echo ""
echo "==== Done! Check output above for results and next steps. ===="
