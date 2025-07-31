# BIP39 Solver GPU - Ethereum Edition

This project iterates through possible BIP39 mnemonics to find those that generate a target Ethereum address. It has been migrated from the original Bitcoin implementation to support Ethereum wallet cracking using GPU acceleration.

## Features

- **Ethereum Support**: Derives Ethereum addresses using BIP44 derivation path `m/44'/60'/0'/0/0`
- **GPU Acceleration**: Uses OpenCL for high-performance parallel processing (currently CPU fallback)
- **Configurable Constraints**: Specify known word prefixes or exact words for any mnemonic position
- **Slack Notifications**: Get notified when a matching mnemonic is found
- **Progress Tracking**: Real-time progress reporting and rate monitoring

## Configuration

The tool uses a JSON configuration file instead of command-line flags. Run without arguments to see a sample configuration:

```bash
./bip39-solver-gpu --config config.json
```

### Sample Configuration

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
- **worker**: Distributed processing settings (optional, for future use)
- **batch_size**: Number of mnemonics to process in each batch
- **passphrase**: BIP39 passphrase (empty string if none)

## Usage

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

4. **Run**:
   ```bash
   ./target/release/bip39-solver-gpu --config config.json
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

## Performance

The current CPU implementation processes approximately 2,500-3,000 mnemonics per second. GPU acceleration will significantly improve this rate when fully implemented.

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

## Security Considerations

- This tool is for legitimate recovery purposes only
- Never use on wallets that don't belong to you
- The search space for unrestricted mnemonics is computationally infeasible
- Always verify recovered mnemonics independently

## Roadmap

- [ ] Complete GPU OpenCL kernel implementation for Ethereum
- [ ] Proper BIP32 hierarchical deterministic derivation
- [ ] Support for custom derivation paths
- [ ] Distributed processing across multiple machines
- [ ] Hardware wallet integration for verification

## Migration from Bitcoin

This is a migration of the original Bitcoin BIP39 solver. Key changes:

- Replaced double-SHA256 with Keccak-256 for address derivation
- Updated from Bitcoin address format to Ethereum address format
- Changed from BIP44 Bitcoin path to Ethereum path
- Added configuration file support
- Replaced hardcoded values with configurable options

## License

This project maintains the same license as the original Bitcoin implementation.