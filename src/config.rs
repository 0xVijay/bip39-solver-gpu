use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WordConstraint {
    /// Position in the 12-word mnemonic (0-11)
    pub position: usize,
    /// List of possible words for this position
    pub words: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EthereumConfig {
    /// Target Ethereum address to find (with 0x prefix)
    pub target_address: String,
    /// Derivation path for Ethereum addresses (e.g., "m/44'/60'/0'/0/2")
    pub derivation_path: String,
    /// BIP39 passphrase (empty string if none)
    pub passphrase: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    /// Wallet type (must be "ethereum")
    pub wallet_type: String,
    /// Mnemonic length (must be 12)
    pub mnemonic_length: u8,
    /// Ethereum-specific configuration
    pub ethereum: EthereumConfig,
    /// Word constraints for generating candidate mnemonics
    pub word_constraints: Vec<WordConstraint>,
}

impl Config {
    /// Load configuration from a JSON file
    pub fn load(path: &str) -> Result<Config, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&content)?;
        
        // Validate configuration
        if config.wallet_type != "ethereum" {
            return Err("Only 'ethereum' wallet type is supported".into());
        }
        
        if config.mnemonic_length != 12 {
            return Err("Only 12-word mnemonics are supported".into());
        }
        
        Ok(config)
    }

    /// Save configuration to a JSON file
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}
