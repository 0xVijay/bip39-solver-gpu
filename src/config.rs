use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WordConstraint {
    /// Position in the 12-word mnemonic (0-11)
    pub position: usize,
    /// Known prefix for the word at this position
    pub prefix: Option<String>,
    /// List of possible words for this position (if empty, all words are possible)
    pub words: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EthereumConfig {
    /// Derivation path for Ethereum addresses (e.g., "m/44'/60'/0'/0/0")
    pub derivation_path: String,
    /// Target Ethereum address to find (with 0x prefix)
    pub target_address: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SlackConfig {
    /// Slack webhook URL for notifications
    pub webhook_url: String,
    /// Channel to send notifications to (optional)
    pub channel: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WorkerConfig {
    /// URL of work server for distributed processing
    pub server_url: String,
    /// Secret key for work server authentication
    pub secret: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GpuConfig {
    /// GPU backend to use ("opencl" or "cuda")
    pub backend: String,
    /// List of GPU device IDs to use (empty = use all available)
    pub devices: Vec<u32>,
    /// Enable multi-GPU processing
    pub multi_gpu: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        GpuConfig {
            backend: "auto".to_string(), // Auto-detect best backend
            devices: vec![], // Use all available devices
            multi_gpu: true, // Enable multi-GPU by default
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    /// Word constraints for generating candidate mnemonics
    pub word_constraints: Vec<WordConstraint>,
    /// Ethereum-specific configuration
    pub ethereum: EthereumConfig,
    /// Slack notification configuration (optional)
    pub slack: Option<SlackConfig>,
    /// Worker/distributed processing configuration (optional)
    pub worker: Option<WorkerConfig>,
    /// GPU configuration (optional)
    pub gpu: Option<GpuConfig>,
    /// Batch size for GPU processing
    pub batch_size: u64,
    /// BIP39 passphrase (empty string if none)
    pub passphrase: String,
}

impl Config {
    /// Load configuration from a JSON file
    pub fn load(path: &str) -> Result<Config, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a JSON file
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Create a default configuration template
    pub fn default() -> Config {
        Config {
            word_constraints: vec![
                WordConstraint {
                    position: 0,
                    prefix: Some("aban".to_string()),
                    words: vec![],
                },
                WordConstraint {
                    position: 11,
                    prefix: None,
                    words: vec![
                        "abandon".to_string(),
                        "ability".to_string(),
                        "about".to_string(),
                    ],
                },
            ],
            ethereum: EthereumConfig {
                derivation_path: "m/44'/60'/0'/0/0".to_string(),
                target_address: "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f".to_string(),
            },
            slack: Some(SlackConfig {
                webhook_url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK".to_string(),
                channel: Some("#notifications".to_string()),
            }),
            worker: Some(WorkerConfig {
                server_url: "http://localhost:3000".to_string(),
                secret: "your-secret-key".to_string(),
            }),
            gpu: Some(GpuConfig {
                backend: "opencl".to_string(),
                devices: vec![], // Empty = use all available devices
                multi_gpu: true,
            }),
            batch_size: 1000000,
            passphrase: "".to_string(),
        }
    }
}
