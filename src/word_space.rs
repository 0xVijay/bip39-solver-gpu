use crate::config::Config;

/// BIP39 word list (first few words for testing - in real implementation, include all 2048 words)
const BIP39_WORDS: &[&str] = &[
    "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract", "absurd",
    "abuse", "access", "accident", "account", "accuse", "achieve", "acid", "acoustic", "acquire",
    "across", "act", "action", "actor", "actress", "actual", "adapt", "add", "addict", "address",
    "adjust", "admit", "adult", "advance", "advice", "aerobic", "affair", "afford", "afraid",
    "again", "age", "agent",
    // ... (in real implementation, include all 2048 BIP39 words)
];

#[derive(Debug, Clone)]
pub struct WordSpace {
    /// For each position (0-11), list of possible word indices
    pub positions: Vec<Vec<u16>>,
    /// Total number of possible combinations
    pub total_combinations: u128,
}

impl WordSpace {
    /// Generate word space based on configuration constraints
    pub fn from_config(config: &Config) -> WordSpace {
        let mut positions: Vec<Vec<u16>> = vec![Vec::new(); 12];

        // Initialize all positions with all possible words
        for pos in 0..12 {
            for (word_idx, _word) in BIP39_WORDS.iter().enumerate() {
                positions[pos].push(word_idx as u16);
            }
        }

        // Apply word constraints
        for constraint in &config.word_constraints {
            if constraint.position >= 12 {
                continue; // Skip invalid positions
            }

            let mut valid_indices = Vec::new();

            if !constraint.words.is_empty() {
                // If specific words are provided, use only those
                for word in &constraint.words {
                    if let Some(idx) = BIP39_WORDS.iter().position(|&w| w == word) {
                        valid_indices.push(idx as u16);
                    }
                }
            } else if let Some(prefix) = &constraint.prefix {
                // If prefix is provided, find all words that start with it
                for (word_idx, word) in BIP39_WORDS.iter().enumerate() {
                    if word.starts_with(prefix) {
                        valid_indices.push(word_idx as u16);
                    }
                }
            }

            if !valid_indices.is_empty() {
                positions[constraint.position] = valid_indices;
            }
        }

        // Calculate total combinations
        let total_combinations = positions.iter().map(|pos| pos.len() as u128).product();

        WordSpace {
            positions,
            total_combinations,
        }
    }

    /// Convert a combination index to word indices for each position
    pub fn index_to_words(&self, mut index: u128) -> Option<[u16; 12]> {
        if index >= self.total_combinations {
            return None;
        }

        let mut result = [0u16; 12];

        // Convert index to word indices (reverse order for little-endian style)
        for pos in (0..12).rev() {
            let pos_size = self.positions[pos].len() as u128;
            if pos_size == 0 {
                return None;
            }

            let word_idx_in_pos = (index % pos_size) as usize;
            result[pos] = self.positions[pos][word_idx_in_pos];
            index /= pos_size;
        }

        Some(result)
    }

    /// Convert word indices to actual words
    pub fn words_to_mnemonic(word_indices: &[u16; 12]) -> Option<String> {
        let mut words = Vec::new();

        for &word_idx in word_indices {
            if (word_idx as usize) >= BIP39_WORDS.len() {
                return None;
            }
            words.push(BIP39_WORDS[word_idx as usize]);
        }

        Some(words.join(" "))
    }

    /// Get the word at a specific index in the BIP39 word list
    pub fn get_word(index: u16) -> Option<&'static str> {
        BIP39_WORDS.get(index as usize).copied()
    }

    /// Get all BIP39 words
    pub fn get_all_words() -> &'static [&'static str] {
        BIP39_WORDS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, EthereumConfig, WordConstraint};

    #[test]
    fn test_word_space_creation() {
        let config = Config {
            word_constraints: vec![
                WordConstraint {
                    position: 0,
                    prefix: Some("aban".to_string()),
                    words: vec![],
                },
                WordConstraint {
                    position: 1,
                    prefix: None,
                    words: vec!["ability".to_string(), "able".to_string()],
                },
            ],
            ethereum: EthereumConfig {
                derivation_path: "m/44'/60'/0'/0/0".to_string(),
                target_address: "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f".to_string(),
            },
            slack: None,
            worker: None,
            gpu: None,
            batch_size: 1000,
            passphrase: "".to_string(),
        };

        let word_space = WordSpace::from_config(&config);

        // Position 0 should have words starting with "aban"
        assert_eq!(word_space.positions[0], vec![0]); // "abandon"

        // Position 1 should have exactly 2 words
        assert_eq!(word_space.positions[1], vec![1, 2]); // "ability", "able"

        // Other positions should have all words
        for pos in 2..12 {
            assert_eq!(word_space.positions[pos].len(), BIP39_WORDS.len());
        }
    }

    #[test]
    fn test_index_to_words() {
        let config = Config::default();
        let word_space = WordSpace::from_config(&config);

        // Test converting index 0 to words
        let words = word_space.index_to_words(0);
        assert!(words.is_some());
        let words = words.unwrap();
        assert_eq!(words.len(), 12);

        // Test converting to mnemonic
        let mnemonic = WordSpace::words_to_mnemonic(&words);
        assert!(mnemonic.is_some());
    }
}
