use crate::config::Config;
use crate::word_lut::WordLut;

/// Cartesian product iterator for generating mnemonic candidates
/// Handles word constraints and generates all valid combinations
pub struct CandidateGenerator {
    /// For each position (0-11), list of possible word indices
    positions: Vec<Vec<u16>>,
    /// Total number of possible combinations
    total_combinations: u128,
}

impl CandidateGenerator {
    /// Create new candidate generator from config
    pub fn new(config: &Config, word_lut: &WordLut) -> Result<Self, Box<dyn std::error::Error>> {
        let mut positions: Vec<Vec<u16>> = vec![Vec::new(); 12];
        
        // Initialize all positions with all possible words by default
        for pos in 0..12 {
            for word_idx in 0..word_lut.word_count() {
                positions[pos].push(word_idx as u16);
            }
        }
        
        // Apply word constraints
        for constraint in &config.word_constraints {
            if constraint.position >= 12 {
                continue; // Skip invalid positions
            }
            
            let mut valid_indices = Vec::new();
            
            // If specific words are provided, use only those
            for word in &constraint.words {
                if let Some(idx) = word_lut.word_to_index(word) {
                    valid_indices.push(idx);
                } else {
                    if std::env::var("DEBUG").is_ok() {
                        println!("[WARN] Word '{}' at position {} not found in BIP39 wordlist", word, constraint.position);
                    }
                }
            }
            
            if !valid_indices.is_empty() {
                if std::env::var("DEBUG").is_ok() {
                    println!("[DEBUG] Position {}: {} valid words out of {} specified", 
                        constraint.position, valid_indices.len(), constraint.words.len());
                }
                positions[constraint.position] = valid_indices;
            } else {
                if std::env::var("DEBUG").is_ok() {
                    println!("[WARN] Position {} has no valid words, keeping all 2048", constraint.position);
                }
            }
        }
        
        // Calculate total combinations
        let mut total = 1u128;
        for pos in &positions {
            if pos.is_empty() {
                return Err("Position with no valid words found".into());
            }
            total = total.checked_mul(pos.len() as u128)
                .ok_or("Too many combinations (overflow)")?;
        }
        
        Ok(Self {
            positions,
            total_combinations: total,
        })
    }
    
    /// Get total number of combinations
    pub fn total_combinations(&self) -> u128 {
        self.total_combinations
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
    
    /// Generate a batch of candidates starting from offset
    pub fn generate_batch(&self, start_offset: u128, count: u128) -> Result<Vec<[u16; 12]>, Box<dyn std::error::Error>> {
        let mut batch = Vec::new();
        let end_offset = std::cmp::min(start_offset + count, self.total_combinations);
        
        for i in start_offset..end_offset {
            if let Some(word_indices) = self.index_to_words(i) {
                batch.push(word_indices);
            }
        }
        
        Ok(batch)
    }
    
    /// Get word options for a specific position
    pub fn position_words(&self, position: usize) -> &[u16] {
        if position < 12 {
            &self.positions[position]
        } else {
            &[]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EthereumConfig;
    
    #[test]
    fn test_candidate_generator() {
        let word_lut = WordLut::new();
        
        // Create config with limited word constraints
        let config = Config {
            wallet_type: "ethereum".to_string(),
            mnemonic_length: 12,
            ethereum: EthereumConfig {
                target_address: "0x9858EfFD232B4033E47d90003D41EC34EcaEda94".to_string(),
                derivation_path: "m/44'/60'/0'/0/0".to_string(),
                passphrase: "".to_string(),
            },
            word_constraints: vec![
                crate::config::WordConstraint {
                    position: 0,
                    words: vec!["abandon".to_string(), "ability".to_string()],
                },
                crate::config::WordConstraint {
                    position: 11,
                    words: vec!["about".to_string()],
                },
            ],
        };
        
        let gen = CandidateGenerator::new(&config, &word_lut).unwrap();
        
        // Should have 2 words for position 0, 1 word for position 11, 
        // and 2048 words for all other positions
        let expected_combinations = 2u128 * 2048u128.pow(10) * 1u128;
        assert_eq!(gen.total_combinations(), expected_combinations);
        
        // Test index to words conversion
        let words = gen.index_to_words(0).unwrap();
        assert_eq!(words[0], 0); // "abandon"
        assert_eq!(words[11], word_lut.word_to_index("about").unwrap());
    }
    
    #[test]
    fn test_small_combination_space() {
        let word_lut = WordLut::new();
        
        // Create config that matches the PRD example with 486 combinations
        let config = Config {
            wallet_type: "ethereum".to_string(),
            mnemonic_length: 12,
            ethereum: EthereumConfig {
                target_address: "0x543Bd35F52147370C0deCBd440863bc2a002C5c5".to_string(),
                derivation_path: "m/44'/60'/0'/0/2".to_string(),
                passphrase: "".to_string(),
            },
            word_constraints: vec![
                crate::config::WordConstraint { position: 0, words: vec!["frequent".to_string(), "frame".to_string()] },
                crate::config::WordConstraint { position: 1, words: vec!["lucky".to_string()] },
                crate::config::WordConstraint { position: 2, words: vec!["inquiry".to_string(), "input".to_string(), "inner".to_string()] },
                crate::config::WordConstraint { position: 3, words: vec!["vendor".to_string()] },
                crate::config::WordConstraint { position: 4, words: vec!["engine".to_string(), "energy".to_string(), "engage".to_string()] },
                crate::config::WordConstraint { position: 5, words: vec!["dragon".to_string(), "draft".to_string(), "drama".to_string()] },
                crate::config::WordConstraint { position: 6, words: vec!["horse".to_string(), "honor".to_string(), "hope".to_string()] },
                crate::config::WordConstraint { position: 7, words: vec!["gorilla".to_string()] },
                crate::config::WordConstraint { position: 8, words: vec!["pear".to_string(), "peace".to_string(), "peak".to_string()] },
                crate::config::WordConstraint { position: 9, words: vec!["old".to_string(), "ocean".to_string(), "offer".to_string()] },
                crate::config::WordConstraint { position: 10, words: vec!["dance".to_string()] },
                crate::config::WordConstraint { position: 11, words: vec!["shield".to_string()] },
            ],
        };
        
        let gen = CandidateGenerator::new(&config, &word_lut).unwrap();
        
        // Should have exactly 486 combinations (2×1×3×1×3×3×3×1×3×3×1×1)
        assert_eq!(gen.total_combinations(), 486);
    }
}