use bip39::{Language, Mnemonic};
use std::collections::HashMap;
use std::str::FromStr;

/// BIP39 word → u11 index lookup table
/// Optimizes word-to-index conversion for performance
pub struct WordLut {
    /// Map from word string to 11-bit index (0-2047)
    word_to_index: HashMap<String, u16>,
    /// Array of words indexed by u11 value
    index_to_word: Vec<&'static str>,
}

impl WordLut {
    /// Build the lookup table once at startup
    pub fn new() -> Self {
        let wordlist = Language::English.word_list();
        let mut word_to_index = HashMap::new();
        let mut index_to_word = Vec::with_capacity(2048);
        
        for (index, &word) in wordlist.iter().enumerate() {
            word_to_index.insert(word.to_string(), index as u16);
            index_to_word.push(word);
        }
        
        Self {
            word_to_index,
            index_to_word,
        }
    }
    
    /// Convert word string to 11-bit index
    pub fn word_to_index(&self, word: &str) -> Option<u16> {
        self.word_to_index.get(word).copied()
    }
    
    /// Convert 11-bit index to word string
    pub fn index_to_word(&self, index: u16) -> Option<&'static str> {
        if (index as usize) < self.index_to_word.len() {
            Some(self.index_to_word[index as usize])
        } else {
            None
        }
    }
    
    /// Get all BIP39 words
    pub fn all_words(&self) -> &[&'static str] {
        &self.index_to_word
    }
    
    /// Get total number of words (should be 2048)
    pub fn word_count(&self) -> usize {
        self.index_to_word.len()
    }
    
    /// Convert array of word indices to mnemonic string
    pub fn indices_to_mnemonic(&self, indices: &[u16; 12]) -> Option<String> {
        let mut words = Vec::with_capacity(12);
        for &index in indices {
            if let Some(word) = self.index_to_word(index) {
                words.push(word);
            } else {
                return None;
            }
        }
        Some(words.join(" "))
    }
    
    /// Validate that a mnemonic string is valid BIP39
    pub fn validate_mnemonic(&self, mnemonic: &str) -> bool {
        Mnemonic::from_str(mnemonic).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_word_lut_creation() {
        let lut = WordLut::new();
        assert_eq!(lut.word_count(), 2048);
        
        // Test some known words
        assert_eq!(lut.word_to_index("abandon"), Some(0));
        assert_eq!(lut.index_to_word(0), Some("abandon"));
        
        // Test word that should be at end
        assert_eq!(lut.word_to_index("zoo"), Some(2047));
        assert_eq!(lut.index_to_word(2047), Some("zoo"));
    }
    
    #[test]
    fn test_indices_to_mnemonic() {
        let lut = WordLut::new();
        
        // Test with all "abandon" (index 0)
        let indices = [0; 12];
        let mnemonic = lut.indices_to_mnemonic(&indices).unwrap();
        assert_eq!(mnemonic, "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon");
    }
    
    #[test]
    fn test_validate_mnemonic() {
        let lut = WordLut::new();
        
        // Known valid mnemonic
        assert!(lut.validate_mnemonic("abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"));
        
        // Invalid mnemonic (wrong checksum)
        assert!(!lut.validate_mnemonic("abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon"));
    }
}