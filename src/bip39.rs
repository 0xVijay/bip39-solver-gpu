use bip39::{Mnemonic, Language};
use pbkdf2::pbkdf2;
use hmac::Hmac;
use sha2::Sha512;
use std::str::FromStr;

type HmacSha512 = Hmac<Sha512>;

/// BIP39 entropy-checksum and seed generation
pub struct Bip39 {
}

impl Bip39 {
    /// Generate seed from mnemonic and passphrase using PBKDF2-HMAC-SHA512
    pub fn mnemonic_to_seed(mnemonic: &str, passphrase: &str) -> Result<[u8; 64], Box<dyn std::error::Error>> {
        // Validate mnemonic first
        let _mnemonic = Mnemonic::from_str(mnemonic)?;
        
        // BIP39 seed generation using PBKDF2-HMAC-SHA512
        // Salt = "mnemonic" + passphrase
        let salt = format!("mnemonic{}", passphrase);
        let mut seed = [0u8; 64];
        
        pbkdf2::<HmacSha512>(
            mnemonic.as_bytes(),
            salt.as_bytes(),
            2048, // BIP39 specifies 2048 iterations
            &mut seed
        )?;
        
        Ok(seed)
    }
    
    /// Validate mnemonic checksum
    pub fn validate_mnemonic(mnemonic: &str) -> bool {
        Mnemonic::from_str(mnemonic).is_ok()
    }
    
    /// Create mnemonic from word indices (with checksum validation)
    pub fn indices_to_mnemonic(word_indices: &[u16; 12]) -> Result<String, Box<dyn std::error::Error>> {
        let wordlist = Language::English.word_list();
        let mut words = Vec::with_capacity(12);
        
        for &index in word_indices {
            if (index as usize) >= wordlist.len() {
                return Err(format!("Invalid word index: {}", index).into());
            }
            words.push(wordlist[index as usize]);
        }
        
        let mnemonic_str = words.join(" ");
        
        // Validate the mnemonic has correct checksum
        match Mnemonic::from_str(&mnemonic_str) {
            Ok(_) => Ok(mnemonic_str),
            Err(_) => Err("Invalid mnemonic checksum".into()),
        }
    }
    
    /// Fast mnemonic validation without full parsing (for brute force)
    pub fn quick_validate_indices(word_indices: &[u16; 12]) -> bool {
        let wordlist = Language::English.word_list();
        
        // Check all indices are valid
        for &index in word_indices {
            if (index as usize) >= wordlist.len() {
                return false;
            }
        }
        
        // Build mnemonic string
        let words: Vec<&str> = word_indices.iter()
            .map(|&i| wordlist[i as usize])
            .collect();
        let mnemonic_str = words.join(" ");
        
        // Quick checksum validation
        Mnemonic::from_str(&mnemonic_str).is_ok()
    }
    
    /// Get entropy from word indices (for advanced users)
    pub fn indices_to_entropy(word_indices: &[u16; 12]) -> Result<[u8; 16], Box<dyn std::error::Error>> {
        // Convert 12 words (11 bits each) to 132 bits total
        // First 128 bits are entropy, last 4 bits are checksum
        let mut entropy_with_checksum = 0u128;
        
        for &index in word_indices {
            entropy_with_checksum = (entropy_with_checksum << 11) | (index as u128);
        }
        
        // Extract entropy (first 128 bits)
        let entropy = (entropy_with_checksum >> 4) as u128;
        
        // Convert to byte array
        let mut entropy_bytes = [0u8; 16];
        for i in 0..16 {
            entropy_bytes[15 - i] = ((entropy >> (i * 8)) & 0xff) as u8;
        }
        
        Ok(entropy_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_known_mnemonic_to_seed() {
        // Test with known mnemonic
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let seed = Bip39::mnemonic_to_seed(mnemonic, "").unwrap();
        
        // Known expected seed for this mnemonic
        let expected_hex = "5eb00bbddcf069084889a8ab9155568165f5c453ccb85e70811aaed6f6da5fc19a5ac40b389cd370d086206dec8aa6c43daea6690f20ad3d8d48b2d2ce9e38e4";
        let expected_bytes = hex::decode(expected_hex).unwrap();
        
        assert_eq!(seed.to_vec(), expected_bytes);
    }
    
    #[test]
    fn test_validate_mnemonic() {
        // Valid mnemonic
        assert!(Bip39::validate_mnemonic("abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"));
        
        // Invalid mnemonic (wrong checksum)
        assert!(!Bip39::validate_mnemonic("abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon"));
    }
    
    #[test]
    fn test_indices_to_mnemonic() {
        // Known valid indices for "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        let indices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]; // "about" is index 3
        
        let mnemonic = Bip39::indices_to_mnemonic(&indices).unwrap();
        assert_eq!(mnemonic, "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about");
    }
    
    #[test]
    fn test_quick_validate_indices() {
        // Valid indices
        let valid_indices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3];
        assert!(Bip39::quick_validate_indices(&valid_indices));
        
        // Invalid indices (wrong checksum)
        let invalid_indices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(!Bip39::quick_validate_indices(&invalid_indices));
    }
}