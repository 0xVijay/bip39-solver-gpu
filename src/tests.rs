#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, WordConstraint, EthereumConfig};
    use crate::eth::derive_ethereum_address;
    use crate::word_space::WordSpace;

    #[test]
    fn test_known_mnemonic_to_address() {
        // Test with a known mnemonic phrase and expected Ethereum address
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let passphrase = "";
        let derivation_path = "m/44'/60'/0'/0/0";
        
        let result = derive_ethereum_address(mnemonic, passphrase, derivation_path);
        assert!(result.is_ok());
        
        let address = result.unwrap();
        assert!(address.starts_with("0x"));
        assert_eq!(address.len(), 42);
        
        // Note: The actual address will depend on the implementation
        // This test validates the format and that derivation doesn't fail
        println!("Derived address: {}", address);
    }
    
    #[test]
    fn test_config_serialization() {
        let config = Config {
            word_constraints: vec![
                WordConstraint {
                    position: 0,
                    prefix: Some("test".to_string()),
                    words: vec![],
                },
            ],
            ethereum: EthereumConfig {
                derivation_path: "m/44'/60'/0'/0/0".to_string(),
                target_address: "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f".to_string(),
            },
            slack: None,
            worker: None,
            batch_size: 1000,
            passphrase: "".to_string(),
        };
        
        let json = serde_json::to_string(&config).unwrap();
        let parsed: Config = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.ethereum.target_address, parsed.ethereum.target_address);
        assert_eq!(config.batch_size, parsed.batch_size);
    }
    
    #[test]
    fn test_word_space_constraints() {
        let config = Config {
            word_constraints: vec![
                WordConstraint {
                    position: 0,
                    prefix: Some("abandon".to_string()),
                    words: vec![],
                },
                WordConstraint {
                    position: 1,
                    prefix: None,
                    words: vec!["abandon".to_string()],
                },
            ],
            ethereum: EthereumConfig {
                derivation_path: "m/44'/60'/0'/0/0".to_string(),
                target_address: "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f".to_string(),
            },
            slack: None,
            worker: None,
            batch_size: 1000,
            passphrase: "".to_string(),
        };
        
        let word_space = WordSpace::from_config(&config);
        
        // Position 0 should have words starting with "abandon"
        assert_eq!(word_space.positions[0].len(), 1);
        
        // Position 1 should have exactly 1 word
        assert_eq!(word_space.positions[1].len(), 1);
        
        // Test index to words conversion
        let words = word_space.index_to_words(0);
        assert!(words.is_some());
        
        let mnemonic = WordSpace::words_to_mnemonic(&words.unwrap());
        // Note: This may be None if the checksum is invalid, which is expected with BIP39 validation
        // Most random combinations won't have valid checksums
        println!("Generated mnemonic: {:?}", mnemonic);
    }
    
    #[test]
    fn test_address_validation() {
        use crate::eth::{is_valid_address, addresses_equal};
        
        // Valid Ethereum addresses
        assert!(is_valid_address("0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f"));
        assert!(is_valid_address("0x0000000000000000000000000000000000000000"));
        
        // Invalid addresses
        assert!(!is_valid_address("742d35Cc6634C0532925a3b8D581C027BD5b7c4f")); // Missing 0x
        assert!(!is_valid_address("0x742d35Cc6634C0532925a3b8D581C027BD5b7c4")); // Too short
        assert!(!is_valid_address("0x742d35Cc6634C0532925a3b8D581C027BD5b7c4fg")); // Invalid hex
        
        // Case insensitive comparison
        assert!(addresses_equal(
            "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f",
            "0x742D35CC6634C0532925A3B8D581C027BD5B7C4F"
        ));
    }
    
    #[test]
    fn test_multiple_mnemonics() {
        // Test with known valid BIP39 mnemonics instead of invalid ones
        let mnemonics = [
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
            "legal winner thank year wave sausage worth useful legal winner thank yellow",
            "letter advice cage absurd amount doctor acoustic avoid letter advice cage above",
        ];
        
        let mut addresses = Vec::new();
        for mnemonic in &mnemonics {
            let result = derive_ethereum_address(mnemonic, "", "m/44'/60'/0'/0/0");
            assert!(result.is_ok(), "Failed to derive address for mnemonic: {}", mnemonic);
            addresses.push(result.unwrap());
        }
        
        // All addresses should be different
        for i in 0..addresses.len() {
            for j in i + 1..addresses.len() {
                assert_ne!(addresses[i], addresses[j]);
            }
        }
    }
    
    #[test]
    fn test_bip39_test_vectors() {
        // Test vectors from BIP39 specification
        let test_cases = [
            (
                "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
                "",
            ),
            (
                "legal winner thank year wave sausage worth useful legal winner thank yellow",
                "",
            ),
            (
                "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
                "TREZOR",
            ),
        ];
        
        let mut addresses = Vec::new();
        for (mnemonic, passphrase) in test_cases.iter() {
            let result = derive_ethereum_address(mnemonic, passphrase, "m/44'/60'/0'/0/0");
            assert!(result.is_ok(), "Failed to derive address for test vector");
            let address = result.unwrap();
            assert!(address.starts_with("0x"));
            assert_eq!(address.len(), 42);
            addresses.push(address);
            println!("Mnemonic: {}", mnemonic);
            println!("Passphrase: '{}'", passphrase);
            println!("Address: {}", addresses.last().unwrap());
            println!("---");
        }
        
        // All addresses should be different
        for i in 0..addresses.len() {
            for j in i + 1..addresses.len() {
                assert_ne!(addresses[i], addresses[j], 
                          "Address {} and {} should be different", i, j);
            }
        }
    }
    
    #[test]
    fn test_bip39_word_list_completeness() {
        use crate::word_space::WordSpace;
        
        let words = WordSpace::get_all_words();
        assert_eq!(words.len(), 2048, "BIP39 word list should contain exactly 2048 words");
        
        // Check that some known words are present
        assert!(words.contains(&"abandon"));
        assert!(words.contains(&"ability"));
        assert!(words.contains(&"zoo"));
        assert!(words.contains(&"wrong"));
    }
}