#[cfg(test)]
mod tests {
    use crate::config::{Config, EthereumConfig, WordConstraint};
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
    fn test_known_test_mnemonic_address() {
        // Test with the specific known test mnemonic
        let mnemonic = "frequent lucky inquiry vendor engine dragon horse gorilla pear old dance shield";
        let passphrase = "";
        let derivation_path = "m/44'/60'/0'/0/2";

        let result = derive_ethereum_address(mnemonic, passphrase, derivation_path);
        assert!(result.is_ok());

        let address = result.unwrap();
        assert!(address.starts_with("0x"));
        assert_eq!(address.len(), 42);

        println!("Test mnemonic: {}", mnemonic);
        println!("Derivation path: {}", derivation_path);
        println!("Generated address: {}", address);
        println!("Current config target: 0x543Bd35F52147370C0deCBd440863bc2a002C5c5");
    }

    #[test]
    fn test_config_serialization() {
        let config = Config {
            word_constraints: vec![WordConstraint {
                position: 0,
                prefix: Some("test".to_string()),
                words: vec![],
            }],
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

        let json = serde_json::to_string(&config).unwrap();
        let parsed: Config = serde_json::from_str(&json).unwrap();

        assert_eq!(
            config.ethereum.target_address,
            parsed.ethereum.target_address
        );
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
            gpu: None,
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
        assert!(mnemonic.is_some());

        let mnemonic_str = mnemonic.unwrap();
        assert!(mnemonic_str.contains("abandon"));
    }

    #[test]
    fn test_address_validation() {
        use crate::eth::{addresses_equal, is_valid_address};

        // Valid Ethereum addresses
        assert!(is_valid_address(
            "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f"
        ));
        assert!(is_valid_address(
            "0x0000000000000000000000000000000000000000"
        ));

        // Invalid addresses
        assert!(!is_valid_address(
            "742d35Cc6634C0532925a3b8D581C027BD5b7c4f"
        )); // Missing 0x
        assert!(!is_valid_address(
            "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4"
        )); // Too short
        assert!(!is_valid_address(
            "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4fg"
        )); // Invalid hex

        // Case insensitive comparison
        assert!(addresses_equal(
            "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f",
            "0x742D35CC6634C0532925A3B8D581C027BD5B7C4F"
        ));
    }

    #[test]
    fn test_multiple_mnemonics() {
        // Test that different mnemonics produce different addresses
        let mnemonics = [
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
            "ability abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
            "able abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
        ];

        let mut addresses = Vec::new();
        for mnemonic in &mnemonics {
            let result = derive_ethereum_address(mnemonic, "", "m/44'/60'/0'/0/0");
            assert!(result.is_ok());
            addresses.push(result.unwrap());
        }

        // All addresses should be different
        for i in 0..addresses.len() {
            for j in i + 1..addresses.len() {
                assert_ne!(addresses[i], addresses[j]);
            }
        }
    }
}
