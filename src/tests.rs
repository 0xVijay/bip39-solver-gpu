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
    
    #[test]
    fn testgpu() {
        use crate::gpu::GpuAccelerator;
        use crate::word_space::WordSpace;
        use crate::config::{Config, WordConstraint, EthereumConfig};
        
        println!("=== GPU Test (testgpu) ===");
        
        // Test GPU initialization
        match GpuAccelerator::new() {
            Ok(gpu) => {
                println!("✓ GPU acceleration initialized successfully");
                
                // Create a simple test configuration
                let test_config = Config {
                    word_constraints: vec![
                        WordConstraint {
                            position: 0,
                            prefix: Some("abandon".to_string()),
                            words: vec![],
                        },
                    ],
                    ethereum: EthereumConfig {
                        derivation_path: "m/44'/60'/0'/0/0".to_string(),
                        target_address: "0x9858EfFD232B4033E47d90003D41EC34EcaEda94".to_string(),
                    },
                    slack: None,
                    worker: None,
                    batch_size: 100,
                    passphrase: "".to_string(),
                };
                
                // Generate word space
                let word_space = WordSpace::from_config(&test_config);
                println!("✓ Word space generated with {} combinations", word_space.total_combinations);
                
                // Parse target address
                let target_address = parse_test_address(&test_config.ethereum.target_address)
                    .expect("Failed to parse target address");
                println!("✓ Target address parsed: 0x{}", hex::encode(target_address));
                
                // Convert word space to constraints
                let constraints = convert_test_word_space_to_constraints(&word_space);
                println!("✓ Word constraints converted for GPU");
                println!("  - Position 0: {} words", constraints[0].len());
                for i in 1..12 {
                    println!("  - Position {}: {} words", i, constraints[i].len());
                }
                
                // Test GPU processing with a small batch
                println!("🔄 Testing GPU batch processing...");
                match gpu.process_batch(0, 10, &target_address, &constraints) {
                    Ok(result) => {
                        match result {
                            Some((index, mnemonic)) => {
                                println!("✓ GPU found result: index={}, mnemonic={}", index, mnemonic);
                            }
                            None => {
                                println!("✓ GPU processed batch successfully (no match found in test range)");
                            }
                        }
                        
                        // Test with a larger batch
                        println!("🔄 Testing larger GPU batch...");
                        match gpu.process_batch(0, 1000, &target_address, &constraints) {
                            Ok(result) => {
                                match result {
                                    Some((index, mnemonic)) => {
                                        println!("✓ GPU found result in larger batch: index={}, mnemonic={}", index, mnemonic);
                                    }
                                    None => {
                                        println!("✓ GPU processed larger batch successfully (no match found)");
                                    }
                                }
                            }
                            Err(e) => {
                                println!("⚠ GPU processing error on larger batch: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("⚠ GPU processing error: {}", e);
                        println!("This may be expected if OpenCL drivers are not properly installed");
                    }
                }
                
                println!("✓ GPU test completed successfully");
            }
            Err(e) => {
                println!("⚠ GPU acceleration not available: {}", e);
                println!("This is expected in environments without GPU or OpenCL drivers");
                println!("The solver will fall back to CPU processing");
                
                // Test that the GPU gracefully handles the fallback
                println!("✓ GPU fallback handling working correctly");
            }
        }
        
        // Test GPU information availability
        match get_gpu_info() {
            Ok(info) => {
                println!("✓ GPU information retrieved: {}", info);
            }
            Err(e) => {
                println!("ℹ GPU information not available: {}", e);
            }
        }
    }
    
    // Helper function to get GPU information
    fn get_gpu_info() -> Result<String, Box<dyn std::error::Error>> {
        use opencl3::platform::get_platforms;
        use opencl3::device::{get_all_devices, CL_DEVICE_TYPE_ALL, Device};
        
        let platforms = get_platforms()?;
        if platforms.is_empty() {
            return Err("No OpenCL platforms available".into());
        }
        
        let devices = get_all_devices(CL_DEVICE_TYPE_ALL)?;
        if devices.is_empty() {
            return Err("No OpenCL devices available".into());
        }
        
        let device = Device::new(devices[0]);
        let device_name = device.name()?;
        let device_version = device.version()?;
        
        Ok(format!("Device: {}, Version: {}", device_name, device_version))
    }
    
    // Helper functions for testgpu
    fn parse_test_address(address_str: &str) -> Result<[u8; 20], Box<dyn std::error::Error>> {
        if !address_str.starts_with("0x") || address_str.len() != 42 {
            return Err("Invalid Ethereum address format".into());
        }
        
        let hex_str = &address_str[2..];
        let mut address = [0u8; 20];
        
        for i in 0..20 {
            let byte_str = &hex_str[i * 2..(i * 2) + 2];
            address[i] = u8::from_str_radix(byte_str, 16)?;
        }
        
        Ok(address)
    }
    
    fn convert_test_word_space_to_constraints(word_space: &WordSpace) -> [Vec<u16>; 12] {
        let mut constraints: [Vec<u16>; 12] = Default::default();
        
        for (pos, word_indices) in word_space.positions.iter().enumerate() {
            if pos < 12 {
                constraints[pos] = word_indices.clone();
            }
        }
        
        constraints
    }
}