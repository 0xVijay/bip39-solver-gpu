use k256::{ecdsa::SigningKey, elliptic_curve::sec1::ToEncodedPoint};
use tiny_keccak::{Hasher, Keccak};

/// Ethereum address generation from secp256k1 private keys
pub struct EthAddr {
}

impl EthAddr {
    /// Convert private key to Ethereum address
    pub fn private_key_to_address(private_key: &[u8; 32]) -> Result<String, Box<dyn std::error::Error>> {
        // Create secp256k1 signing key
        let signing_key = SigningKey::from_bytes(private_key.into())?;
        
        // Get the public key
        let verifying_key = signing_key.verifying_key();
        
        // Get uncompressed public key bytes (65 bytes: 0x04 + 32 bytes X + 32 bytes Y)
        let public_key_bytes = verifying_key.to_encoded_point(false).as_bytes().to_vec();
        
        // Remove the 0x04 prefix for uncompressed public key
        if public_key_bytes.len() != 65 || public_key_bytes[0] != 0x04 {
            return Err("Invalid public key format".into());
        }
        let public_key_no_prefix = &public_key_bytes[1..]; // 64 bytes: 32 bytes X + 32 bytes Y
        
        // Compute Keccak-256 hash of the public key
        let mut hasher = Keccak::v256();
        hasher.update(public_key_no_prefix);
        let mut hash = [0u8; 32];
        hasher.finalize(&mut hash);
        
        // Take the last 20 bytes as the address
        let address_bytes = &hash[12..];
        
        // Format as hex string with 0x prefix
        let address = format!("0x{}", hex::encode(address_bytes));
        
        Ok(address)
    }
    
    /// Convert private key to checksummed Ethereum address
    pub fn private_key_to_checksummed_address(private_key: &[u8; 32]) -> Result<String, Box<dyn std::error::Error>> {
        let address = Self::private_key_to_address(private_key)?;
        Ok(Self::to_checksum_address(&address)?)
    }
    
    /// Convert address to EIP-55 checksummed format
    pub fn to_checksum_address(address: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Remove 0x prefix if present
        let address_clean = if address.starts_with("0x") {
            &address[2..]
        } else {
            address
        };
        
        if address_clean.len() != 40 {
            return Err("Invalid address length".into());
        }
        
        // Convert to lowercase
        let address_lower = address_clean.to_lowercase();
        
        // Compute Keccak-256 hash of the lowercase address
        let mut hasher = Keccak::v256();
        hasher.update(address_lower.as_bytes());
        let mut hash = [0u8; 32];
        hasher.finalize(&mut hash);
        
        // Apply checksum: uppercase if corresponding hash nibble >= 8
        let mut checksummed = String::with_capacity(42);
        checksummed.push_str("0x");
        
        for (i, c) in address_lower.chars().enumerate() {
            let hash_nibble = if i % 2 == 0 {
                hash[i / 2] >> 4
            } else {
                hash[i / 2] & 0x0f
            };
            
            if hash_nibble >= 8 {
                checksummed.push(c.to_ascii_uppercase());
            } else {
                checksummed.push(c);
            }
        }
        
        Ok(checksummed)
    }
    
    /// Validate Ethereum address format (with or without checksum)
    pub fn is_valid_address(address: &str) -> bool {
        // Check basic format
        if !address.starts_with("0x") || address.len() != 42 {
            return false;
        }
        
        let address_part = &address[2..];
        
        // Check if all characters are valid hex
        if !address_part.chars().all(|c| c.is_ascii_hexdigit()) {
            return false;
        }
        
        true
    }
    
    /// Compare two addresses (case insensitive)
    pub fn addresses_equal(addr1: &str, addr2: &str) -> bool {
        if !Self::is_valid_address(addr1) || !Self::is_valid_address(addr2) {
            return false;
        }
        
        addr1.to_lowercase() == addr2.to_lowercase()
    }
    
    /// Extract public key from private key (for debugging)
    pub fn private_key_to_public_key(private_key: &[u8; 32]) -> Result<[u8; 64], Box<dyn std::error::Error>> {
        let signing_key = SigningKey::from_bytes(private_key.into())?;
        let verifying_key = signing_key.verifying_key();
        let public_key_bytes = verifying_key.to_encoded_point(false).as_bytes().to_vec();
        
        if public_key_bytes.len() != 65 || public_key_bytes[0] != 0x04 {
            return Err("Invalid public key format".into());
        }
        
        let mut result = [0u8; 64];
        result.copy_from_slice(&public_key_bytes[1..]); // Remove 0x04 prefix
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bip39::Bip39;
    use crate::bip44::Bip44;
    
    #[test]
    fn test_known_private_key_to_address() {
        // Test with known seed and derivation
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let seed = Bip39::mnemonic_to_seed(mnemonic, "").unwrap();
        let private_key = Bip44::derive_private_key(&seed, "m/44'/60'/0'/0/0").unwrap();
        
        let address = EthAddr::private_key_to_address(&private_key).unwrap();
        
        // Should produce deterministic address
        assert!(address.starts_with("0x"));
        assert_eq!(address.len(), 42);
        
        // Validate it's a valid address format
        assert!(EthAddr::is_valid_address(&address));
    }
    
    #[test]
    fn test_checksum_address() {
        let test_address = "0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed";
        let checksummed = EthAddr::to_checksum_address(test_address).unwrap();
        
        // Should be properly checksummed according to EIP-55
        assert_eq!(checksummed, "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed");
    }
    
    #[test]
    fn test_address_validation() {
        // Valid addresses
        assert!(EthAddr::is_valid_address("0x1234567890123456789012345678901234567890"));
        assert!(EthAddr::is_valid_address("0xabcdefABCDEF123456789012345678901234567890"));
        
        // Invalid addresses
        assert!(!EthAddr::is_valid_address("1234567890123456789012345678901234567890")); // No 0x prefix
        assert!(!EthAddr::is_valid_address("0x123456789012345678901234567890123456789")); // Too short
        assert!(!EthAddr::is_valid_address("0x12345678901234567890123456789012345678901")); // Too long
        assert!(!EthAddr::is_valid_address("0x123456789012345678901234567890123456789g")); // Invalid hex
    }
    
    #[test]
    fn test_addresses_equal() {
        let addr1 = "0x1234567890123456789012345678901234567890";
        let addr2 = "0x1234567890123456789012345678901234567890";
        let addr3 = "0x1234567890123456789012345678901234567891";
        
        assert!(EthAddr::addresses_equal(addr1, addr2));
        assert!(!EthAddr::addresses_equal(addr1, addr3));
        
        // Case insensitive comparison
        let addr_lower = "0x1234567890123456789012345678901234567890";
        let addr_upper = "0x1234567890123456789012345678901234567890".to_uppercase();
        assert!(EthAddr::addresses_equal(addr_lower, &addr_upper));
    }
}