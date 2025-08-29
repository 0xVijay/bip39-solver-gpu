use hmac::{Hmac, Mac};
use sha2::Sha512;
use num_bigint::BigUint;

type HmacSha512 = Hmac<Sha512>;

/// BIP44 key derivation implementation
/// Derives private keys at specified BIP44 paths
pub struct Bip44 {
}

impl Bip44 {
    /// Derive private key from seed at the specified BIP44 path
    pub fn derive_private_key(seed: &[u8; 64], derivation_path: &str) -> Result<[u8; 32], Box<dyn std::error::Error>> {
        // Generate master private key using HMAC-SHA512 with "Bitcoin seed"
        let mut hmac = HmacSha512::new_from_slice(b"Bitcoin seed")?;
        hmac.update(seed);
        let hash = hmac.finalize().into_bytes();
        
        let mut private_key = [0u8; 32];
        private_key.copy_from_slice(&hash[..32]);
        let mut chain_code = [0u8; 32];
        chain_code.copy_from_slice(&hash[32..]);
        
        // Parse derivation path like "m/44'/60'/0'/0/2"
        let path_parts: Vec<&str> = derivation_path.split('/').collect();
        
        if path_parts.is_empty() || path_parts[0] != "m" {
            return Err("Invalid derivation path: must start with 'm/'".into());
        }
        
        // Derive each level
        for part in &path_parts[1..] {
            let (index, hardened) = if part.ends_with('\'') {
                let index_str = &part[..part.len() - 1];
                let index = index_str.parse::<u32>()?;
                (index + 0x80000000, true) // Hardened derivation
            } else {
                let index = part.parse::<u32>()?;
                (index, false)
            };
            
            let (new_private_key, new_chain_code) = Self::derive_child_key(&private_key, &chain_code, index, hardened)?;
            private_key = new_private_key;
            chain_code = new_chain_code;
        }
        
        Ok(private_key)
    }
    
    /// Derive child key using BIP32 derivation
    fn derive_child_key(
        parent_private_key: &[u8; 32],
        parent_chain_code: &[u8; 32],
        index: u32,
        hardened: bool,
    ) -> Result<([u8; 32], [u8; 32]), Box<dyn std::error::Error>> {
        let mut hmac = HmacSha512::new_from_slice(parent_chain_code)?;
        
        if hardened {
            // Hardened derivation: HMAC-SHA512(Key = parent_chain_code, Data = 0x00 || parent_private_key || index)
            hmac.update(&[0x00]);
            hmac.update(parent_private_key);
        } else {
            // Non-hardened derivation: HMAC-SHA512(Key = parent_chain_code, Data = parent_public_key || index)
            // For simplicity, we'll compute public key from private key
            let public_key = Self::private_to_public_key(parent_private_key)?;
            hmac.update(&public_key);
        }
        
        hmac.update(&index.to_be_bytes());
        let hash = hmac.finalize().into_bytes();
        
        // Left 256 bits become the child private key
        let mut child_private_key = [0u8; 32];
        child_private_key.copy_from_slice(&hash[..32]);
        
        // Add parent private key (mod n, where n is secp256k1 order)
        let child_key = Self::add_private_keys(parent_private_key, &child_private_key)?;
        
        // Right 256 bits become the child chain code
        let mut child_chain_code = [0u8; 32];
        child_chain_code.copy_from_slice(&hash[32..]);
        
        Ok((child_key, child_chain_code))
    }
    
    /// Add two private keys modulo secp256k1 order
    fn add_private_keys(key1: &[u8; 32], key2: &[u8; 32]) -> Result<[u8; 32], Box<dyn std::error::Error>> {
        // secp256k1 order: 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        let secp256k1_order = BigUint::parse_bytes(
            b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
            16,
        ).unwrap();
        
        let k1 = BigUint::from_bytes_be(key1);
        let k2 = BigUint::from_bytes_be(key2);
        
        let sum = (k1 + k2) % &secp256k1_order;
        
        // Convert back to 32-byte array
        let sum_bytes = sum.to_bytes_be();
        let mut result = [0u8; 32];
        
        if sum_bytes.len() <= 32 {
            let offset = 32 - sum_bytes.len();
            result[offset..].copy_from_slice(&sum_bytes);
        } else {
            return Err("Private key addition overflow".into());
        }
        
        Ok(result)
    }
    
    /// Convert private key to compressed public key 
    fn private_to_public_key(private_key: &[u8; 32]) -> Result<[u8; 33], Box<dyn std::error::Error>> {
        use k256::ecdsa::SigningKey;
        use k256::elliptic_curve::sec1::ToEncodedPoint;
        
        // Create signing key from private key
        let signing_key = SigningKey::from_bytes(private_key.into())?;
        
        // Get the verifying (public) key
        let verifying_key = signing_key.verifying_key();
        
        // Get compressed public key (33 bytes: 0x02/0x03 prefix + 32 bytes X coordinate)
        let public_key_point = verifying_key.to_encoded_point(true); // true = compressed
        let public_key_bytes = public_key_point.as_bytes();
        
        if public_key_bytes.len() != 33 {
            return Err(format!("Invalid compressed public key length: {}", public_key_bytes.len()).into());
        }
        
        let mut result = [0u8; 33];
        result.copy_from_slice(public_key_bytes);
        
        Ok(result)
    }
    
    /// Parse BIP44 derivation path and validate format
    pub fn parse_derivation_path(path: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let parts: Vec<&str> = path.split('/').collect();
        
        if parts.is_empty() || parts[0] != "m" {
            return Err("Derivation path must start with 'm/'".into());
        }
        
        let mut indices = Vec::new();
        for part in &parts[1..] {
            if part.ends_with('\'') {
                let index_str = &part[..part.len() - 1];
                let index = index_str.parse::<u32>()?;
                indices.push(index + 0x80000000); // Hardened
            } else {
                let index = part.parse::<u32>()?;
                indices.push(index);
            }
        }
        
        Ok(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bip39::Bip39;
    
    #[test]
    fn test_derive_private_key() {
        // Test with known seed
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let seed = Bip39::mnemonic_to_seed(mnemonic, "").unwrap();
        
        // Derive at path m/44'/60'/0'/0/0
        let private_key = Bip44::derive_private_key(&seed, "m/44'/60'/0'/0/0").unwrap();
        
        // Should produce deterministic result
        assert_eq!(private_key.len(), 32);
        assert_ne!(private_key, [0u8; 32]); // Should not be all zeros
    }
    
    #[test]
    fn test_parse_derivation_path() {
        let indices = Bip44::parse_derivation_path("m/44'/60'/0'/0/2").unwrap();
        
        // Should parse to: [44+0x80000000, 60+0x80000000, 0+0x80000000, 0, 2]
        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 44 + 0x80000000);
        assert_eq!(indices[1], 60 + 0x80000000);
        assert_eq!(indices[2], 0 + 0x80000000);
        assert_eq!(indices[3], 0);
        assert_eq!(indices[4], 2);
    }
    
    #[test]
    fn test_invalid_derivation_path() {
        assert!(Bip44::parse_derivation_path("44'/60'/0'/0/2").is_err()); // Missing 'm/'
        assert!(Bip44::parse_derivation_path("m/44x/60'/0'/0/2").is_err()); // Invalid index
    }
}