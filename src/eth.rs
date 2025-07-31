use k256::ecdsa::SigningKey;
use ethers_core::utils::keccak256;
use bip39::{Language, Mnemonic};
use hmac::Hmac;
use pbkdf2::pbkdf2;
use sha2::Sha512;

/// Derive an Ethereum address from a BIP39 mnemonic
pub fn derive_ethereum_address(
    mnemonic: &str,
    passphrase: &str,
    derivation_path: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    // Parse and validate the BIP39 mnemonic
    let mnemonic = Mnemonic::parse_in(Language::English, mnemonic)?;
    
    // Generate the seed using proper PBKDF2
    let seed = mnemonic.to_seed(passphrase);
    
    // Derive the private key using BIP32/BIP44 (simplified implementation)
    let private_key = derive_private_key_from_seed(&seed, derivation_path)?;
    
    // Convert private key to Ethereum address
    let address = private_key_to_address(&private_key)?;
    
    Ok(format!("0x{}", hex::encode(address)))
}

/// Generate seed from mnemonic using proper PBKDF2 (now handled by bip39 crate)
/// This function is kept for compatibility but the bip39 crate handles this properly
fn _generate_seed_from_mnemonic(
    mnemonic: &str,
    passphrase: &str,
) -> Result<[u8; 64], Box<dyn std::error::Error>> {
    let mut seed = [0u8; 64];
    let salt = format!("mnemonic{}", passphrase);
    
    pbkdf2::<Hmac<Sha512>>(
        mnemonic.as_bytes(),
        salt.as_bytes(),
        2048,
        &mut seed,
    )?;
    
    Ok(seed)
}

/// Derive private key from seed using BIP32 (simplified implementation)
/// TODO: Implement full BIP32 hierarchical derivation
fn derive_private_key_from_seed(
    seed: &[u8; 64],
    _derivation_path: &str,
) -> Result<[u8; 32], Box<dyn std::error::Error>> {
    // This is still a simplified derivation
    // Full BIP32 implementation would parse the derivation path and perform hierarchical derivation
    let mut private_key = [0u8; 32];
    
    // Use HMAC-SHA512 for key derivation (closer to BIP32)
    use hmac::{Hmac, Mac};
    let mut mac = Hmac::<Sha512>::new_from_slice(b"Bitcoin seed")?;
    mac.update(seed);
    let result = mac.finalize().into_bytes();
    
    // Take the first 32 bytes as the private key
    private_key.copy_from_slice(&result[..32]);
    
    Ok(private_key)
}

/// Convert private key to Ethereum address
fn private_key_to_address(private_key: &[u8; 32]) -> Result<[u8; 20], Box<dyn std::error::Error>> {
    // Create signing key from private key bytes
    let signing_key = SigningKey::from_bytes(private_key.into())?;
    
    // Get the public key
    let public_key = signing_key.verifying_key();
    
    // Convert to uncompressed public key bytes (64 bytes)
    let public_key_bytes = public_key.to_encoded_point(false);
    let public_key_slice = &public_key_bytes.as_bytes()[1..]; // Skip the 0x04 prefix
    
    // Hash the public key with Keccak-256
    let hash = keccak256(public_key_slice);
    
    // Take the last 20 bytes as the Ethereum address
    let mut address = [0u8; 20];
    address.copy_from_slice(&hash[12..]);
    
    Ok(address)
}

/// Convert address bytes to checksummed Ethereum address string (EIP-55)
pub fn to_checksum_address(address: &[u8; 20]) -> String {
    let address_hex = hex::encode(address);
    let hash = keccak256(address_hex.as_bytes());
    
    let mut result = String::with_capacity(42);
    result.push_str("0x");
    
    for (i, c) in address_hex.chars().enumerate() {
        if c.is_ascii_digit() {
            result.push(c);
        } else {
            // Check if the corresponding bit in the hash is set
            let byte_index = i / 2;
            let bit_index = if i % 2 == 0 { 4 } else { 0 };
            if (hash[byte_index] >> bit_index) & 0x08 != 0 {
                result.push(c.to_ascii_uppercase());
            } else {
                result.push(c.to_ascii_lowercase());
            }
        }
    }
    
    result
}

/// Validate that a string is a valid Ethereum address
pub fn is_valid_address(address: &str) -> bool {
    if !address.starts_with("0x") || address.len() != 42 {
        return false;
    }
    
    let hex_part = &address[2..];
    hex_part.chars().all(|c| c.is_ascii_hexdigit())
}

/// Compare two Ethereum addresses (case-insensitive)
pub fn addresses_equal(addr1: &str, addr2: &str) -> bool {
    if !is_valid_address(addr1) || !is_valid_address(addr2) {
        return false;
    }
    
    addr1.to_lowercase() == addr2.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_address_validation() {
        assert!(is_valid_address("0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f"));
        assert!(!is_valid_address("742d35Cc6634C0532925a3b8D581C027BD5b7c4f")); // Missing 0x
        assert!(!is_valid_address("0x742d35Cc6634C0532925a3b8D581C027BD5b7c4")); // Too short
        assert!(!is_valid_address("0x742d35Cc6634C0532925a3b8D581C027BD5b7c4fg")); // Invalid hex
    }
    
    #[test]
    fn test_address_comparison() {
        let addr1 = "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f";
        let addr2 = "0x742D35CC6634C0532925A3B8D581C027BD5B7C4F";
        assert!(addresses_equal(addr1, addr2));
    }
    
    #[test]
    fn test_checksum_address() {
        let address_bytes = [
            0x74, 0x2d, 0x35, 0xcc, 0x66, 0x34, 0xc0, 0x53, 0x29, 0x25,
            0xa3, 0xb8, 0xd5, 0x81, 0xc0, 0x27, 0xbd, 0x5b, 0x7c, 0x4f,
        ];
        let checksum = to_checksum_address(&address_bytes);
        assert!(checksum.starts_with("0x"));
        assert_eq!(checksum.len(), 42);
    }
    
    #[test]
    fn test_derive_ethereum_address() {
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let result = derive_ethereum_address(mnemonic, "", "m/44'/60'/0'/0/0");
        assert!(result.is_ok());
        let address = result.unwrap();
        assert!(is_valid_address(&address));
    }
}