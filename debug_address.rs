use seed_crack::{bip39::Bip39, bip44::Bip44, eth_addr::EthAddr};

fn main() {
    let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
    println!("Testing mnemonic: {}", mnemonic);
    
    // Generate seed
    let seed = Bip39::mnemonic_to_seed(mnemonic, "").unwrap();
    println!("Seed (hex): {}", hex::encode(&seed));
    
    // Derive private key
    let private_key = Bip44::derive_private_key(&seed, "m/44'/60'/0'/0/0").unwrap();
    println!("Private key (hex): {}", hex::encode(&private_key));
    
    // Generate address
    let address = EthAddr::private_key_to_address(&private_key).unwrap();
    println!("Generated address: {}", address);
    
    println!("Expected address: 0x9858EfFD232B4033E47d90003D41EC34EcaEda94");
    println!("Match: {}", EthAddr::addresses_equal(&address, "0x9858EfFD232B4033E47d90003D41EC34EcaEda94"));
}