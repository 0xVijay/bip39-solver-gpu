use bip39_solver_gpu::*;

fn main() {
    println!("Testing individual components...");
    
    // Test BIP39 seed generation
    let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
    let seed = bip39::Bip39::mnemonic_to_seed(mnemonic, "").unwrap();
    println!("Seed: {}", hex::encode(&seed));
    
    // Test BIP44 derivation
    let private_key = bip44::Bip44::derive_private_key(&seed, "m/44'/60'/0'/0/0").unwrap();
    println!("Private key: {}", hex::encode(&private_key));
    
    // Test Ethereum address generation
    let address = eth_addr::EthAddr::private_key_to_address(&private_key).unwrap();
    println!("Derived address: {}", address);
    
    // Test our target
    let target = "0x9858EfFD232B4033E47d90003D41EC34EcaEda94";
    println!("Target address: {}", target);
    println!("Addresses match: {}", eth_addr::EthAddr::addresses_equal(&address, target));
}