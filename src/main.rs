use std::env;
use std::time::Instant;

mod config;
mod word_lut;
mod candidate_gen;
mod bip39;
mod bip44;
mod eth_addr;
mod gpu_worker;
mod cpu_worker;
mod utils;

use config::Config;
use word_lut::WordLut;
use candidate_gen::CandidateGenerator;
use gpu_worker::GpuWorker;
use cpu_worker::CpuWorker;

#[derive(Debug)]
struct WorkResult {
    mnemonic: String,
    address: String,
    offset: u128,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <config.json>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} config.json", args[0]);
        std::process::exit(1);
    }

    let config_path = &args[1];
    
    // Validate config file path
    if !std::path::Path::new(config_path).exists() {
        eprintln!("Error: Config file '{}' not found", config_path);
        std::process::exit(1);
    }
    
    let config = Config::load(config_path)?;

    // Build word lookup table
    println!("[INFO] Building BIP39 word lookup table...");
    let word_lut = WordLut::new();

    // Generate candidate space
    println!("[INFO] Analyzing word constraints...");
    let candidate_gen = CandidateGenerator::new(&config, &word_lut)?;
    
    println!("[INFO] {} candidates", candidate_gen.total_combinations());

    // Debug: Test the known mnemonic directly
    if std::env::var("DEBUG").is_ok() {
        println!("[DEBUG] Testing known mnemonic...");
        let test_mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        if let Ok(seed) = bip39::Bip39::mnemonic_to_seed(test_mnemonic, &config.ethereum.passphrase) {
            if let Ok(private_key) = bip44::Bip44::derive_private_key(&seed, &config.ethereum.derivation_path) {
                if let Ok(address) = eth_addr::EthAddr::private_key_to_address(&private_key) {
                    println!("[DEBUG] Known mnemonic produces address: {}", address);
                    println!("[DEBUG] Target address: {}", config.ethereum.target_address);
                    println!("[DEBUG] Addresses match: {}", eth_addr::EthAddr::addresses_equal(&address, &config.ethereum.target_address));
                }
            }
        }
    }

    // Auto-detect GPU or fallback to CPU
    let gpu_worker = GpuWorker::new();
    if gpu_worker.is_cuda_available() {
        println!("[INFO] CUDA device 0 detected");
        run_gpu_search(&config, &candidate_gen, &word_lut, &gpu_worker)?;
    } else {
        println!("[INFO] No CUDA device found, using CPU");
        run_cpu_search(&config, &candidate_gen, &word_lut)?;
    }

    Ok(())
}

fn run_gpu_search(
    config: &Config,
    candidate_gen: &CandidateGenerator,
    word_lut: &WordLut,
    gpu_worker: &GpuWorker,
) -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = 64000; // As specified in PRD
    let total = candidate_gen.total_combinations();
    
    for batch_start in (0..total).step_by(batch_size) {
        let batch_end = std::cmp::min(batch_start + batch_size as u128, total);
        let batch_candidates = candidate_gen.generate_batch(batch_start, batch_end - batch_start)?;
        
        if let Some(result) = gpu_worker.process_batch(&batch_candidates, config, word_lut)? {
            println!("[INFO] Address found: {}", result);
            return Ok(());
        }
        
        if batch_end >= total {
            break;
        }
    }
    
    println!("[INFO] Search completed. No matching mnemonic found.");
    Ok(())
}

fn run_cpu_search(
    config: &Config,
    candidate_gen: &CandidateGenerator,
    word_lut: &WordLut,
) -> Result<(), Box<dyn std::error::Error>> {
    let cpu_worker = CpuWorker::new();
    let start_time = Instant::now();
    
    let result = cpu_worker.search_parallel(config, candidate_gen, word_lut)?;
    
    match result {
        Some(mnemonic) => {
            println!("[INFO] Address found: {}", mnemonic);
        }
        None => {
            println!("[INFO] Search completed. No matching mnemonic found.");
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("[INFO] Search took {:.2} seconds", elapsed.as_secs_f64());
    
    Ok(())
}
