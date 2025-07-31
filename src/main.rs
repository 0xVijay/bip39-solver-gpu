use std::env;
use std::time::Instant;
use rayon::prelude::*;

mod config;
mod word_space;
mod eth;
mod slack;
mod gpu;
#[cfg(test)]
mod tests;

use config::Config;
use word_space::WordSpace;
use eth::{derive_ethereum_address, addresses_equal};
use slack::SlackNotifier;
use gpu::GpuAccelerator;

#[derive(Debug)]
struct WorkResult {
    mnemonic: String,
    address: String,
    offset: u128,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 || args[1] != "--config" {
        eprintln!("Usage: {} --config <config.json>", args[0]);
        eprintln!("\nExample config:");
        let default_config = Config::default();
        println!("{}", serde_json::to_string_pretty(&default_config)?);
        std::process::exit(1);
    }

    let config_path = &args[2];
    let config = Config::load(config_path)?;
    
    println!("Loaded config from: {}", config_path);
    println!("Target address: {}", config.ethereum.target_address);
    println!("Derivation path: {}", config.ethereum.derivation_path);
    
    // Initialize Slack notifier if configured
    let slack_notifier = config.slack.as_ref().map(|slack_config| {
        SlackNotifier::new(slack_config.clone())
    });

    // Generate word space based on constraints
    let word_space = WordSpace::from_config(&config);
    println!("Total combinations to search: {}", word_space.total_combinations);
    
    // Try to initialize GPU acceleration
    let gpu_accelerator = match GpuAccelerator::new() {
        Ok(gpu) => {
            println!("✓ GPU acceleration enabled");
            Some(gpu)
        }
        Err(e) => {
            println!("⚠ GPU acceleration not available, falling back to CPU: {}", e);
            None
        }
    };
    
    // Notify search start
    if let Some(notifier) = &slack_notifier {
        notifier.notify_search_started(&config.ethereum.target_address, word_space.total_combinations)?;
    }

    let start_time = Instant::now();
    let batch_size = config.batch_size as u128;
    let mut current_offset = 0u128;
    
    // Main search loop
    loop {
        let batch_end = std::cmp::min(current_offset + batch_size, word_space.total_combinations);
        
        println!("Searching batch: {} to {}", current_offset, batch_end);
        
        // Choose GPU or CPU processing based on availability
        let result: Option<WorkResult> = if let Some(ref gpu) = gpu_accelerator {
            // GPU processing
            match gpu.process_batch(
                current_offset,
                (batch_end - current_offset) as u32,
                &parse_target_address(&config.ethereum.target_address)?,
                &convert_word_space_to_constraints(&word_space),
            ) {
                Ok(Some((index, mnemonic))) => {
                    match derive_ethereum_address(
                        &mnemonic, 
                        &config.passphrase, 
                        &config.ethereum.derivation_path
                    ) {
                        Ok(address) => {
                            if addresses_equal(&address, &config.ethereum.target_address) {
                                Some(WorkResult {
                                    mnemonic,
                                    address,
                                    offset: index,
                                })
                            } else {
                                None
                            }
                        }
                        Err(_) => None,
                    }
                }
                Ok(None) => None,
                Err(e) => {
                    eprintln!("GPU processing error: {}, falling back to CPU", e);
                    None
                }
            }
        } else {
            None
        };
        
        // If GPU didn't find anything or isn't available, use CPU processing
        let result = result.or_else(|| {
            (current_offset..batch_end)
                .into_par_iter()
                .find_map_any(|index| {
                    if let Some(word_indices) = word_space.index_to_words(index) {
                        if let Some(mnemonic) = WordSpace::words_to_mnemonic(&word_indices) {
                            match derive_ethereum_address(
                                &mnemonic, 
                                &config.passphrase, 
                                &config.ethereum.derivation_path
                            ) {
                                Ok(address) => {
                                    if addresses_equal(&address, &config.ethereum.target_address) {
                                        return Some(WorkResult {
                                            mnemonic,
                                            address,
                                            offset: index,
                                        });
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Error deriving address for mnemonic: {}", e);
                                }
                            }
                        }
                    }
                    None
                })
        });

        if let Some(work_result) = result {
            println!("🎉 Found matching mnemonic!");
            println!("Mnemonic: {}", work_result.mnemonic);
            println!("Address: {}", work_result.address);
            println!("Offset: {}", work_result.offset);
            
            // Notify via Slack
            if let Some(notifier) = &slack_notifier {
                notifier.notify_solution_found(
                    &work_result.mnemonic,
                    &work_result.address,
                    work_result.offset,
                )?;
            }
            
            return Ok(());
        }

        current_offset = batch_end;
        
        // Report progress
        let elapsed = start_time.elapsed();
        let rate = current_offset as f64 / elapsed.as_secs_f64();
        
        println!("Progress: {}/{} ({:.2}%) - Rate: {:.2} mnemonics/sec - Elapsed: {:?}", 
                 current_offset, 
                 word_space.total_combinations,
                 (current_offset as f64 / word_space.total_combinations as f64) * 100.0,
                 rate,
                 elapsed);
        
        // Periodically notify progress via Slack (every 10 minutes)
        if elapsed.as_secs() % 600 == 0 && elapsed.as_secs() > 0 {
            if let Some(notifier) = &slack_notifier {
                notifier.notify_progress(current_offset, rate, elapsed.as_secs())?;
            }
        }
        
        // Check if we've searched everything
        if current_offset >= word_space.total_combinations {
            println!("Search completed. No matching mnemonic found.");
            break;
        }
    }

    Ok(())
}

/// Parse target Ethereum address to bytes
fn parse_target_address(address_str: &str) -> Result<[u8; 20], Box<dyn std::error::Error>> {
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

/// Convert word space to GPU-compatible format
fn convert_word_space_to_constraints(word_space: &WordSpace) -> [Vec<u16>; 12] {
    let mut constraints: [Vec<u16>; 12] = Default::default();
    
    for (pos, word_indices) in word_space.positions.iter().enumerate() {
        if pos < 12 {
            constraints[pos] = word_indices.clone();
        }
    }
    
    constraints
}
