use std::env;
use std::time::Instant;
use rayon::prelude::*;

pub mod config;
pub mod word_space;
pub mod eth;
pub mod slack;
pub mod job_types;
pub mod job_server;
pub mod worker_client;
#[cfg(test)]
mod tests;

use config::Config;
use word_space::WordSpace;
use eth::{derive_ethereum_address, addresses_equal};
use slack::SlackNotifier;

#[derive(Debug)]
struct WorkResult {
    mnemonic: String,
    address: String,
    offset: u128,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args[1] != "--config" {
        eprintln!("Usage: {} --config <config.json> [--mode <standalone|worker>]", args[0]);
        eprintln!("\nExample config:");
        let default_config = Config::default();
        println!("{}", serde_json::to_string_pretty(&default_config)?);
        std::process::exit(1);
    }

    let config_path = &args[2];
    let config = Config::load(config_path)?;
    
    // Check if running as worker
    let mode = if args.len() >= 5 && args[3] == "--mode" {
        args[4].clone()
    } else {
        "standalone".to_string()
    };
    
    match mode.as_str() {
        "worker" => {
            // Run as distributed worker
            worker_client::run_worker(&config, None)?;
        }
        "standalone" => {
            // Run standalone version (original behavior)
            run_standalone(&config, config_path)?;
        }
        _ => {
            eprintln!("Invalid mode: {}. Use 'standalone' or 'worker'", mode);
            std::process::exit(1);
        }
    }
    
    Ok(())
}

fn run_standalone(config: &Config, config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
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
        
        // Process batch in parallel
        let result: Option<WorkResult> = (current_offset..batch_end)
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
