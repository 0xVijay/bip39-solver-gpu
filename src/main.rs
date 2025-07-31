use rayon::prelude::*;
use std::env;
use std::time::Instant;

pub mod config;
pub mod cuda_backend;
pub mod eth;
pub mod gpu_backend;
pub mod gpu_manager;
pub mod job_server;
pub mod job_types;
pub mod opencl_backend;
pub mod slack;
#[cfg(test)]
mod tests;
pub mod word_space;
pub mod worker_client;

use config::{Config, GpuConfig};
use eth::{addresses_equal, derive_ethereum_address};
use gpu_manager::GpuManager;
use slack::SlackNotifier;
use word_space::WordSpace;

#[derive(Debug)]
struct WorkResult {
    mnemonic: String,
    address: String,
    offset: u128,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args[1] != "--config" {
        eprintln!("Usage: {} --config <config.json> [options]", args[0]);
        eprintln!("\nOptions:");
        eprintln!("  --mode <standalone|worker>     Set processing mode (default: standalone)");
        eprintln!("  --gpu-backend <opencl|cuda>    Set GPU backend (overrides config)");
        eprintln!("  --gpu-device <device_id>       Use specific GPU device (can be repeated)");
        eprintln!("  --multi-gpu                     Enable multi-GPU processing");
        eprintln!("\nExample config:");
        let default_config = Config::default();
        println!("{}", serde_json::to_string_pretty(&default_config)?);
        std::process::exit(1);
    }

    let config_path = &args[2];
    let mut config = Config::load(config_path)?;

    // Parse command line arguments to override config
    let mut i = 3;
    let mut mode = "standalone".to_string();
    let mut gpu_backend: Option<String> = None;
    let mut gpu_devices: Vec<u32> = Vec::new();
    let mut multi_gpu: Option<bool> = None;

    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                if i + 1 < args.len() {
                    mode = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: --mode requires a value");
                    std::process::exit(1);
                }
            }
            "--gpu-backend" => {
                if i + 1 < args.len() {
                    gpu_backend = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --gpu-backend requires a value");
                    std::process::exit(1);
                }
            }
            "--gpu-device" => {
                if i + 1 < args.len() {
                    match args[i + 1].parse::<u32>() {
                        Ok(device_id) => {
                            gpu_devices.push(device_id);
                            i += 2;
                        }
                        Err(_) => {
                            eprintln!("Error: --gpu-device requires a numeric device ID");
                            std::process::exit(1);
                        }
                    }
                } else {
                    eprintln!("Error: --gpu-device requires a value");
                    std::process::exit(1);
                }
            }
            "--multi-gpu" => {
                multi_gpu = Some(true);
                i += 1;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    // Override GPU config from command line arguments
    if gpu_backend.is_some() || !gpu_devices.is_empty() || multi_gpu.is_some() {
        let mut gpu_config = config.gpu.unwrap_or_else(|| GpuConfig {
            backend: "opencl".to_string(),
            devices: vec![],
            multi_gpu: false,
        });

        if let Some(backend) = gpu_backend {
            gpu_config.backend = backend;
        }

        if !gpu_devices.is_empty() {
            gpu_config.devices = gpu_devices;
        }

        if let Some(multi) = multi_gpu {
            gpu_config.multi_gpu = multi;
        }

        config.gpu = Some(gpu_config);
    }

    match mode.as_str() {
        "worker" => {
            // Run as distributed worker
            worker_client::run_worker(&config, None)?;
        }
        "standalone" => {
            // Run standalone version with GPU support
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

    // Initialize GPU manager
    let mut gpu_manager = match GpuManager::from_config(config) {
        Ok(manager) => {
            println!("GPU backend initialized: {}", manager.backend_name());
            if manager.is_multi_gpu_enabled() {
                println!(
                    "Multi-GPU processing enabled with {} device(s)",
                    manager.devices().len()
                );
            }
            Some(manager)
        }
        Err(e) => {
            println!("Warning: Failed to initialize GPU backend: {}", e);
            println!("Falling back to CPU processing");
            None
        }
    };

    // Initialize Slack notifier if configured
    let slack_notifier = config
        .slack
        .as_ref()
        .map(|slack_config| SlackNotifier::new(slack_config.clone()));

    // Generate word space based on constraints
    let word_space = WordSpace::from_config(&config);
    println!(
        "Total combinations to search: {}",
        word_space.total_combinations
    );

    // Notify search start
    if let Some(notifier) = &slack_notifier {
        notifier.notify_search_started(
            &config.ethereum.target_address,
            word_space.total_combinations,
        )?;
    }

    let start_time = Instant::now();
    let batch_size = config.batch_size as u128;
    let mut current_offset = 0u128;

    // Main search loop
    loop {
        let batch_end = std::cmp::min(current_offset + batch_size, word_space.total_combinations);

        println!("Searching batch: {} to {}", current_offset, batch_end);

        let result = if let Some(ref manager) = gpu_manager {
            // GPU processing
            search_with_gpu(manager, current_offset, batch_end - current_offset, config)?
        } else {
            // CPU fallback processing
            search_with_cpu(&word_space, current_offset, batch_end, config)?
        };

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

            // Shutdown GPU manager
            if let Some(ref mut manager) = gpu_manager {
                manager.shutdown()?;
            }

            return Ok(());
        }

        current_offset = batch_end;

        // Report progress
        let elapsed = start_time.elapsed();
        let rate = current_offset as f64 / elapsed.as_secs_f64();

        println!(
            "Progress: {}/{} ({:.2}%) - Rate: {:.2} mnemonics/sec - Elapsed: {:?}",
            current_offset,
            word_space.total_combinations,
            (current_offset as f64 / word_space.total_combinations as f64) * 100.0,
            rate,
            elapsed
        );

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

    // Shutdown GPU manager
    if let Some(ref mut gpu_manager) = gpu_manager {
        gpu_manager.shutdown()?;
    }

    Ok(())
}

/// Search using GPU backends
fn search_with_gpu(
    gpu_manager: &GpuManager,
    start_offset: u128,
    batch_size: u128,
    config: &Config,
) -> Result<Option<WorkResult>, Box<dyn std::error::Error>> {
    let results = gpu_manager.execute_multi_gpu_batch(
        start_offset,
        batch_size,
        &config.ethereum.target_address,
        &config.ethereum.derivation_path,
        &config.passphrase,
    )?;

    // Check results from all devices
    for result in results {
        if let (Some(mnemonic), Some(address), Some(offset)) =
            (result.mnemonic, result.address, result.offset)
        {
            return Ok(Some(WorkResult {
                mnemonic,
                address,
                offset,
            }));
        }
    }

    Ok(None)
}

/// Search using CPU fallback (original implementation)
fn search_with_cpu(
    word_space: &WordSpace,
    start_offset: u128,
    end_offset: u128,
    config: &Config,
) -> Result<Option<WorkResult>, Box<dyn std::error::Error>> {
    // Process batch in parallel using rayon
    let result: Option<WorkResult> =
        (start_offset..end_offset)
            .into_par_iter()
            .find_map_any(|index| {
                if let Some(word_indices) = word_space.index_to_words(index) {
                    if let Some(mnemonic) = WordSpace::words_to_mnemonic(&word_indices) {
                        match derive_ethereum_address(
                            &mnemonic,
                            &config.passphrase,
                            &config.ethereum.derivation_path,
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

    Ok(result)
}
