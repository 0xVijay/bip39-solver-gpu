use rayon::prelude::*;
use std::env;
use std::time::Instant;
use std::io::Write;

pub mod config;
pub mod cuda_backend;
pub mod error_handling;
pub mod eth;
pub mod gpu_backend;
pub mod gpu_manager;
pub mod job_server;
pub mod job_types;
pub mod opencl_backend;
pub mod slack;
pub mod stress_testing;
#[cfg(test)]
mod tests;
pub mod word_space;
pub mod worker_client;

use config::{Config, GpuConfig};
use eth::{addresses_equal, derive_ethereum_address};
use gpu_manager::GpuManager;
use slack::SlackNotifier;
use stress_testing::StressTester;
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
        eprintln!("  --stress-test                   Run comprehensive stress tests");
        eprintln!("\nRecommended config file: example_test_config.json");
        eprintln!("Example usage: {} --config example_test_config.json", args[0]);
        eprintln!("\nExample config:");
        let default_config = Config::default();
        println!("{}", serde_json::to_string_pretty(&default_config)?);
        std::process::exit(1);
    }

    let config_path = &args[2];
    
    // Validate config file path and recommend example_test_config.json
    if !std::path::Path::new(config_path).exists() {
        eprintln!("Error: Config file '{}' not found", config_path);
        eprintln!("Tip: Use 'example_test_config.json' for testing");
        std::process::exit(1);
    }
    
    if !config_path.ends_with("example_test_config.json") {
        println!("Warning: Using config file '{}'. For testing, consider using 'example_test_config.json'", config_path);
    }
    
    let mut config = Config::load(config_path)?;

    // Parse command line arguments to override config
    let mut i = 3;
    let mut mode = "standalone".to_string();
    let mut gpu_backend: Option<String> = None;
    let mut gpu_devices: Vec<u32> = Vec::new();
    let mut multi_gpu: Option<bool> = None;
    let mut run_stress_test = false;

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
            "--stress-test" => {
                run_stress_test = true;
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

    // Handle stress testing mode
    if run_stress_test {
        return run_stress_tests(&config);
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

    // Initialize GPU manager with enhanced error handling
    let mut gpu_manager = match GpuManager::from_config(config) {
        Ok(manager) => {
            println!("✅ GPU backend initialized: {}", manager.backend_name());
            if manager.is_multi_gpu_enabled() {
                println!(
                    "🔥 Multi-GPU processing enabled with {} device(s)",
                    manager.devices().len()
                );
                for device in manager.devices() {
                    println!("   • Device {}: {} ({} MB memory, {} compute units)", 
                        device.id, device.name, device.memory / 1024 / 1024, device.compute_units);
                }
            } else {
                println!("🖥️  Single GPU processing with {} device(s)", manager.devices().len());
                for device in manager.devices() {
                    println!("   • Device {}: {} ({} MB memory, {} compute units)", 
                        device.id, device.name, device.memory / 1024 / 1024, device.compute_units);
                }
            }
            Some(manager)
        }
        Err(e) => {
            println!("⚠️  Failed to initialize GPU backend: {}", e);
            println!("🔄 Falling back to CPU processing");
            println!("   This may be significantly slower than GPU processing.");
            
            // Provide helpful troubleshooting tips
            println!("\n💡 GPU Troubleshooting Tips:");
            if config.gpu.as_ref().map(|g| &g.backend) == Some(&"cuda".to_string()) {
                println!("   • For CUDA: Ensure NVIDIA drivers and CUDA toolkit are installed");
                println!("   • Try: nvidia-smi to check GPU status");
                println!("   • Try: nvcc --version to check CUDA installation");
            } else {
                println!("   • For OpenCL: Install OpenCL drivers for your GPU");
                println!("   • Try: clinfo to check OpenCL installation");
            }
            println!("   • Try --gpu-backend cuda or --gpu-backend opencl to switch backends");
            println!("   • Build with features: cargo build --features cuda,opencl\n");
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

        // Use carriage return to overwrite previous line for compact output
        print!("\rSearching: {} to {} ", current_offset, batch_end);
        std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());

        let result = if let Some(ref manager) = gpu_manager {
            // GPU processing
            search_with_gpu(manager, current_offset, batch_end - current_offset, config)?
        } else {
            // CPU fallback processing
            search_with_cpu(&word_space, current_offset, batch_end, config)?
        };

        if let Some(work_result) = result {
            println!("\n🎉 Found matching mnemonic!");
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

        // Report progress with compact format
        let elapsed = start_time.elapsed();
        let rate = current_offset as f64 / elapsed.as_secs_f64();

        print!(
            "\rProgress: {}/{} ({:.2}%) | Rate: {:.2} mnemonics/sec | Elapsed: {:.1}s",
            current_offset,
            word_space.total_combinations,
            (current_offset as f64 / word_space.total_combinations as f64) * 100.0,
            rate,
            elapsed.as_secs_f64()
        );
        std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());

        // Periodically notify progress via Slack (every 10 minutes)
        if elapsed.as_secs() % 600 == 0 && elapsed.as_secs() > 0 {
            if let Some(notifier) = &slack_notifier {
                notifier.notify_progress(current_offset, rate, elapsed.as_secs())?;
            }
        }

        // Check if we've searched everything
        if current_offset >= word_space.total_combinations {
            println!("\nSearch completed. No matching mnemonic found.");
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

/// Run comprehensive stress tests
fn run_stress_tests(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Advanced Error Handling & Stress Testing for CUDA Backend");
    println!("=====================================================================\n");

    let tester = StressTester::new();
    
    // Run all stress tests
    let results = tester.run_all_tests(config)?;
    
    // Generate and display report
    let report = tester.generate_report();
    println!("{}", report);
    
    // Summary statistics
    let total_tests = results.len();
    let passed_tests = results.iter().filter(|r| r.success).count();
    let failed_tests = total_tests - passed_tests;
    
    if failed_tests > 0 {
        println!("⚠️  Some stress tests failed. Review the results above for details.");
        println!("   Note: Some failures may be expected in environments without GPU hardware.");
    } else {
        println!("✅ All stress tests passed successfully!");
    }
    
    println!("\n🎯 Error Handling Features Demonstrated:");
    println!("   • Comprehensive CUDA error checking for all FFI calls");
    println!("   • Device health monitoring and failure detection");
    println!("   • Automatic failover from failed devices to healthy ones");
    println!("   • Graceful CPU fallback when all GPUs fail");
    println!("   • Structured logging with device info and timestamps");
    println!("   • Edge-case and stress testing for huge batch sizes");
    println!("   • Out-of-memory scenario handling");
    println!("   • Distributed network stress testing (simulated)");
    
    Ok(())
}
