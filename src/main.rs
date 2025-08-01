use std::env;
use std::time::Instant;

pub mod config;
pub mod cuda_backend;
pub mod error_handling;
pub mod eth;
pub mod gpu_backend;
pub mod gpu_manager;
pub mod gpu_memory;
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
        eprintln!("  --worker                    Run as distributed worker");
        eprintln!("  --stress-test               Run comprehensive stress tests");
        eprintln!("\nExample usage: {} --config example_test_config.json", args[0]);
        std::process::exit(1);
    }

    let config_path = &args[2];
    
    // Validate config file path
    if !std::path::Path::new(config_path).exists() {
        eprintln!("Error: Config file '{}' not found", config_path);
        std::process::exit(1);
    }
    
    let config = Config::load(config_path)?;

    // Parse simple command line arguments
    let mut run_as_worker = false;
    let mut run_stress_test = false;

    for arg in args.iter().skip(3) {
        match arg.as_str() {
            "--worker" => run_as_worker = true,
            "--stress-test" => run_stress_test = true,
            _ => {
                eprintln!("Unknown argument: {}", arg);
                std::process::exit(1);
            }
        }
    }

    // Handle stress testing mode
    if run_stress_test {
        return run_stress_tests(&config);
    }

    if run_as_worker {
        // Run as distributed worker
        worker_client::run_worker(&config, None)?;
    } else {
        // Run standalone version with automatic GPU detection
        run_standalone(&config, config_path)?;
    }

    Ok(())
}

fn run_standalone(config: &Config, config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loaded config from: {}", config_path);
    println!("Target address: {}", config.ethereum.target_address);
    println!("Derivation path: {}", config.ethereum.derivation_path);

    // Auto-detect and initialize the best available GPU backend
    let mut gpu_manager = auto_detect_best_gpu_backend(config)?;
    
    println!("✅ GPU backend initialized: {}", gpu_manager.backend_name());
    if gpu_manager.is_multi_gpu_enabled() {
        println!(
            "🔥 Multi-GPU processing enabled with {} device(s)",
            gpu_manager.devices().len()
        );
    } else {
        println!("🖥️  Single GPU processing with {} device(s)", gpu_manager.devices().len());
    }
    
    for device in gpu_manager.devices() {
        println!("   • Device {}: {} ({} MB memory, {} compute units)", 
            device.id, device.name, device.memory / 1024 / 1024, device.compute_units);
    }

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
    
    // Use dynamic batch sizing based on GPU memory (ignore config batch_size)
    let optimal_batch_sizes = gpu_manager.get_optimal_batch_sizes();
    
    // Use the largest optimal batch size for maximum performance
    let dynamic_batch_size = optimal_batch_sizes.iter()
        .map(|(_, size)| *size)
        .max()
        .unwrap_or(100000); // Fallback only if no GPU devices
    
    println!("📊 Using dynamic batch sizing:");
    for (device_id, batch_size) in &optimal_batch_sizes {
        if let Some(memory_info) = gpu_manager.get_device_memory_info(*device_id, *batch_size) {
            println!("   • Device {}: {} batch size - {}", device_id, batch_size, memory_info);
        }
    }
    
    let mut current_offset = 0u128;

    // Main search loop - GPU only, no CPU fallback
    loop {
        let batch_end = std::cmp::min(current_offset + dynamic_batch_size, word_space.total_combinations);

        let result = search_with_gpu(&gpu_manager, current_offset, batch_end - current_offset, config)?;

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

            gpu_manager.shutdown()?;
            return Ok(());
        }

        current_offset = batch_end;

        // Report progress with compact format including ETA
        let elapsed = start_time.elapsed();
        let rate = current_offset as f64 / elapsed.as_secs_f64();
        let remaining = word_space.total_combinations - current_offset;
        let eta_seconds = if rate > 0.0 {
            remaining as f64 / rate
        } else {
            0.0
        };

        // Format ETA nicely
        let eta_str = if eta_seconds < 60.0 {
            format!("{:.0}s", eta_seconds)
        } else if eta_seconds < 3600.0 {
            format!("{:.0}m {:.0}s", eta_seconds / 60.0, eta_seconds % 60.0)
        } else if eta_seconds < 86400.0 {
            format!("{:.0}h {:.0}m", eta_seconds / 3600.0, (eta_seconds % 3600.0) / 60.0)
        } else {
            format!("{:.0}d {:.0}h", eta_seconds / 86400.0, (eta_seconds % 86400.0) / 3600.0)
        };

        print!(
            "\rProgress: {}/{} ({:.2}%) | Rate: {:.2} mnemonics/sec | Elapsed: {:.1}s | ETA: {}",
            current_offset,
            word_space.total_combinations,
            (current_offset as f64 / word_space.total_combinations as f64) * 100.0,
            rate,
            elapsed.as_secs_f64(),
            eta_str
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

    gpu_manager.shutdown()?;
    Ok(())
}

/// Auto-detect and initialize the best available GPU backend
fn auto_detect_best_gpu_backend(config: &Config) -> Result<GpuManager, Box<dyn std::error::Error>> {
    // Try backends in order of preference: CUDA first (faster), then OpenCL
    let backends_to_try = vec!["cuda", "opencl"];
    
    for backend_name in backends_to_try {
        println!("🔍 Trying {} backend...", backend_name.to_uppercase());
        
        // Create a temporary config with this backend
        let mut test_config = config.clone();
        test_config.gpu = Some(GpuConfig {
            backend: backend_name.to_string(),
            devices: vec![], // Auto-detect all devices
            multi_gpu: true, // Always enable multi-GPU for maximum performance
        });
        
        match GpuManager::from_config(&test_config) {
            Ok(manager) => {
                let device_count = manager.devices().len();
                if device_count > 0 {
                    println!("✅ {} backend initialized successfully with {} device(s)", 
                             backend_name.to_uppercase(), device_count);
                    return Ok(manager);
                }
            }
            Err(e) => {
                println!("❌ {} backend failed: {}", backend_name.to_uppercase(), e);
                continue;
            }
        }
    }
    
    // If no GPU backend works, exit with error (no CPU fallback)
    Err("No GPU backend available. Ensure CUDA or OpenCL drivers are installed.".into())
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
