use std::env;

// Import from the local crate
use bip39_solver_gpu::config::Config;
use bip39_solver_gpu::worker_client::run_worker;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args[1] != "--config" {
        eprintln!("Usage: {} --config <config.json> [--worker-id <id>]", args[0]);
        std::process::exit(1);
    }

    let config_path = &args[2];
    let config = Config::load(config_path)?;
    
    // Parse optional worker ID
    let worker_id = if args.len() >= 5 && args[3] == "--worker-id" {
        Some(args[4].clone())
    } else {
        None
    };
    
    println!("Starting worker with config from: {}", config_path);
    if let Some(ref id) = worker_id {
        println!("Worker ID: {}", id);
    }
    
    // Verify worker configuration exists
    if config.worker.is_none() {
        eprintln!("Error: Worker configuration not found in config file");
        eprintln!("Please add a 'worker' section with 'server_url' and 'secret'");
        std::process::exit(1);
    }
    
    let worker_config = config.worker.as_ref().unwrap();
    println!("Connecting to server: {}", worker_config.server_url);
    
    // Run the worker
    run_worker(&config, worker_id)?;
    
    Ok(())
}