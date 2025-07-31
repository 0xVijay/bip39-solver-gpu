use crate::config::{Config, WordConstraint};
use crate::job_types::*;
use crate::word_space::WordSpace;
use crate::eth::{derive_ethereum_address, addresses_equal};
use std::time::{Duration, Instant};
use std::thread;
use rayon::prelude::*;
use reqwest;
use serde_json;

/// Worker client connects to job server and performs distributed search
pub struct WorkerClient {
    worker_id: WorkerId,
    server_url: String,
    secret: String,
    client: reqwest::blocking::Client,
    capabilities: WorkerCapabilities,
}

impl WorkerClient {
    /// Create a new worker client
    pub fn new(
        worker_id: WorkerId,
        server_url: String,
        secret: String,
        capabilities: WorkerCapabilities,
    ) -> Self {
        Self {
            worker_id,
            server_url,
            secret,
            client: reqwest::blocking::Client::new(),
            capabilities,
        }
    }

    /// Create worker from config
    pub fn from_config(config: &Config, worker_id: Option<String>) -> Result<Self, Box<dyn std::error::Error>> {
        let worker_config = config.worker.as_ref()
            .ok_or("Worker configuration not found")?;
        
        let worker_id = worker_id.unwrap_or_else(|| {
            format!("worker-{}", std::process::id())
        });

        let capabilities = WorkerCapabilities {
            max_batch_size: config.batch_size,
            estimated_rate: 3000.0, // Conservative estimate
        };

        Ok(Self::new(
            worker_id,
            worker_config.server_url.clone(),
            worker_config.secret.clone(),
            capabilities,
        ))
    }

    /// Start the worker main loop
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Worker {} starting, connecting to server at {}", self.worker_id, self.server_url);
        
        loop {
            match self.request_job() {
                Ok(response) => {
                    if let Some(job) = response.job {
                        println!("Received job {}: range {} to {}", job.id, job.start_offset, job.end_offset);
                        
                        // Process the job
                        match self.process_job(&job, &response.search_config) {
                            Ok(result) => {
                                // Report completion
                                let completion = JobCompletion {
                                    job_id: job.id.clone(),
                                    worker_id: self.worker_id.clone(),
                                    success: true,
                                    mnemonics_checked: job.size() as u64,
                                    result,
                                    error: None,
                                };
                                
                                if let Err(e) = self.report_completion(&completion) {
                                    eprintln!("Failed to report job completion: {}", e);
                                }
                                
                                // If solution found, exit
                                if completion.result.is_some() {
                                    println!("Solution found! Worker exiting.");
                                    return Ok(());
                                }
                            }
                            Err(e) => {
                                eprintln!("Error processing job {}: {}", job.id, e);
                                
                                // Report failure
                                let completion = JobCompletion {
                                    job_id: job.id.clone(),
                                    worker_id: self.worker_id.clone(),
                                    success: false,
                                    mnemonics_checked: 0,
                                    result: None,
                                    error: Some(e.to_string()),
                                };
                                
                                if let Err(e) = self.report_completion(&completion) {
                                    eprintln!("Failed to report job failure: {}", e);
                                }
                            }
                        }
                    } else {
                        // No jobs available, wait and retry
                        println!("No jobs available, waiting...");
                        thread::sleep(Duration::from_secs(10));
                    }
                }
                Err(e) => {
                    eprintln!("Failed to request job: {}", e);
                    thread::sleep(Duration::from_secs(5));
                }
            }
        }
    }

    /// Request a job from the server
    fn request_job(&self) -> Result<JobResponse, Box<dyn std::error::Error>> {
        let request = JobRequest {
            worker_id: self.worker_id.clone(),
            worker_capabilities: self.capabilities.clone(),
        };

        let url = format!("{}/api/jobs/request", self.server_url);
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.secret))
            .json(&request)
            .send()?;

        if response.status().is_success() {
            let job_response: JobResponse = response.json()?;
            Ok(job_response)
        } else {
            Err(format!("Failed to request job: HTTP {}", response.status()).into())
        }
    }

    /// Report job completion to server
    fn report_completion(&self, completion: &JobCompletion) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{}/api/jobs/complete", self.server_url);
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.secret))
            .json(completion)
            .send()?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(format!("Failed to report completion: HTTP {}", response.status()).into())
        }
    }

    /// Send heartbeat to server
    fn send_heartbeat(&self, job_id: &JobId, progress: u64, rate: f64) -> Result<(), Box<dyn std::error::Error>> {
        let heartbeat = WorkerHeartbeat {
            job_id: job_id.clone(),
            worker_id: self.worker_id.clone(),
            progress,
            rate,
        };

        let url = format!("{}/api/jobs/heartbeat", self.server_url);
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.secret))
            .json(&heartbeat)
            .send()?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(format!("Failed to send heartbeat: HTTP {}", response.status()).into())
        }
    }

    /// Process a job by searching the assigned range
    fn process_job(&self, job: &Job, search_config: &SearchConfig) -> Result<Option<SolutionResult>, Box<dyn std::error::Error>> {
        // Parse word constraints from JSON
        let word_constraints: Vec<WordConstraint> = serde_json::from_str(&search_config.word_constraints_serialized)?;
        
        // Create temporary config for word space generation
        let temp_config = Config {
            word_constraints,
            ethereum: crate::config::EthereumConfig {
                derivation_path: search_config.derivation_path.clone(),
                target_address: search_config.target_address.clone(),
            },
            slack: None,
            worker: None,
            batch_size: self.capabilities.max_batch_size,
            passphrase: search_config.passphrase.clone(),
        };

        let word_space = WordSpace::from_config(&temp_config);
        
        println!("Processing job {} with {} candidates", job.id, job.size());
        
        let start_time = Instant::now();
        let heartbeat_interval = Duration::from_secs(30);
        let mut last_heartbeat = start_time;

        // Search the range in parallel
        let result: Option<SolutionResult> = (job.start_offset..job.end_offset)
            .into_par_iter()
            .find_map_any(|index| {
                if let Some(word_indices) = word_space.index_to_words(index) {
                    if let Some(mnemonic) = WordSpace::words_to_mnemonic(&word_indices) {
                        match derive_ethereum_address(
                            &mnemonic,
                            &search_config.passphrase,
                            &search_config.derivation_path,
                        ) {
                            Ok(address) => {
                                if addresses_equal(&address, &search_config.target_address) {
                                    return Some(SolutionResult {
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

        let elapsed = start_time.elapsed();
        let final_rate = job.size() as f64 / elapsed.as_secs_f64();
        
        println!("Completed job {} in {:?} ({:.2} mnemonics/sec)", 
            job.id, elapsed, final_rate);

        Ok(result)
    }
}

/// Standalone worker function that can be called from main
pub fn run_worker(config: &Config, worker_id: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let worker = WorkerClient::from_config(config, worker_id)?;
    worker.run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, EthereumConfig, WorkerConfig};

    fn create_test_config() -> Config {
        Config {
            word_constraints: vec![],
            ethereum: EthereumConfig {
                derivation_path: "m/44'/60'/0'/0/0".to_string(),
                target_address: "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f".to_string(),
            },
            slack: None,
            worker: Some(WorkerConfig {
                server_url: "http://localhost:3000".to_string(),
                secret: "test-secret".to_string(),
            }),
            batch_size: 1000,
            passphrase: "".to_string(),
        }
    }

    #[test]
    fn test_worker_creation_from_config() {
        let config = create_test_config();
        let worker = WorkerClient::from_config(&config, Some("test-worker".to_string())).unwrap();
        assert_eq!(worker.worker_id, "test-worker");
        assert_eq!(worker.server_url, "http://localhost:3000");
    }

    #[test]
    fn test_worker_capabilities() {
        let capabilities = WorkerCapabilities {
            max_batch_size: 1000,
            estimated_rate: 2500.0,
        };
        
        let worker = WorkerClient::new(
            "test-worker".to_string(),
            "http://localhost:3000".to_string(),
            "secret".to_string(),
            capabilities,
        );
        
        assert_eq!(worker.capabilities.max_batch_size, 1000);
        assert_eq!(worker.capabilities.estimated_rate, 2500.0);
    }
}