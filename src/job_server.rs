use crate::config::Config;
use crate::job_types::*;
use crate::slack::SlackNotifier;
use crate::word_space::WordSpace;
use serde_json;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Job server manages the distribution of work to multiple workers
pub struct JobServer {
    config: Config,
    jobs: Arc<Mutex<HashMap<JobId, Job>>>,
    word_space: WordSpace,
    job_counter: Arc<Mutex<u64>>,
    start_time: Instant,
    solution: Arc<Mutex<Option<SolutionResult>>>,
    slack_notifier: Option<SlackNotifier>,
    job_timeout_seconds: u64,
    job_size: u128,
}

impl JobServer {
    /// Create a new job server
    pub fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        let word_space = WordSpace::from_config(&config);

        // Initialize Slack notifier if configured
        let slack_notifier = config
            .slack
            .as_ref()
            .map(|slack_config| SlackNotifier::new(slack_config.clone()));

        let server = Self {
            config,
            jobs: Arc::new(Mutex::new(HashMap::new())),
            word_space,
            job_counter: Arc::new(Mutex::new(0)),
            start_time: Instant::now(),
            solution: Arc::new(Mutex::new(None)),
            slack_notifier,
            job_timeout_seconds: 300, // 5 minutes default timeout
            job_size: 1000000,        // Default job size
        };

        Ok(server)
    }

    /// Initialize jobs by dividing the search space
    pub fn initialize_jobs(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut jobs = self.jobs.lock().unwrap();
        let mut job_counter = self.job_counter.lock().unwrap();

        println!(
            "Initializing jobs for search space of {} combinations",
            self.word_space.total_combinations
        );

        let mut current_offset = 0u128;
        let total_jobs_to_create = std::cmp::min(
            10000,
            (self.word_space.total_combinations / self.job_size) + 1,
        );
        let mut jobs_created = 0;

        while current_offset < self.word_space.total_combinations
            && jobs_created < total_jobs_to_create
        {
            let end_offset = std::cmp::min(
                current_offset + self.job_size,
                self.word_space.total_combinations,
            );

            *job_counter += 1;
            let job_id = format!("job-{}", *job_counter);
            let job = Job::new(job_id.clone(), current_offset, end_offset);

            jobs.insert(job_id, job);
            current_offset = end_offset;
            jobs_created += 1;
        }

        println!(
            "Initialized {} jobs (limited to {} for testing)",
            jobs.len(),
            total_jobs_to_create
        );

        // Notify search started
        if let Some(notifier) = &self.slack_notifier {
            notifier.notify_search_started(
                &self.config.ethereum.target_address,
                self.word_space.total_combinations,
            )?;
        }

        Ok(())
    }

    /// Get next available job for a worker
    pub fn assign_job(&self, request: &JobRequest) -> Result<JobResponse, ApiError> {
        // Check if solution already found
        if self.solution.lock().unwrap().is_some() {
            return Ok(JobResponse {
                job: None,
                search_config: self.get_search_config(),
            });
        }

        let mut jobs = self.jobs.lock().unwrap();

        // Find first pending job
        let job_to_assign = jobs
            .values_mut()
            .find(|job| job.status == JobStatus::Pending);

        if let Some(job) = job_to_assign {
            job.assign_to_worker(request.worker_id.clone());

            println!(
                "Assigned job {} to worker {} (range: {} to {})",
                job.id, request.worker_id, job.start_offset, job.end_offset
            );

            Ok(JobResponse {
                job: Some(job.clone()),
                search_config: self.get_search_config(),
            })
        } else {
            // No jobs available
            Ok(JobResponse {
                job: None,
                search_config: self.get_search_config(),
            })
        }
    }

    /// Handle job completion from worker
    pub fn complete_job(&self, completion: &JobCompletion) -> Result<(), ApiError> {
        let mut jobs = self.jobs.lock().unwrap();

        let job = jobs.get_mut(&completion.job_id).ok_or_else(|| ApiError {
            error: "Job not found".to_string(),
            code: 404,
        })?;

        if completion.success {
            job.mark_completed();
            println!(
                "Job {} completed by worker {} ({} mnemonics checked)",
                completion.job_id, completion.worker_id, completion.mnemonics_checked
            );

            // Check if solution was found
            if let Some(result) = &completion.result {
                let mut solution = self.solution.lock().unwrap();
                *solution = Some(result.clone());

                println!("🎉 Solution found! Mnemonic: {}", result.mnemonic);

                // Notify via Slack
                if let Some(notifier) = &self.slack_notifier {
                    if let Err(e) = notifier.notify_solution_found(
                        &result.mnemonic,
                        &result.address,
                        result.offset,
                    ) {
                        eprintln!("Failed to send Slack notification: {}", e);
                    }
                }
            }
        } else {
            let error = completion
                .error
                .clone()
                .unwrap_or_else(|| "Unknown error".to_string());
            job.mark_failed(error);
            println!(
                "Job {} failed on worker {}: {}",
                completion.job_id,
                completion.worker_id,
                completion
                    .error
                    .as_ref()
                    .unwrap_or(&"Unknown error".to_string())
            );
        }

        Ok(())
    }

    /// Handle worker heartbeat
    pub fn update_heartbeat(&self, heartbeat: &WorkerHeartbeat) -> Result<(), ApiError> {
        let mut jobs = self.jobs.lock().unwrap();

        let job = jobs.get_mut(&heartbeat.job_id).ok_or_else(|| ApiError {
            error: "Job not found".to_string(),
            code: 404,
        })?;

        job.update_heartbeat();
        Ok(())
    }

    /// Get server status and statistics
    pub fn get_status(&self) -> ServerStatus {
        let jobs = self.jobs.lock().unwrap();
        let solution = self.solution.lock().unwrap();

        let total_jobs = jobs.len();
        let pending_jobs = jobs
            .values()
            .filter(|job| job.status == JobStatus::Pending)
            .count();
        let assigned_jobs = jobs
            .values()
            .filter(|job| matches!(job.status, JobStatus::Assigned { .. }))
            .count();
        let completed_jobs = jobs
            .values()
            .filter(|job| job.status == JobStatus::Completed)
            .count();
        let failed_jobs = jobs
            .values()
            .filter(|job| matches!(job.status, JobStatus::Failed { .. }))
            .count();

        // Calculate combinations searched
        let combinations_searched: u128 = jobs
            .values()
            .filter(|job| job.status == JobStatus::Completed)
            .map(|job| job.size())
            .sum();

        // Get unique active workers
        let active_workers: std::collections::HashSet<String> = jobs
            .values()
            .filter_map(|job| {
                if let JobStatus::Assigned { worker_id } = &job.status {
                    Some(worker_id.clone())
                } else {
                    None
                }
            })
            .collect();

        ServerStatus {
            total_jobs,
            pending_jobs,
            assigned_jobs,
            completed_jobs,
            failed_jobs,
            active_workers: active_workers.len(),
            total_combinations: self.word_space.total_combinations,
            combinations_searched,
            search_rate: 0.0, // Would need to track this from heartbeats
            uptime_seconds: self.start_time.elapsed().as_secs(),
            solution_found: solution.clone(),
        }
    }

    /// Start background task to handle job timeouts
    pub fn start_timeout_handler(&self) {
        let jobs = Arc::clone(&self.jobs);
        let timeout_seconds = self.job_timeout_seconds;

        thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_secs(60)); // Check every minute

                let mut jobs = jobs.lock().unwrap();
                let timed_out_jobs: Vec<JobId> = jobs
                    .values()
                    .filter(|job| {
                        matches!(job.status, JobStatus::Assigned { .. })
                            && job.is_timed_out(timeout_seconds)
                    })
                    .map(|job| job.id.clone())
                    .collect();

                for job_id in timed_out_jobs {
                    if let Some(job) = jobs.get_mut(&job_id) {
                        println!("Job {} timed out, marking as pending", job_id);
                        job.status = JobStatus::Pending;
                        job.assigned_at = None;
                        job.worker_heartbeat = None;
                    }
                }
            }
        });
    }

    /// Get search configuration for workers
    fn get_search_config(&self) -> SearchConfig {
        SearchConfig {
            target_address: self.config.ethereum.target_address.clone(),
            derivation_path: self.config.ethereum.derivation_path.clone(),
            passphrase: self.config.passphrase.clone(),
            word_constraints_serialized: serde_json::to_string(&self.config.word_constraints)
                .unwrap_or_else(|_| "[]".to_string()),
        }
    }

    /// Check if search is complete (solution found or all jobs done)
    pub fn is_search_complete(&self) -> bool {
        if self.solution.lock().unwrap().is_some() {
            return true;
        }

        let jobs = self.jobs.lock().unwrap();
        let pending_jobs = jobs
            .values()
            .filter(|job| job.status == JobStatus::Pending)
            .count();
        let assigned_jobs = jobs
            .values()
            .filter(|job| matches!(job.status, JobStatus::Assigned { .. }))
            .count();

        pending_jobs == 0 && assigned_jobs == 0
    }

    /// Get word space for job calculations
    pub fn get_word_space(&self) -> &WordSpace {
        &self.word_space
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, EthereumConfig, WordConstraint};

    fn create_test_config() -> Config {
        Config {
            word_constraints: vec![WordConstraint {
                position: 0,
                prefix: Some("abandon".to_string()),
                words: vec![],
            }],
            ethereum: EthereumConfig {
                derivation_path: "m/44'/60'/0'/0/0".to_string(),
                target_address: "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f".to_string(),
            },
            slack: None,
            worker: None,
            gpu: None,
            batch_size: 1000,
            passphrase: "".to_string(),
        }
    }

    #[test]
    fn test_job_server_creation() {
        let config = create_test_config();
        let server = JobServer::new(config).unwrap();
        assert!(server.jobs.lock().unwrap().is_empty());
    }

    #[test]
    fn test_job_initialization() {
        let config = create_test_config();
        let server = JobServer::new(config).unwrap();
        server.initialize_jobs().unwrap();

        let jobs = server.jobs.lock().unwrap();
        assert!(!jobs.is_empty());

        // All jobs should be pending initially
        assert!(jobs.values().all(|job| job.status == JobStatus::Pending));
    }

    #[test]
    fn test_job_assignment() {
        let config = create_test_config();
        let server = JobServer::new(config).unwrap();
        server.initialize_jobs().unwrap();

        let request = JobRequest {
            worker_id: "test-worker".to_string(),
            worker_capabilities: WorkerCapabilities {
                max_batch_size: 1000,
                estimated_rate: 1000.0,
            },
        };

        let response = server.assign_job(&request).unwrap();
        assert!(response.job.is_some());

        let job = response.job.unwrap();
        match job.status {
            JobStatus::Assigned { worker_id } => assert_eq!(worker_id, "test-worker"),
            _ => panic!("Job should be assigned"),
        }
    }
}
