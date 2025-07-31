use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Unique identifier for a job
pub type JobId = String;

/// Unique identifier for a worker
pub type WorkerId = String;

/// Job status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JobStatus {
    Pending,
    Assigned { worker_id: WorkerId },
    Completed,
    Failed { error: String },
}

/// A job represents a range of candidates to search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: JobId,
    pub start_offset: u128,
    pub end_offset: u128,
    pub status: JobStatus,
    pub assigned_at: Option<u64>,      // Unix timestamp
    pub completed_at: Option<u64>,     // Unix timestamp
    pub worker_heartbeat: Option<u64>, // Last heartbeat from worker
}

/// Request to assign a job to a worker
#[derive(Debug, Serialize, Deserialize)]
pub struct JobRequest {
    pub worker_id: WorkerId,
    pub worker_capabilities: WorkerCapabilities,
}

/// Worker capabilities and configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WorkerCapabilities {
    pub max_batch_size: u64,
    pub estimated_rate: f64, // mnemonics per second
}

/// Response containing an assigned job
#[derive(Debug, Serialize, Deserialize)]
pub struct JobResponse {
    pub job: Option<Job>,
    pub search_config: SearchConfig,
}

/// Search configuration shared with workers
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchConfig {
    pub target_address: String,
    pub derivation_path: String,
    pub passphrase: String,
    pub word_constraints_serialized: String, // JSON serialized word constraints
}

/// Worker reporting job completion
#[derive(Debug, Serialize, Deserialize)]
pub struct JobCompletion {
    pub job_id: JobId,
    pub worker_id: WorkerId,
    pub success: bool,
    pub mnemonics_checked: u64,
    pub result: Option<SolutionResult>,
    pub error: Option<String>,
}

/// Solution found during search
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SolutionResult {
    pub mnemonic: String,
    pub address: String,
    pub offset: u128,
}

/// Worker heartbeat to indicate it's still working
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerHeartbeat {
    pub job_id: JobId,
    pub worker_id: WorkerId,
    pub progress: u64, // Number of mnemonics processed so far
    pub rate: f64,     // Current processing rate
}

/// Server progress and statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct ServerStatus {
    pub total_jobs: usize,
    pub pending_jobs: usize,
    pub assigned_jobs: usize,
    pub completed_jobs: usize,
    pub failed_jobs: usize,
    pub active_workers: usize,
    pub total_combinations: u128,
    pub combinations_searched: u128,
    pub search_rate: f64, // Combined rate from all workers
    pub uptime_seconds: u64,
    pub solution_found: Option<SolutionResult>,
}

/// Error types for job operations
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiError {
    pub error: String,
    pub code: u16,
}

impl Job {
    /// Create a new pending job
    pub fn new(id: JobId, start_offset: u128, end_offset: u128) -> Self {
        Self {
            id,
            start_offset,
            end_offset,
            status: JobStatus::Pending,
            assigned_at: None,
            completed_at: None,
            worker_heartbeat: None,
        }
    }

    /// Assign job to a worker
    pub fn assign_to_worker(&mut self, worker_id: WorkerId) {
        self.status = JobStatus::Assigned { worker_id };
        self.assigned_at = Some(current_timestamp());
        self.worker_heartbeat = Some(current_timestamp());
    }

    /// Mark job as completed
    pub fn mark_completed(&mut self) {
        self.status = JobStatus::Completed;
        self.completed_at = Some(current_timestamp());
    }

    /// Mark job as failed
    pub fn mark_failed(&mut self, error: String) {
        self.status = JobStatus::Failed { error };
        self.completed_at = Some(current_timestamp());
    }

    /// Update worker heartbeat
    pub fn update_heartbeat(&mut self) {
        self.worker_heartbeat = Some(current_timestamp());
    }

    /// Check if job has timed out (no heartbeat for too long)
    pub fn is_timed_out(&self, timeout_seconds: u64) -> bool {
        if let Some(heartbeat) = self.worker_heartbeat {
            let now = current_timestamp();
            now.saturating_sub(heartbeat) > timeout_seconds
        } else {
            false
        }
    }

    /// Get the size of the job (number of candidates to check)
    pub fn size(&self) -> u128 {
        self.end_offset - self.start_offset
    }
}

/// Get current Unix timestamp
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_creation() {
        let job = Job::new("test-1".to_string(), 0, 1000);
        assert_eq!(job.id, "test-1");
        assert_eq!(job.start_offset, 0);
        assert_eq!(job.end_offset, 1000);
        assert_eq!(job.size(), 1000);
        assert_eq!(job.status, JobStatus::Pending);
    }

    #[test]
    fn test_job_assignment() {
        let mut job = Job::new("test-1".to_string(), 0, 1000);
        job.assign_to_worker("worker-1".to_string());

        match job.status {
            JobStatus::Assigned { worker_id } => assert_eq!(worker_id, "worker-1"),
            _ => panic!("Job should be assigned"),
        }
        assert!(job.assigned_at.is_some());
    }

    #[test]
    fn test_job_timeout() {
        let mut job = Job::new("test-1".to_string(), 0, 1000);
        job.assign_to_worker("worker-1".to_string());

        // Should not be timed out immediately
        assert!(!job.is_timed_out(60));

        // Simulate old heartbeat
        job.worker_heartbeat = Some(current_timestamp() - 120);
        assert!(job.is_timed_out(60));
    }
}
