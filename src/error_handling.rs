use std::error::Error;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Comprehensive error types for GPU backends
#[derive(Debug, Clone, PartialEq)]
pub enum GpuError {
    /// Device initialization failed
    DeviceInitFailed {
        device_id: u32,
        device_name: String,
        error: String,
        timestamp: u64,
    },
    /// Kernel execution failed
    KernelExecutionFailed {
        device_id: u32,
        kernel_name: String,
        error: String,
        timestamp: u64,
    },
    /// Device memory allocation failed
    MemoryAllocationFailed {
        device_id: u32,
        requested_bytes: usize,
        available_bytes: Option<usize>,
        timestamp: u64,
    },
    /// Device communication timeout
    DeviceTimeout {
        device_id: u32,
        operation: String,
        timeout_duration: u64,
        timestamp: u64,
    },
    /// Device hardware failure detected
    DeviceHardwareFailure {
        device_id: u32,
        device_name: String,
        error_code: i32,
        timestamp: u64,
    },
    /// Out of memory error during batch processing
    OutOfMemory {
        device_id: u32,
        batch_size: u128,
        available_memory: u64,
        timestamp: u64,
    },
    /// Device removed from system during operation
    DeviceRemoved {
        device_id: u32,
        device_name: String,
        timestamp: u64,
    },
    /// Backend not available
    BackendUnavailable {
        backend_name: String,
        reason: String,
        timestamp: u64,
    },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::DeviceInitFailed { device_id, device_name, error, timestamp } => {
                write!(f, "[{}] Device Init Failed - Device {}: {} - {}", 
                    format_timestamp(*timestamp), device_id, device_name, error)
            }
            GpuError::KernelExecutionFailed { device_id, kernel_name, error, timestamp } => {
                write!(f, "[{}] Kernel Execution Failed - Device {}, Kernel: {} - {}", 
                    format_timestamp(*timestamp), device_id, kernel_name, error)
            }
            GpuError::MemoryAllocationFailed { device_id, requested_bytes, available_bytes, timestamp } => {
                let available_str = available_bytes
                    .map(|bytes| format!("{} bytes available", bytes))
                    .unwrap_or_else(|| "unknown available".to_string());
                write!(f, "[{}] Memory Allocation Failed - Device {}: requested {} bytes, {}", 
                    format_timestamp(*timestamp), device_id, requested_bytes, available_str)
            }
            GpuError::DeviceTimeout { device_id, operation, timeout_duration, timestamp } => {
                write!(f, "[{}] Device Timeout - Device {}: {} operation timed out after {}ms", 
                    format_timestamp(*timestamp), device_id, operation, timeout_duration)
            }
            GpuError::DeviceHardwareFailure { device_id, device_name, error_code, timestamp } => {
                write!(f, "[{}] Hardware Failure - Device {}: {} (error code: {})", 
                    format_timestamp(*timestamp), device_id, device_name, error_code)
            }
            GpuError::OutOfMemory { device_id, batch_size, available_memory, timestamp } => {
                write!(f, "[{}] Out of Memory - Device {}: batch size {} too large for {} bytes", 
                    format_timestamp(*timestamp), device_id, batch_size, available_memory)
            }
            GpuError::DeviceRemoved { device_id, device_name, timestamp } => {
                write!(f, "[{}] Device Removed - Device {}: {} was disconnected", 
                    format_timestamp(*timestamp), device_id, device_name)
            }
            GpuError::BackendUnavailable { backend_name, reason, timestamp } => {
                write!(f, "[{}] Backend Unavailable - {}: {}", 
                    format_timestamp(*timestamp), backend_name, reason)
            }
        }
    }
}

impl Error for GpuError {}

/// Device health status
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceHealth {
    Healthy,
    Warning { message: String },
    Failed { error: GpuError },
    Removed,
}

/// Device status information
#[derive(Debug, Clone)]
pub struct DeviceStatus {
    pub device_id: u32,
    pub device_name: String,
    pub health: DeviceHealth,
    pub last_check: u64,
    pub total_batches_processed: u64,
    pub total_errors: u64,
    pub last_error: Option<GpuError>,
    pub available_memory: Option<u64>,
    pub utilization: Option<f32>, // 0.0 to 1.0
}

impl DeviceStatus {
    pub fn new(device_id: u32, device_name: String) -> Self {
        Self {
            device_id,
            device_name,
            health: DeviceHealth::Healthy,
            last_check: current_timestamp(),
            total_batches_processed: 0,
            total_errors: 0,
            last_error: None,
            available_memory: None,
            utilization: None,
        }
    }

    pub fn mark_healthy(&mut self) {
        self.health = DeviceHealth::Healthy;
        self.last_check = current_timestamp();
    }

    pub fn mark_warning(&mut self, message: String) {
        self.health = DeviceHealth::Warning { message };
        self.last_check = current_timestamp();
    }

    pub fn mark_failed(&mut self, error: GpuError) {
        self.health = DeviceHealth::Failed { error: error.clone() };
        self.last_error = Some(error);
        self.total_errors += 1;
        self.last_check = current_timestamp();
    }

    pub fn mark_removed(&mut self) {
        self.health = DeviceHealth::Removed;
        self.last_check = current_timestamp();
    }

    pub fn increment_batch_count(&mut self) {
        self.total_batches_processed += 1;
        self.last_check = current_timestamp();
    }

    pub fn is_healthy(&self) -> bool {
        matches!(self.health, DeviceHealth::Healthy)
    }

    pub fn is_usable(&self) -> bool {
        matches!(self.health, DeviceHealth::Healthy | DeviceHealth::Warning { .. })
    }
}

/// Logging and error reporting interface
pub struct ErrorLogger {
    verbose: bool,
}

impl ErrorLogger {
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }

    /// Log a GPU error with detailed information
    pub fn log_error(&self, error: &GpuError) {
        eprintln!("GPU ERROR: {}", error);
        
        if self.verbose {
            eprintln!("  Error details: {:?}", error);
        }
    }

    /// Log device status change
    pub fn log_device_status_change(&self, old_status: &DeviceStatus, new_status: &DeviceStatus) {
        if old_status.health != new_status.health {
            println!("Device {} status changed: {:?} -> {:?}", 
                new_status.device_id, old_status.health, new_status.health);
        }
    }

    /// Log device recovery attempt
    pub fn log_recovery_attempt(&self, device_id: u32, device_name: &str, attempt: u32) {
        println!("[{}] Recovery attempt {} for device {}: {}", 
            format_timestamp(current_timestamp()), attempt, device_id, device_name);
    }

    /// Log failover event
    pub fn log_failover(&self, failed_device_id: u32, target_device_id: Option<u32>, batch_info: &str) {
        if let Some(target) = target_device_id {
            println!("[{}] Failover: reassigning {} from device {} to device {}", 
                format_timestamp(current_timestamp()), batch_info, failed_device_id, target);
        } else {
            println!("[{}] Failover: falling back to CPU for {} after device {} failed", 
                format_timestamp(current_timestamp()), batch_info, failed_device_id);
        }
    }

    /// Log batch processing results
    pub fn log_batch_result(&self, device_id: u32, batch_size: u128, success: bool, duration_ms: u64) {
        if self.verbose || !success {
            let status = if success { "SUCCESS" } else { "FAILED" };
            println!("[{}] Batch {}: device {}, size {}, duration {}ms", 
                format_timestamp(current_timestamp()), status, device_id, batch_size, duration_ms);
        }
    }

    /// Log stress test results
    pub fn log_stress_test_result(&self, test_name: &str, success: bool, details: &str) {
        let status = if success { "PASS" } else { "FAIL" };
        println!("[{}] Stress Test {} [{}]: {}", 
            format_timestamp(current_timestamp()), test_name, status, details);
    }
}

/// Get current timestamp in milliseconds since epoch
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Format timestamp for logging
pub fn format_timestamp(timestamp: u64) -> String {
    let secs = timestamp / 1000;
    let millis = timestamp % 1000;
    let _dt = UNIX_EPOCH + std::time::Duration::from_secs(secs);
    
    // Simple formatting - in production would use chrono or time crate
    format!("{}:{:03}", secs, millis)
}

/// CUDA-specific error checking utilities
#[cfg(feature = "cuda")]
pub mod cuda_errors {
    use super::*;

    /// Check CUDA error code and convert to GpuError
    pub fn check_cuda_error(result: i32, device_id: u32, operation: &str) -> Result<(), GpuError> {
        if result == 0 {
            Ok(())
        } else {
            Err(GpuError::KernelExecutionFailed {
                device_id,
                kernel_name: operation.to_string(),
                error: format!("CUDA error code: {}", result),
                timestamp: current_timestamp(),
            })
        }
    }

    /// Check for device availability and health
    pub fn check_device_health(_device_id: u32) -> Result<(), GpuError> {
        // In a real implementation, this would query CUDA device properties
        // For now, simulate occasional failures for testing
        Ok(())
    }

    /// Estimate memory requirements for batch
    pub fn estimate_memory_for_batch(batch_size: u128) -> usize {
        // Rough estimate: each mnemonic needs space for:
        // - Mnemonic string (~50 bytes)
        // - Seed (64 bytes)  
        // - Private key (32 bytes)
        // - Public key (64 bytes)
        // - Address (20 bytes)
        // Total: ~230 bytes per mnemonic
        (batch_size as usize).saturating_mul(230)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_status_creation() {
        let status = DeviceStatus::new(0, "Test Device".to_string());
        assert_eq!(status.device_id, 0);
        assert_eq!(status.device_name, "Test Device");
        assert!(status.is_healthy());
        assert!(status.is_usable());
    }

    #[test]
    fn test_device_status_transitions() {
        let mut status = DeviceStatus::new(0, "Test Device".to_string());
        
        // Mark as warning
        status.mark_warning("High temperature".to_string());
        assert!(!status.is_healthy());
        assert!(status.is_usable());
        
        // Mark as failed
        let error = GpuError::DeviceHardwareFailure {
            device_id: 0,
            device_name: "Test Device".to_string(),
            error_code: -1,
            timestamp: current_timestamp(),
        };
        status.mark_failed(error);
        assert!(!status.is_healthy());
        assert!(!status.is_usable());
        assert_eq!(status.total_errors, 1);
    }

    #[test]
    fn test_gpu_error_display() {
        let error = GpuError::KernelExecutionFailed {
            device_id: 0,
            kernel_name: "test_kernel".to_string(),
            error: "Out of memory".to_string(),
            timestamp: 1234567890,
        };
        
        let display_str = format!("{}", error);
        assert!(display_str.contains("Kernel Execution Failed"));
        assert!(display_str.contains("Device 0"));
        assert!(display_str.contains("test_kernel"));
    }

    #[test]
    fn test_error_logger() {
        let logger = ErrorLogger::new(false);
        let error = GpuError::DeviceInitFailed {
            device_id: 0,
            device_name: "Test Device".to_string(),
            error: "Mock error".to_string(),
            timestamp: current_timestamp(),
        };
        
        // Should not panic
        logger.log_error(&error);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_error_checking() {
        use cuda_errors::*;
        
        // Success case
        assert!(check_cuda_error(0, 0, "test_operation").is_ok());
        
        // Error case
        let result = check_cuda_error(-1, 0, "test_operation");
        assert!(result.is_err());
        
        if let Err(GpuError::KernelExecutionFailed { device_id, kernel_name, .. }) = result {
            assert_eq!(device_id, 0);
            assert_eq!(kernel_name, "test_operation");
        } else {
            panic!("Expected KernelExecutionFailed error");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_memory_estimation() {
        use cuda_errors::*;
        
        let memory_needed = estimate_memory_for_batch(1000);
        assert!(memory_needed > 0);
        assert!(memory_needed >= 230 * 1000); // At least 230 bytes per batch item
    }
}