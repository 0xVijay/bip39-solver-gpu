use crate::config::{Config, GpuConfig};
use crate::cuda_backend::CudaBackend;
use crate::gpu_backend::{GpuBackend, GpuBatchResult, GpuDevice};
use crate::opencl_backend::OpenClBackend;
use crate::word_space::WordSpace;
use crate::error_handling::{GpuError, DeviceStatus, ErrorLogger, current_timestamp};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// Import GPU memory functions directly
use crate::gpu_memory::{calculate_optimal_batch_size, get_memory_info};

/// GPU manager for handling multiple GPU backends and devices with advanced error handling
pub struct GpuManager {
    backend: Box<dyn GpuBackend>,
    devices: Vec<GpuDevice>,
    config: GpuConfig,
    device_status: Arc<Mutex<Vec<DeviceStatus>>>,
    error_logger: ErrorLogger,
}

impl GpuManager {
    /// Create a new GPU manager from configuration
    pub fn from_config(config: &Config) -> Result<Self, Box<dyn Error>> {
        let gpu_config = config
            .gpu
            .as_ref()
            .unwrap_or(&GpuConfig {
                backend: "opencl".to_string(),
                devices: vec![],
                multi_gpu: false,
            })
            .clone();

        let mut backend: Box<dyn GpuBackend> = match gpu_config.backend.as_str() {
            "cuda" => {
                println!("Initializing CUDA backend...");
                let mut cuda_backend = CudaBackend::new();

                // Set up word space for CUDA backend
                let word_space = WordSpace::from_config(config);
                cuda_backend.set_word_space(Arc::new(word_space));

                Box::new(cuda_backend)
            }
            "opencl" | _ => {
                println!("Initializing OpenCL backend...");
                let mut opencl_backend = OpenClBackend::new();

                // Set up word space for OpenCL backend
                let word_space = WordSpace::from_config(config);
                opencl_backend.set_word_space(Arc::new(word_space));

                Box::new(opencl_backend)
            }
        };

        // Initialize the backend
        backend.initialize()?;

        // Check if backend is available
        if !backend.is_available() {
            return Err(format!(
                "GPU backend '{}' is not available on this system",
                gpu_config.backend
            )
            .into());
        }

        // Enumerate devices
        let all_devices = backend.enumerate_devices()?;

        // Filter devices based on configuration
        let devices = if gpu_config.devices.is_empty() {
            // Use all available devices
            all_devices
        } else {
            // Use only specified devices
            all_devices
                .into_iter()
                .filter(|device| gpu_config.devices.contains(&device.id))
                .collect()
        };

        if devices.is_empty() {
            return Err("No GPU devices available or specified devices not found".into());
        }

        println!(
            "GPU Manager initialized with {} backend and {} device(s)",
            backend.backend_name(),
            devices.len()
        );

        // Initialize device status tracking
        let mut device_status_vec = Vec::new();
        for device in &devices {
            println!(
                "  Device {}: {} ({} MB memory, {} compute units)",
                device.id,
                device.name,
                device.memory / 1024 / 1024,
                device.compute_units
            );
            
            device_status_vec.push(DeviceStatus::new(device.id, device.name.clone()));
        }

        Ok(GpuManager {
            backend,
            devices,
            config: gpu_config,
            device_status: Arc::new(Mutex::new(device_status_vec)),
            error_logger: ErrorLogger::new(false), // Non-verbose for manager
        })
    }

    /// Get the backend name
    pub fn backend_name(&self) -> &'static str {
        self.backend.backend_name()
    }

    /// Get the list of available devices
    pub fn devices(&self) -> &[GpuDevice] {
        &self.devices
    }

    /// Check if multi-GPU processing is enabled
    pub fn is_multi_gpu_enabled(&self) -> bool {
        self.config.multi_gpu && self.devices.len() > 1
    }

    /// Check if multi-GPU was requested in config
    pub fn is_multi_gpu_requested(&self) -> bool {
        self.config.multi_gpu
    }

    /// Get healthy devices that can accept work
    pub fn get_healthy_devices(&self) -> Vec<u32> {
        if let Ok(status_vec) = self.device_status.lock() {
            status_vec.iter()
                .filter(|status| status.is_usable())
                .map(|status| status.device_id)
                .collect()
        } else {
            // Fallback to all devices if status lock fails
            self.devices.iter().map(|d| d.id).collect()
        }
    }

    /// Mark device as failed and remove from active pool
    fn mark_device_failed(&self, device_id: u32, error: GpuError) -> bool {
        if let Ok(mut status_vec) = self.device_status.lock() {
            if let Some(status) = status_vec.iter_mut().find(|s| s.device_id == device_id) {
                let old_status = status.clone();
                status.mark_failed(error.clone());
                self.error_logger.log_device_status_change(&old_status, status);
                self.error_logger.log_error(&error);
                return true;
            }
        }
        false
    }

    /// Attempt to recover a failed device
    fn attempt_device_recovery(&self, device_id: u32) -> bool {
        // In a real implementation, this would:
        // 1. Check if device is responsive
        // 2. Reset device context if needed
        // 3. Reinitialize device memory
        // 4. Update device status
        
        if let Ok(mut status_vec) = self.device_status.lock() {
            if let Some(status) = status_vec.iter_mut().find(|s| s.device_id == device_id) {
                if !status.is_usable() {
                    self.error_logger.log_recovery_attempt(device_id, &status.device_name, 1);
                    
                    // Mock recovery attempt
                    status.mark_healthy();
                    println!("Device {} recovery attempt completed", device_id);
                    return true;
                }
            }
        }
        false
    }

    /// Perform periodic health checks on all devices
    pub fn perform_health_checks(&self) {
        if let Ok(mut status_vec) = self.device_status.lock() {
            for status in status_vec.iter_mut() {
                if status.is_usable() {
                    // Perform a quick health check
                    // In a real implementation, this would query device properties
                    status.mark_healthy(); // Mock healthy status
                }
            }
        }
    }

    /// Get device status for monitoring
    pub fn get_device_status(&self) -> Vec<DeviceStatus> {
        if let Ok(status_vec) = self.device_status.lock() {
            status_vec.clone()
        } else {
            Vec::new()
        }
    }

    /// Get optimal batch size for a specific device
    pub fn get_optimal_batch_size(&self, device_id: u32) -> Option<u128> {
        self.devices.iter()
            .find(|device| device.id == device_id)
            .map(|device| calculate_optimal_batch_size(device))
    }

    /// Get optimal batch size for all devices
    pub fn get_optimal_batch_sizes(&self) -> Vec<(u32, u128)> {
        self.devices.iter()
            .map(|device| (device.id, calculate_optimal_batch_size(device)))
            .collect()
    }

    /// Get memory info for a device and batch size
    pub fn get_device_memory_info(&self, device_id: u32, batch_size: u128) -> Option<String> {
        self.devices.iter()
            .find(|device| device.id == device_id)
            .map(|device| get_memory_info(device, batch_size))
    }

    /// Find alternative device for failover
    fn find_failover_device(&self, failed_device_id: u32) -> Option<u32> {
        let healthy_devices = self.get_healthy_devices();
        healthy_devices.into_iter()
            .find(|&id| id != failed_device_id)
    }

    /// Execute a batch on a specific device with failover support (GPU only)
    pub fn execute_batch(
        &self,
        device_id: u32,
        start_offset: u128,
        batch_size: u128,
        target_address: &str,
        derivation_path: &str,
        passphrase: &str,
    ) -> Result<GpuBatchResult, Box<dyn Error>> {
        let start_time = Instant::now();
        
        // Check if device is in healthy state
        let healthy_devices = self.get_healthy_devices();
        if !healthy_devices.contains(&device_id) {
            self.error_logger.log_error(&GpuError::DeviceHardwareFailure {
                device_id,
                device_name: format!("Device {}", device_id),
                error_code: -1,
                timestamp: current_timestamp(),
            });
            
            // Try to find an alternative device
            if let Some(alternative_device) = self.find_failover_device(device_id) {
                self.error_logger.log_failover(device_id, Some(alternative_device), 
                    &format!("batch size {}", batch_size));
                return self.execute_batch(alternative_device, start_offset, batch_size, 
                    target_address, derivation_path, passphrase);
            } else {
                return Err("No healthy GPU devices available".into());
            }
        }

        // Verify device exists in our device list
        if !self.devices.iter().any(|d| d.id == device_id) {
            let error = GpuError::DeviceRemoved {
                device_id,
                device_name: format!("Device {}", device_id),
                timestamp: current_timestamp(),
            };
            self.mark_device_failed(device_id, error.clone());
            return Err(error.to_string().into());
        }

        // Execute the batch with error handling
        match self.backend.execute_batch(
            device_id,
            start_offset,
            batch_size,
            target_address,
            derivation_path,
            passphrase,
        ) {
            Ok(result) => {
                // Update device status on success
                if let Ok(mut status_vec) = self.device_status.lock() {
                    if let Some(status) = status_vec.iter_mut().find(|s| s.device_id == device_id) {
                        status.increment_batch_count();
                        status.mark_healthy();
                    }
                }
                
                let duration = start_time.elapsed().as_millis() as u64;
                self.error_logger.log_batch_result(device_id, batch_size, true, duration);
                Ok(result)
            }
            Err(error) => {
                let duration = start_time.elapsed().as_millis() as u64;
                self.error_logger.log_batch_result(device_id, batch_size, false, duration);
                
                // Convert error to GpuError and mark device as failed
                let gpu_error = GpuError::KernelExecutionFailed {
                    device_id,
                    kernel_name: "batch_execution".to_string(),
                    error: error.to_string(),
                    timestamp: current_timestamp(),
                };
                
                self.mark_device_failed(device_id, gpu_error);
                
                // Attempt recovery
                if self.attempt_device_recovery(device_id) {
                    // Retry on the same device after recovery
                    return self.execute_batch(device_id, start_offset, batch_size, 
                        target_address, derivation_path, passphrase);
                }
                
                // Try failover to another device
                if let Some(alternative_device) = self.find_failover_device(device_id) {
                    self.error_logger.log_failover(device_id, Some(alternative_device), 
                        &format!("batch size {}", batch_size));
                    return self.execute_batch(alternative_device, start_offset, batch_size, 
                        target_address, derivation_path, passphrase);
                } else {
                    return Err("No healthy GPU devices available".into());
                }
            }
        }
    }

    /// Execute batches across multiple devices (if multi-GPU is enabled) with failover
    pub fn execute_multi_gpu_batch(
        &self,
        start_offset: u128,
        total_batch_size: u128,
        target_address: &str,
        derivation_path: &str,
        passphrase: &str,
    ) -> Result<Vec<GpuBatchResult>, Box<dyn Error>> {
        // Get currently healthy devices
        let healthy_devices = self.get_healthy_devices();
        
        if healthy_devices.is_empty() {
            return Err("No healthy GPU devices available".into());
        }

        if !self.is_multi_gpu_enabled() || healthy_devices.len() == 1 {
            // Single GPU execution
            let device_id = healthy_devices[0];
            let result = self.execute_batch(
                device_id,
                start_offset,
                total_batch_size,
                target_address,
                derivation_path,
                passphrase,
            )?;
            return Ok(vec![result]);
        }

        // Multi-GPU execution: divide work among healthy devices
        let device_count = healthy_devices.len();
        let batch_per_device = total_batch_size / device_count as u128;
        let remainder = total_batch_size % device_count as u128;

        let mut results = Vec::new();
        let mut current_offset = start_offset;
        let mut completed_devices = 0;

        for (i, &device_id) in healthy_devices.iter().enumerate() {
            let device_batch_size = if i == device_count - 1 {
                batch_per_device + remainder // Last device gets remainder
            } else {
                batch_per_device
            };

            if device_batch_size == 0 {
                continue;
            }

            match self.execute_batch(
                device_id,
                current_offset,
                device_batch_size,
                target_address,
                derivation_path,
                passphrase,
            ) {
                Ok(result) => {
                    results.push(result);
                    completed_devices += 1;
                    current_offset += device_batch_size;
                    
                    // If we found a match, we can stop
                    if results.last().unwrap().mnemonic.is_some() {
                        break;
                    }
                }
                Err(e) => {
                    println!("Device {} failed during multi-GPU execution: {}", device_id, e);
                    
                    // Try to redistribute work to remaining devices
                    let remaining_work = device_batch_size;
                    let remaining_healthy = self.get_healthy_devices();
                    
                    if !remaining_healthy.is_empty() {
                        println!("Redistributing {} work units to remaining devices", remaining_work);
                        // For simplicity, give the work to the first remaining healthy device
                        if let Some(&fallback_device) = remaining_healthy.first() {
                            match self.execute_batch(
                                fallback_device,
                                current_offset,
                                remaining_work,
                                target_address,
                                derivation_path,
                                passphrase,
                            ) {
                                Ok(fallback_result) => {
                                    results.push(fallback_result);
                                    completed_devices += 1;
                                }
                                Err(fallback_error) => {
                                    println!("Fallback device {} also failed: {}", fallback_device, fallback_error);
                                }
                            }
                        }
                    }
                    
                    current_offset += device_batch_size;
                }
            }
        }

        if completed_devices == 0 {
            return Err("All GPU devices failed".into());
        }

        Ok(results)
    }

    /// Shutdown the GPU manager
    pub fn shutdown(&mut self) -> Result<(), Box<dyn Error>> {
        println!("Shutting down GPU manager...");
        self.backend.shutdown()
    }
}

impl Drop for GpuManager {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EthereumConfig, GpuConfig};

    #[test]
    fn test_gpu_manager_creation() {
        let config = Config {
            word_constraints: vec![],
            ethereum: EthereumConfig {
                derivation_path: "m/44'/60'/0'/0/0".to_string(),
                target_address: "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f".to_string(),
            },
            slack: None,
            worker: None,
            gpu: Some(GpuConfig {
                backend: "opencl".to_string(),
                devices: vec![],
                multi_gpu: false,
            }),
            batch_size: 1000,
            passphrase: "".to_string(),
        };

        let manager = GpuManager::from_config(&config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_gpu_manager_device_enumeration() {
        let config = Config {
            word_constraints: vec![],
            ethereum: EthereumConfig {
                derivation_path: "m/44'/60'/0'/0/0".to_string(),
                target_address: "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f".to_string(),
            },
            slack: None,
            worker: None,
            gpu: Some(GpuConfig {
                backend: "opencl".to_string(),
                devices: vec![],
                multi_gpu: true,
            }),
            batch_size: 1000,
            passphrase: "".to_string(),
        };

        if let Ok(manager) = GpuManager::from_config(&config) {
            assert!(!manager.devices().is_empty());
            assert_eq!(manager.backend_name(), "OpenCL");
        }
    }
}
