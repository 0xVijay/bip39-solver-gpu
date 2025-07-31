use crate::config::{Config, GpuConfig};
use crate::cuda_backend::CudaBackend;
use crate::gpu_backend::{GpuBackend, GpuBatchResult, GpuDevice};
use crate::opencl_backend::OpenClBackend;
use crate::word_space::WordSpace;
use std::error::Error;
use std::sync::Arc;

/// GPU manager for handling multiple GPU backends and devices
pub struct GpuManager {
    backend: Box<dyn GpuBackend>,
    devices: Vec<GpuDevice>,
    config: GpuConfig,
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

        for device in &devices {
            println!(
                "  Device {}: {} ({} MB memory, {} compute units)",
                device.id,
                device.name,
                device.memory / 1024 / 1024,
                device.compute_units
            );
        }

        Ok(GpuManager {
            backend,
            devices,
            config: gpu_config,
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

    /// Execute a batch on a specific device
    pub fn execute_batch(
        &self,
        device_id: u32,
        start_offset: u128,
        batch_size: u128,
        target_address: &str,
        derivation_path: &str,
        passphrase: &str,
    ) -> Result<GpuBatchResult, Box<dyn Error>> {
        // Verify device exists
        if !self.devices.iter().any(|d| d.id == device_id) {
            return Err(format!("Device ID {} not found in available devices", device_id).into());
        }

        self.backend.execute_batch(
            device_id,
            start_offset,
            batch_size,
            target_address,
            derivation_path,
            passphrase,
        )
    }

    /// Execute batches across multiple devices (if multi-GPU is enabled)
    pub fn execute_multi_gpu_batch(
        &self,
        start_offset: u128,
        total_batch_size: u128,
        target_address: &str,
        derivation_path: &str,
        passphrase: &str,
    ) -> Result<Vec<GpuBatchResult>, Box<dyn Error>> {
        if !self.is_multi_gpu_enabled() {
            // Single GPU execution
            let result = self.execute_batch(
                self.devices[0].id,
                start_offset,
                total_batch_size,
                target_address,
                derivation_path,
                passphrase,
            )?;
            return Ok(vec![result]);
        }

        // Multi-GPU execution: divide work among devices
        let device_count = self.devices.len();
        let batch_per_device = total_batch_size / device_count as u128;
        let remainder = total_batch_size % device_count as u128;

        let mut results = Vec::new();
        let mut current_offset = start_offset;

        for (i, device) in self.devices.iter().enumerate() {
            let device_batch_size = if i == device_count - 1 {
                batch_per_device + remainder // Last device gets remainder
            } else {
                batch_per_device
            };

            if device_batch_size == 0 {
                continue;
            }

            let result = self.execute_batch(
                device.id,
                current_offset,
                device_batch_size,
                target_address,
                derivation_path,
                passphrase,
            )?;

            results.push(result);
            current_offset += device_batch_size;
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
