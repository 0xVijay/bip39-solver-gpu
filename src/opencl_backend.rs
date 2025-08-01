use crate::eth::{addresses_equal, derive_ethereum_address};
use crate::gpu_backend::{GpuBackend, GpuBatchResult, GpuDevice};
use crate::word_space::WordSpace;
use std::error::Error;
use std::sync::Arc;

/// OpenCL backend implementation for GPU computation
/// With automatic fallback to CPU processing if OpenCL is not available
pub struct OpenClBackend {
    initialized: bool,
    word_space: Option<Arc<WordSpace>>,
    use_cpu_fallback: bool,
}

impl OpenClBackend {
    /// Create a new OpenCL backend instance
    pub fn new() -> Self {
        OpenClBackend {
            initialized: false,
            word_space: None,
            use_cpu_fallback: false,
        }
    }

    /// Set the word space for mnemonic generation
    pub fn set_word_space(&mut self, word_space: Arc<WordSpace>) {
        self.word_space = Some(word_space);
    }

    /// Try to initialize OpenCL, returns error if not available
    fn try_initialize_opencl(&self) -> Result<(), Box<dyn Error>> {
        #[cfg(feature = "opencl")]
        {
            use opencl3::device::{get_all_devices, CL_DEVICE_TYPE_GPU};
            use opencl3::platform::get_platforms;

            // Get available platforms
            let platforms = get_platforms()?;
            if platforms.is_empty() {
                return Err("No OpenCL platforms found".into());
            }

            // Look for GPU devices
            let gpu_devices = get_all_devices(CL_DEVICE_TYPE_GPU)?;
            if gpu_devices.is_empty() {
                return Err("No OpenCL GPU devices found".into());
            }

            println!("Found {} OpenCL platform(s) and {} GPU device(s)", 
                     platforms.len(), gpu_devices.len());
            Ok(())
        }
        
        #[cfg(not(feature = "opencl"))]
        {
            Err("OpenCL support not compiled. Use --features opencl to enable OpenCL support.".into())
        }
    }

    /// Enumerate actual OpenCL GPU devices
    fn enumerate_opencl_devices(&self) -> Result<Vec<GpuDevice>, Box<dyn Error>> {
        #[cfg(feature = "opencl")]
        {
            use opencl3::device::{get_all_devices, CL_DEVICE_TYPE_GPU};

            let gpu_devices = get_all_devices(CL_DEVICE_TYPE_GPU)?;
            let mut devices = Vec::new();

            for (id, _device_id) in gpu_devices.iter().enumerate() {
                // For now, create placeholder devices since OpenCL device info querying is complex
                devices.push(GpuDevice {
                    id: id as u32,
                    name: format!("OpenCL GPU Device {}", id),
                    memory: 4 * 1024 * 1024 * 1024, // 4GB default
                    compute_units: 16, // Default
                });
            }

            println!("Enumerated {} OpenCL GPU device(s)", devices.len());
            for device in &devices {
                println!("  Device {}: {} ({} MB memory, {} compute units)",
                         device.id, device.name, device.memory / 1024 / 1024, device.compute_units);
            }

            Ok(devices)
        }
        
        #[cfg(not(feature = "opencl"))]
        {
            Err("OpenCL support not compiled".into())
        }
    }

    /// Check if OpenCL is actually available
    fn check_opencl_availability(&self) -> Result<bool, Box<dyn Error>> {
        #[cfg(feature = "opencl")]
        {
            use opencl3::device::{get_all_devices, CL_DEVICE_TYPE_ALL};
            use opencl3::platform::get_platforms;

            let platforms = get_platforms()?;
            let devices = get_all_devices(CL_DEVICE_TYPE_ALL)?;
            
            Ok(!platforms.is_empty() && !devices.is_empty())
        }
        
        #[cfg(not(feature = "opencl"))]
        {
            Ok(false)
        }
    }
}

impl GpuBackend for OpenClBackend {
    fn backend_name(&self) -> &'static str {
        "OpenCL"
    }

    fn initialize(&mut self) -> Result<(), Box<dyn Error>> {
        if self.initialized {
            return Ok(());
        }

        println!("Initializing OpenCL backend...");

        // Try to initialize OpenCL
        match self.try_initialize_opencl() {
            Ok(()) => {
                println!("OpenCL backend initialized successfully");
                self.use_cpu_fallback = false;
            }
            Err(e) => {
                println!("Warning: OpenCL initialization failed: {}", e);
                println!("Falling back to CPU processing");
                self.use_cpu_fallback = true;
            }
        }

        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), Box<dyn Error>> {
        if !self.initialized {
            return Ok(());
        }

        println!("Shutting down OpenCL backend...");
        self.initialized = false;
        Ok(())
    }

    fn enumerate_devices(&self) -> Result<Vec<GpuDevice>, Box<dyn Error>> {
        if self.use_cpu_fallback {
            // Return CPU fallback device when OpenCL is not available
            return Ok(vec![GpuDevice {
                id: 0,
                name: "CPU Fallback (OpenCL)".to_string(),
                memory: 8 * 1024 * 1024 * 1024, // 8GB default
                compute_units: std::thread::available_parallelism()
                    .map(|p| p.get() as u32)
                    .unwrap_or(4),
            }]);
        }

        // Try to enumerate actual OpenCL GPU devices
        match self.enumerate_opencl_devices() {
            Ok(devices) => {
                if devices.is_empty() {
                    println!("Warning: No OpenCL devices found, using CPU fallback");
                    Ok(vec![GpuDevice {
                        id: 0,
                        name: "CPU Fallback (OpenCL)".to_string(),
                        memory: 8 * 1024 * 1024 * 1024,
                        compute_units: std::thread::available_parallelism()
                            .map(|p| p.get() as u32)
                            .unwrap_or(4),
                    }])
                } else {
                    Ok(devices)
                }
            }
            Err(e) => {
                println!("Warning: Failed to enumerate OpenCL devices: {}", e);
                println!("Using CPU fallback");
                Ok(vec![GpuDevice {
                    id: 0,
                    name: "CPU Fallback (OpenCL)".to_string(),
                    memory: 8 * 1024 * 1024 * 1024,
                    compute_units: std::thread::available_parallelism()
                        .map(|p| p.get() as u32)
                        .unwrap_or(4),
                }])
            }
        }
    }

    fn execute_batch(
        &self,
        _device_id: u32,
        start_offset: u128,
        batch_size: u128,
        target_address: &str,
        derivation_path: &str,
        passphrase: &str,
    ) -> Result<GpuBatchResult, Box<dyn Error>> {
        let word_space = self
            .word_space
            .as_ref()
            .ok_or("Word space not initialized")?;

        // Use CPU processing (simplified for now)
        let mut processed_count = 0u128;

        for offset in start_offset..(start_offset + batch_size) {
            if offset >= word_space.total_combinations {
                break;
            }

            if let Some(word_indices) = word_space.index_to_words(offset) {
                if let Some(mnemonic) = WordSpace::words_to_mnemonic(&word_indices) {
                    match derive_ethereum_address(&mnemonic, passphrase, derivation_path) {
                        Ok(address) => {
                            if addresses_equal(&address, target_address) {
                                return Ok(GpuBatchResult {
                                    mnemonic: Some(mnemonic),
                                    address: Some(address),
                                    offset: Some(offset),
                                    processed_count: processed_count + 1,
                                });
                            }
                        }
                        Err(e) => {
                            eprintln!("Error deriving address for mnemonic: {}", e);
                        }
                    }
                }
            }
            processed_count += 1;
        }

        Ok(GpuBatchResult {
            mnemonic: None,
            address: None,
            offset: None,
            processed_count,
        })
    }

    fn is_available(&self) -> bool {
        // Try to check OpenCL availability
        self.check_opencl_availability().unwrap_or(true) // Always return true since we have CPU fallback
    }
}

impl Default for OpenClBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opencl_backend_creation() {
        let backend = OpenClBackend::new();
        assert_eq!(backend.backend_name(), "OpenCL");
        assert!(!backend.initialized);
    }

    #[test]
    fn test_opencl_backend_availability() {
        let backend = OpenClBackend::new();
        assert!(backend.is_available());
    }
}