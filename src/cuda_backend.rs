use crate::gpu_backend::{GpuBackend, GpuBatchResult, GpuDevice};
use crate::word_space::WordSpace;
use crate::error_handling::{GpuError, DeviceStatus, ErrorLogger, current_timestamp};
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// CUDA device properties structure
#[cfg(feature = "cuda")]
#[allow(dead_code)]
#[repr(C)]
struct CudaDeviceProperties {
    name: [u8; 256],
    total_global_mem: usize,
    multiprocessor_count: i32,
    max_threads_per_block: i32,
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
#[repr(C)]
struct CudaDeviceProperties {
    name: [u8; 256],
    total_global_mem: usize,
    multiprocessor_count: i32,
    max_threads_per_block: i32,
}

// CUDA runtime API functions - only available when CUDA is compiled and available
#[cfg(all(feature = "cuda", cuda_available))]
extern "C" {
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaGetDeviceProperties(prop: *mut CudaDeviceProperties, device: i32) -> i32;
    
    // GPU pipeline functions
    fn cuda_complete_pipeline_host(
        mnemonics: *const *const std::os::raw::c_char,
        passphrases: *const *const std::os::raw::c_char,
        address_indices: *const std::os::raw::c_uint,
        target_address: *const u8,
        found_addresses: *mut u8,
        match_results: *mut std::os::raw::c_uint,
        count: std::os::raw::c_uint,
    ) -> std::os::raw::c_int;
}

// Helper function names to avoid conflict - only available when CUDA is compiled and available
#[cfg(all(feature = "cuda", cuda_available))]
unsafe fn cuda_get_device_count(count: *mut i32) -> i32 {
    cudaGetDeviceCount(count)
}

#[cfg(all(feature = "cuda", cuda_available))]
unsafe fn cuda_get_device_properties(prop: *mut CudaDeviceProperties, device: i32) -> i32 {
    cudaGetDeviceProperties(prop, device)
}

#[cfg(not(all(feature = "cuda", cuda_available)))]
#[allow(dead_code)]
unsafe fn cuda_get_device_count(_count: *mut i32) -> i32 {
    -1
}

#[cfg(not(all(feature = "cuda", cuda_available)))]
unsafe fn cuda_get_device_properties(_prop: *mut CudaDeviceProperties, _device: i32) -> i32 {
    -1
}

/// CUDA backend implementation with advanced error handling and failover
pub struct CudaBackend {
    initialized: bool,
    word_space: Option<Arc<WordSpace>>,
    device_status: Arc<Mutex<Vec<DeviceStatus>>>,
    error_logger: ErrorLogger,
}

impl CudaBackend {
    /// Create a new CUDA backend instance with error handling
    pub fn new() -> Self {
        CudaBackend {
            initialized: false,
            word_space: None,
            device_status: Arc::new(Mutex::new(Vec::new())),
            error_logger: ErrorLogger::new(true), // Verbose logging for development
        }
    }

    /// Set the word space for mnemonic generation
    pub fn set_word_space(&mut self, word_space: Arc<WordSpace>) {
        self.word_space = Some(word_space);
    }

    /// Check if CUDA is available by trying to get device count
    #[cfg(feature = "cuda")]
    fn is_cuda_available(&self) -> bool {
        let mut count = 0;
        unsafe {
            cuda_get_device_count(&mut count) == 0 && count > 0
        }
    }
    
    #[cfg(not(feature = "cuda"))]
    fn is_cuda_available(&self) -> bool {
        false
    }

    /// Get the number of CUDA devices
    #[cfg(feature = "cuda")]
    fn get_device_count(&self) -> Result<i32, String> {
        let mut count = 0;
        let result = unsafe { cuda_get_device_count(&mut count) };
        
        if result == 0 {
            Ok(count)
        } else {
            Err(format!("CUDA error getting device count: {}", result))
        }
    }
    
    #[cfg(not(feature = "cuda"))]
    fn get_device_count(&self) -> Result<i32, String> {
        Err("CUDA support not compiled".to_string())
    }

    /// Complete GPU pipeline: Mnemonic → Address with target matching - only when CUDA is available
    #[cfg(all(feature = "cuda", cuda_available))]
    fn gpu_complete_pipeline_batch(
        &self,
        mnemonics: &[String],
        passphrases: &[String],
        address_indices: &[u32],
        target_address: &[u8; 20],
    ) -> Result<(Vec<[u8; 20]>, Vec<bool>), String> {
        use std::os::raw::{c_char, c_uint};
        
        if mnemonics.len() != passphrases.len() || mnemonics.len() != address_indices.len() {
            return Err("All input arrays must have same length".to_string());
        }
        
        let count = mnemonics.len() as c_uint;
        let mut found_addresses = vec![0u8; (count as usize) * 20];
        let mut match_results = vec![0u32; count as usize];
        
        // Convert strings to C string pointers
        let mnemonic_cstrings: Vec<std::ffi::CString> = mnemonics
            .iter()
            .map(|s| std::ffi::CString::new(s.as_str()).unwrap())
            .collect();
        let passphrase_cstrings: Vec<std::ffi::CString> = passphrases
            .iter()
            .map(|s| std::ffi::CString::new(s.as_str()).unwrap())
            .collect();
        
        let mnemonic_ptrs: Vec<*const c_char> = mnemonic_cstrings
            .iter()
            .map(|cs| cs.as_ptr())
            .collect();
        let passphrase_ptrs: Vec<*const c_char> = passphrase_cstrings
            .iter()
            .map(|cs| cs.as_ptr())
            .collect();
        
        unsafe {
            let result = cuda_complete_pipeline_host(
                mnemonic_ptrs.as_ptr(),
                passphrase_ptrs.as_ptr(),
                address_indices.as_ptr() as *const c_uint,
                target_address.as_ptr(),
                found_addresses.as_mut_ptr(),
                match_results.as_mut_ptr(),
                count,
            );
            
            if result != 0 {
                return Err("CUDA complete pipeline kernel execution failed".to_string());
            }
        }
        
        // Convert results
        let mut addresses = Vec::new();
        for i in 0..count as usize {
            let mut address = [0u8; 20];
            address.copy_from_slice(&found_addresses[i * 20..(i + 1) * 20]);
            addresses.push(address);
        }
        
        let matches: Vec<bool> = match_results.iter().map(|&r| r != 0).collect();
        
        Ok((addresses, matches))
    }
    
    #[cfg(not(all(feature = "cuda", cuda_available)))]
    #[allow(dead_code)]
    fn gpu_complete_pipeline_batch(
        &self,
        _mnemonics: &[String],
        _passphrases: &[String],
        _address_indices: &[u32],
        _target_address: &[u8; 20],
    ) -> Result<(Vec<[u8; 20]>, Vec<bool>), String> {
        Err("CUDA support not compiled. Use --features cuda to enable GPU acceleration.".to_string())
    }

    /// Check device health and update status
    fn check_device_health(&self, _device_id: u32) -> Result<(), GpuError> {
        #[cfg(feature = "cuda")]
        {
            use crate::error_handling::cuda_errors::check_device_health;
            check_device_health(_device_id)
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            // Simulate device check for non-CUDA builds
            Ok(())
        }
    }

    /// Get available healthy devices
    fn get_healthy_devices(&self) -> Vec<u32> {
        if let Ok(status_vec) = self.device_status.lock() {
            status_vec.iter()
                .filter(|status| status.is_usable())
                .map(|status| status.device_id)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Mark device as failed and update status
    fn mark_device_failed(&self, device_id: u32, error: GpuError) {
        if let Ok(mut status_vec) = self.device_status.lock() {
            if let Some(status) = status_vec.iter_mut().find(|s| s.device_id == device_id) {
                let old_status = status.clone();
                status.mark_failed(error.clone());
                self.error_logger.log_device_status_change(&old_status, status);
            }
        }
        self.error_logger.log_error(&error);
    }

    /// Execute batch with comprehensive error handling and recovery
    fn execute_batch_with_recovery(
        &self,
        device_id: u32,
        start_offset: u128,
        batch_size: u128,
        target_address: &str,
        derivation_path: &str,
        passphrase: &str,
    ) -> Result<GpuBatchResult, GpuError> {
        let start_time = Instant::now();
        
        // Check if device is available
        if !self.get_healthy_devices().contains(&device_id) {
            return Err(GpuError::DeviceHardwareFailure {
                device_id,
                device_name: format!("CUDA Device {}", device_id),
                error_code: -1,
                timestamp: current_timestamp(),
            });
        }

        // Perform health check before execution
        if let Err(error) = self.check_device_health(device_id) {
            self.mark_device_failed(device_id, error.clone());
            return Err(error);
        }

        // Estimate memory requirements
        #[cfg(feature = "cuda")]
        let estimated_memory = {
            use crate::error_handling::cuda_errors::estimate_memory_for_batch;
            estimate_memory_for_batch(batch_size)
        };
        #[cfg(not(feature = "cuda"))]
        let estimated_memory = (batch_size as usize).saturating_mul(230);

        // Check available memory (simplified check)
        if estimated_memory > 8 * 1024 * 1024 * 1024 { // 8GB limit
            let error = GpuError::OutOfMemory {
                device_id,
                batch_size,
                available_memory: 8 * 1024 * 1024 * 1024,
                timestamp: current_timestamp(),
            };
            self.error_logger.log_error(&error);
            return Err(error);
        }

        // Execute the actual batch processing
        let result = self.execute_batch_internal(
            device_id, start_offset, batch_size, target_address, derivation_path, passphrase
        );

        let duration = start_time.elapsed().as_millis() as u64;
        
        match &result {
            Ok(_) => {
                // Update device status on success
                if let Ok(mut status_vec) = self.device_status.lock() {
                    if let Some(status) = status_vec.iter_mut().find(|s| s.device_id == device_id) {
                        status.increment_batch_count();
                        status.mark_healthy();
                    }
                }
                self.error_logger.log_batch_result(device_id, batch_size, true, duration);
            }
            Err(error) => {
                self.error_logger.log_batch_result(device_id, batch_size, false, duration);
                
                // Convert generic error to GpuError for proper handling
                let gpu_error = GpuError::KernelExecutionFailed {
                    device_id,
                    kernel_name: "batch_execution".to_string(),
                    error: error.to_string(),
                    timestamp: current_timestamp(),
                };
                self.mark_device_failed(device_id, gpu_error);
            }
        }

        result.map_err(|e| GpuError::KernelExecutionFailed {
            device_id,
            kernel_name: "batch_execution".to_string(),
            error: e.to_string(),
            timestamp: current_timestamp(),
        })
    }

    /// Internal batch execution with GPU-optimized processing
    fn execute_batch_internal(
        &self,
        _device_id: u32,
        _start_offset: u128,
        _batch_size: u128,
        _target_address: &str,
        _derivation_path: &str,
        _passphrase: &str,
    ) -> Result<GpuBatchResult, Box<dyn Error>> {
        // Use GPU-optimized processing with real CUDA kernels
        self.execute_batch_gpu_optimized(
            _start_offset,
            _batch_size,
            _target_address,
            _derivation_path,
            _passphrase,
        )
    }

    /// GPU-optimized batch execution using CUDA kernels
    fn execute_batch_gpu_optimized(
        &self,
        start_offset: u128,
        batch_size: u128,
        target_address: &str,
        _derivation_path: &str,
        passphrase: &str,
    ) -> Result<GpuBatchResult, Box<dyn Error>> {
        let word_space = self
            .word_space
            .as_ref()
            .ok_or("Word space not initialized")?;

        // Convert target address to bytes
        let target_bytes = if target_address.starts_with("0x") || target_address.starts_with("0X") {
            hex::decode(&target_address[2..])
                .map_err(|_| "Invalid target address format")?
        } else {
            hex::decode(target_address)
                .map_err(|_| "Invalid target address format")?
        };
        
        if target_bytes.len() != 20 {
            return Err("Target address must be 20 bytes".into());
        }
        
        let mut target_address_array = [0u8; 20];
        target_address_array.copy_from_slice(&target_bytes);

        // Generate mnemonics for this batch
        let mut mnemonics = Vec::new();
        let mut passphrases = Vec::new();
        let mut address_indices = Vec::new();
        
        let batch_end = (start_offset + batch_size).min(word_space.total_combinations);
        
        for offset in start_offset..batch_end {
            if let Some(word_indices) = word_space.index_to_words(offset) {
                if let Some(mnemonic) = WordSpace::words_to_mnemonic(&word_indices) {
                    mnemonics.push(mnemonic);
                    passphrases.push(passphrase.to_string());
                    address_indices.push(2); // BIP44 address index (m/44'/60'/0'/0/2)
                }
            }
        }

        if mnemonics.is_empty() {
            return Ok(GpuBatchResult {
                mnemonic: None,
                address: None,
                offset: None,
                processed_count: 0,
            });
        }

        // Use GPU complete pipeline for maximum performance
        #[cfg(all(feature = "cuda", cuda_available))]
        {
            match self.gpu_complete_pipeline_batch(
                &mnemonics,
                &passphrases,
                &address_indices,
                &target_address_array,
            ) {
                Ok((addresses, matches)) => {
                    // Check for matches
                    for (i, &is_match) in matches.iter().enumerate() {
                        if is_match {
                            let found_offset = start_offset + i as u128;
                            let address_hex = format!("0x{}", hex::encode(&addresses[i]));
                            
                            return Ok(GpuBatchResult {
                                mnemonic: Some(mnemonics[i].clone()),
                                address: Some(address_hex),
                                offset: Some(found_offset),
                                processed_count: mnemonics.len() as u128,
                            });
                        }
                    }
                    
                    Ok(GpuBatchResult {
                        mnemonic: None,
                        address: None,
                        offset: None,
                        processed_count: mnemonics.len() as u128,
                    })
                }
                Err(e) => {
                    // Return GPU error instead of falling back to CPU
                    return Err(format!("CUDA GPU kernel execution failed: {}", e).into());
                }
            }
        }
        
        #[cfg(all(feature = "cuda", not(cuda_available)))]
        {
            return Err("CUDA toolkit not available at build time. Install CUDA toolkit and rebuild.".into());
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            return Err("CUDA support not compiled. Use --features cuda to enable CUDA support.".into());
        }
    }

}

impl GpuBackend for CudaBackend {
    fn backend_name(&self) -> &'static str {
        "CUDA"
    }

    fn initialize(&mut self) -> Result<(), Box<dyn Error>> {
        if self.initialized {
            return Ok(());
        }

        println!("Initializing CUDA backend with advanced error handling...");

        // Check if CUDA is available
        if !self.is_cuda_available() {
            let error = GpuError::BackendUnavailable {
                backend_name: "CUDA".to_string(),
                reason: "CUDA runtime not available".to_string(),
                timestamp: current_timestamp(),
            };
            self.error_logger.log_error(&error);
            return Err("CUDA runtime not available".into());
        }

        // Get device count to verify CUDA setup
        match self.get_device_count() {
            Ok(count) if count > 0 => {
                println!("Found {} CUDA device(s)", count);
                
                // Initialize device status tracking
                let mut status_vec = Vec::new();
                for i in 0..count {
                    let device_name = format!("CUDA Device {}", i);
                    let status = DeviceStatus::new(i as u32, device_name);
                    status_vec.push(status);
                }
                
                if let Ok(mut device_status) = self.device_status.lock() {
                    *device_status = status_vec;
                }
            }
            Ok(_) => {
                let error = GpuError::BackendUnavailable {
                    backend_name: "CUDA".to_string(),
                    reason: "No CUDA devices found".to_string(),
                    timestamp: current_timestamp(),
                };
                self.error_logger.log_error(&error);
                return Err("No CUDA devices found".into());
            }
            Err(e) => {
                let error = GpuError::BackendUnavailable {
                    backend_name: "CUDA".to_string(),
                    reason: format!("Failed to enumerate CUDA devices: {}", e),
                    timestamp: current_timestamp(),
                };
                self.error_logger.log_error(&error);
                return Err(format!("Failed to enumerate CUDA devices: {}", e).into());
            }
        }

        self.initialized = true;
        println!("CUDA backend initialized successfully with error handling");
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), Box<dyn Error>> {
        if !self.initialized {
            return Ok(());
        }

        println!("Shutting down CUDA backend...");

        // Cleanup CUDA resources
        // In a real implementation, this would cleanup CUDA contexts

        self.initialized = false;
        Ok(())
    }

    fn enumerate_devices(&self) -> Result<Vec<GpuDevice>, Box<dyn Error>> {
        println!("Enumerating CUDA devices...");

        let device_count = self.get_device_count()
            .map_err(|e| format!("Failed to get CUDA device count: {}", e))?;

        let mut devices = Vec::new();

        for i in 0..device_count {
            let mut properties = CudaDeviceProperties {
                name: [0; 256],
                total_global_mem: 0,
                multiprocessor_count: 0,
                max_threads_per_block: 0,
            };

            let result = unsafe { cuda_get_device_properties(&mut properties, i) };
            
            let (device_name, memory, compute_units) = if result == 0 {
                // Extract device name from C-string
                let name_end = properties.name.iter().position(|&c| c == 0).unwrap_or(255);
                let device_name = String::from_utf8_lossy(&properties.name[..name_end]).to_string();
                
                (
                    device_name,
                    properties.total_global_mem as u64,
                    properties.multiprocessor_count as u32,
                )
            } else {
                // Fallback to generic naming if property query fails
                (
                    format!("CUDA Device {}", i),
                    8 * 1024 * 1024 * 1024, // 8GB default
                    80,                      // Default compute units
                )
            };

            devices.push(GpuDevice {
                id: i as u32,
                name: device_name,
                memory,
                compute_units,
            });
        }

        println!("Found {} CUDA device(s)", devices.len());
        Ok(devices)
    }

    fn execute_batch(
        &self,
        device_id: u32,
        start_offset: u128,
        batch_size: u128,
        target_address: &str,
        derivation_path: &str,
        passphrase: &str,
    ) -> Result<GpuBatchResult, Box<dyn Error>> {
        // Use the new error handling batch execution
        match self.execute_batch_with_recovery(
            device_id, start_offset, batch_size, target_address, derivation_path, passphrase
        ) {
            Ok(result) => Ok(result),
            Err(gpu_error) => {
                // Log the specific GPU error
                self.error_logger.log_error(&gpu_error);
                Err(gpu_error.to_string().into())
            }
        }
    }

    fn is_available(&self) -> bool {
        // Check if CUDA runtime is available
        self.is_cuda_available() && self.get_device_count().unwrap_or(0) > 0
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_backend_creation() {
        let backend = CudaBackend::new();
        assert_eq!(backend.backend_name(), "CUDA");
        assert!(!backend.initialized);
    }

    #[test]
    fn test_cuda_backend_availability() {
        let backend = CudaBackend::new();

        #[cfg(feature = "cuda")]
        {
            // With CUDA feature, it might be available (depends on runtime)
            // We can't guarantee it's available in test environment
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Without CUDA feature, should never be available
            assert!(!backend.is_available());
        }
    }

    #[test]
    fn test_cuda_device_enumeration() {
        let backend = CudaBackend::new();
        let devices = backend.enumerate_devices();

        #[cfg(feature = "cuda")]
        {
            assert!(devices.is_ok());
            let devices = devices.unwrap();
            assert!(!devices.is_empty());
            assert_eq!(devices[0].name, "CUDA Device 0");
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Without CUDA feature, device enumeration should fail
            assert!(devices.is_err());
        }
    }
}
