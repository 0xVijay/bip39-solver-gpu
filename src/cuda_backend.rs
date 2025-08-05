use crate::gpu_backend::{GpuBackend, GpuBatchResult, GpuDevice};
use bip39_solver_gpu::gpu_models::SUPPORTED_GPU_MODELS;
use crate::word_space::WordSpace;
use crate::error_handling::{GpuError, DeviceStatus, ErrorLogger, current_timestamp};
use crate::eth::{derive_ethereum_address, addresses_equal};
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
    /// This checks runtime availability, not just build-time compilation
    #[cfg(feature = "cuda")]
    fn is_cuda_available(&self) -> bool {
        // First check if CUDA was compiled with kernels
        #[cfg(cuda_available)]
        {
            let mut count = 0;
            unsafe {
                cuda_get_device_count(&mut count) == 0 && count > 0
            }
        }
        
        // If CUDA kernels weren't compiled, try dynamic loading (VastAI/Docker environments)
        #[cfg(not(cuda_available))]
        {
            self.check_cuda_runtime_dynamic()
        }
    }
    
    #[cfg(not(feature = "cuda"))]
    fn is_cuda_available(&self) -> bool {
        false
    }
    
    /// Check CUDA runtime availability dynamically (for VastAI/Docker environments)
    #[cfg(feature = "cuda")]
    fn check_cuda_runtime_dynamic(&self) -> bool {
        use std::process::Command;
        
        // Check if nvidia-smi works (indicates CUDA runtime is available)
        if let Ok(output) = Command::new("nvidia-smi").arg("--query-gpu=count").arg("--format=csv,noheader,nounits").output() {
            if output.status.success() {
                if let Ok(count_str) = String::from_utf8(output.stdout) {
                    if let Ok(count) = count_str.trim().parse::<i32>() {
                        println!("Found {} CUDA device(s) via nvidia-smi", count);
                        return count > 0;
                    }
                }
            }
        }
        
        // Check for CUDA library files
        let cuda_lib_paths = [
            "/usr/lib/x86_64-linux-gnu/libcudart.so",
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/local/cuda-12/lib64/libcudart.so",
            "/usr/local/cuda-11/lib64/libcudart.so",
            "/opt/cuda/lib64/libcudart.so",
        ];
        
        for path in &cuda_lib_paths {
            if std::path::Path::new(path).exists() {
                println!("Found CUDA runtime library at: {}", path);
                return true;
            }
        }
        
        false
    }

    /// Get the number of CUDA devices
    #[cfg(feature = "cuda")]
    fn get_device_count(&self) -> Result<i32, String> {
        // If CUDA was compiled with kernels, use direct API
        #[cfg(cuda_available)]
        {
            let mut count = 0;
            let result = unsafe { cuda_get_device_count(&mut count) };
            
            if result == 0 {
                return Ok(count);
            } else {
                return Err(format!("CUDA error getting device count: {}", result));
            }
        }
        
        // For VastAI/Docker environments, use nvidia-smi
        #[cfg(not(cuda_available))]
        {
            self.get_device_count_dynamic()
        }
    }
    
    #[cfg(not(feature = "cuda"))]
    fn get_device_count(&self) -> Result<i32, String> {
        Err("CUDA support not compiled".to_string())
    }
    
    /// Get device count via nvidia-smi (for VastAI/Docker environments)
    #[cfg(feature = "cuda")]
    fn get_device_count_dynamic(&self) -> Result<i32, String> {
        use std::process::Command;
        
        if let Ok(output) = Command::new("nvidia-smi").arg("--query-gpu=count").arg("--format=csv,noheader,nounits").output() {
            if output.status.success() {
                if let Ok(count_str) = String::from_utf8(output.stdout) {
                    if let Ok(count) = count_str.trim().parse::<i32>() {
                        return Ok(count);
                    }
                }
            }
        }
        
        // Alternative: count GPU devices via nvidia-ml-py or nvidia-smi
        if let Ok(output) = Command::new("nvidia-smi").arg("-L").output() {
            if output.status.success() {
                let gpu_lines = String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .filter(|line| line.contains("GPU"))
                    .count();
                return Ok(gpu_lines as i32);
            }
        }
        
        Err("Could not determine CUDA device count via nvidia-smi".to_string())
    }
    
    /// Get device info via nvidia-smi (for VastAI/Docker environments)
    #[cfg(feature = "cuda")]
    fn get_device_info_via_nvidia_smi(&self, device_id: i32) -> (String, u64, u32) {
        use std::process::Command;
        
        // Query device name and memory
        if let Ok(output) = Command::new("nvidia-smi")
            .args(&[
                "--query-gpu=name,memory.total", 
                "--format=csv,noheader,nounits",
                &format!("--id={}", device_id)
            ])
            .output() 
        {
            if output.status.success() {
                if let Ok(info) = String::from_utf8(output.stdout) {
                    let parts: Vec<&str> = info.trim().split(',').collect();
                    if parts.len() >= 2 {
                        let name = parts[0].trim().to_string();
                        let memory = parts[1].trim().parse::<u64>()
                            .unwrap_or(8192) * 1024 * 1024; // Convert MB to bytes
                        
                        // Estimate compute units based on GPU name
                        let compute_units = self.estimate_compute_units(&name);
                        
                        return (name, memory, compute_units);
                    }
                }
            }
        }
        
        // Fallback to generic naming
        (
            format!("CUDA Device {}", device_id),
            8 * 1024 * 1024 * 1024, // 8GB default
            80,                      // Default compute units
        )
    }

    /// Get comprehensive device information (tries CUDA API first, then nvidia-smi)
    #[cfg(feature = "cuda")]
    fn get_device_info(&self, device_id: i32) -> (String, u64, u32) {
        // Try CUDA API first if kernels are compiled
        #[cfg(cuda_available)]
        {
            use std::mem;
            let mut properties: CudaDeviceProperties = unsafe { mem::zeroed() };
            
            unsafe {
                if cuda_get_device_properties(&mut properties, device_id) == 0 {
                    // Extract device name from C string
                    let mut name_bytes = Vec::new();
                    for &byte in properties.name.iter() {
                        if byte == 0 { break; }
                        name_bytes.push(byte);
                    }
                    
                    let device_name = String::from_utf8(name_bytes)
                        .unwrap_or_else(|_| format!("CUDA Device {}", device_id));
                    
                    let memory = properties.total_global_mem as u64;
                    let compute_units = properties.multiprocessor_count as u32;
                    
                    return (device_name, memory, compute_units);
                }
            }
        }
        
        // Fallback to nvidia-smi
        self.get_device_info_via_nvidia_smi(device_id)
    }

    #[cfg(not(feature = "cuda"))]
    fn get_device_info(&self, device_id: i32) -> (String, u64, u32) {
        (format!("CUDA Device {}", device_id), 8 * 1024 * 1024 * 1024, 80)
    }

    /// Estimate compute units (SMs) based on GPU name
    fn estimate_compute_units(&self, gpu_name: &str) -> u32 {
        let gpu_name_lower = gpu_name.to_lowercase();
        
        // RTX 40XX series (Ada Lovelace)
        if gpu_name_lower.contains("rtx 4090") { return 128; }
        if gpu_name_lower.contains("rtx 4080") { return 76; }
        if gpu_name_lower.contains("rtx 4070") { return 46; }
        if gpu_name_lower.contains("rtx 4060") { return 24; }
        
        // RTX 30XX series (Ampere)
        if gpu_name_lower.contains("rtx 3090") { return 82; }
        if gpu_name_lower.contains("rtx 3080") { return 68; }
        if gpu_name_lower.contains("rtx 3070") { return 46; }
        if gpu_name_lower.contains("rtx 3060 ti") { return 38; }
        if gpu_name_lower.contains("rtx 3060") { return 28; }
        
        // RTX 20XX series (Turing)
        if gpu_name_lower.contains("rtx 2080 ti") { return 68; }
        if gpu_name_lower.contains("rtx 2080") { return 46; }
        if gpu_name_lower.contains("rtx 2070") { return 36; }
        if gpu_name_lower.contains("rtx 2060") { return 30; }
        
        // Tesla/Data Center cards
        if gpu_name_lower.contains("v100") { return 80; }
        if gpu_name_lower.contains("a100") { return 108; }
        if gpu_name_lower.contains("a40") { return 84; }
        if gpu_name_lower.contains("a30") { return 56; }
        if gpu_name_lower.contains("a10") { return 36; }
        if gpu_name_lower.contains("t4") { return 40; }
        
        // GTX 16XX series (Turing)
        if gpu_name_lower.contains("gtx 1660") { return 22; }
        if gpu_name_lower.contains("gtx 1650") { return 14; }
        
        // GTX 10XX series (Pascal)
        if gpu_name_lower.contains("gtx 1080 ti") { return 28; }
        if gpu_name_lower.contains("gtx 1080") { return 20; }
        if gpu_name_lower.contains("gtx 1070") { return 15; }
        if gpu_name_lower.contains("gtx 1060") { return 10; }
        
        // Default fallback
        40
    }
    
    /// Execute batch processing in fallback mode (CUDA runtime available but kernels not compiled)
    #[cfg(feature = "cuda")]
    fn execute_batch_fallback_mode(
        &self,
        _device_id: u32,
        start_offset: u128,
        batch_size: u128,
        target_address: &str,
        derivation_path: &str,
        passphrase: &str,
    ) -> Result<GpuBatchResult, GpuError> {
        // In fallback mode, we do CPU processing with GPU device context
        // This maintains compatibility when CUDA is detected but kernels aren't available
        
        println!("Using fallback CPU processing on GPU device context...");
        
        let word_space = self.word_space.as_ref().ok_or_else(|| {
            GpuError::DeviceInitFailed {
                device_id: _device_id,
                device_name: format!("CUDA Device {}", _device_id),
                error: "Word space not initialized".to_string(),
                timestamp: current_timestamp(),
            }
        })?;

        // Parse target address
        let target_address_cleaned = target_address.trim_start_matches("0x");
        let target_address_bytes = hex::decode(target_address_cleaned)
            .map_err(|_| GpuError::DeviceInitFailed {
                device_id: _device_id,
                device_name: format!("CUDA Device {}", _device_id),
                error: "Invalid target address format".to_string(),
                timestamp: current_timestamp(),
            })?;

        if target_address_bytes.len() != 20 {
            return Err(GpuError::DeviceInitFailed {
                device_id: _device_id,
                device_name: format!("CUDA Device {}", _device_id),
                error: "Target address must be 20 bytes".to_string(),
                timestamp: current_timestamp(),
            });
        }

        let mut target_address_array = [0u8; 20];
        target_address_array.copy_from_slice(&target_address_bytes);

        // Process small batches in CPU mode
        let fallback_batch_size = std::cmp::min(batch_size, 1000) as usize;
        
        for batch_start in (0..fallback_batch_size).step_by(100) {
            let batch_end = std::cmp::min(batch_start + 100, fallback_batch_size);
            
            for i in batch_start..batch_end {
                let current_offset = start_offset + i as u128;
                
                if let Some(word_indices) = word_space.index_to_words(current_offset) {
                    if let Some(mnemonic) = WordSpace::words_to_mnemonic(&word_indices) {
                        let address = derive_ethereum_address(&mnemonic, passphrase, derivation_path)
                            .map_err(|e| GpuError::KernelExecutionFailed {
                                device_id: _device_id,
                                kernel_name: "fallback_cpu".to_string(),
                                error: format!("Address derivation failed: {}", e),
                                timestamp: current_timestamp(),
                            })?;

                        let address_hex = format!("0x{}", hex::encode(address));
                        if addresses_equal(&address_hex, target_address) {
                            return Ok(GpuBatchResult {
                                mnemonic: Some(mnemonic),
                                address: Some(address_hex),
                                offset: Some(current_offset),
                                processed_count: (i + 1) as u128,
                            });
                        }
                    }
                }
            }
        }

        Ok(GpuBatchResult {
            mnemonic: None,
            address: None,
            offset: None,
            processed_count: fallback_batch_size as u128,
        })
    }
    
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

        // Perform comprehensive health check before execution
        if let Err(error) = self.check_device_health(device_id) {
            self.mark_device_failed(device_id, error.clone());
            return Err(error);
        }

        // Validate batch size to prevent memory issues
        let max_safe_batch_size = self.calculate_max_safe_batch_size(device_id);
        let actual_batch_size = batch_size.min(max_safe_batch_size);
        
        if actual_batch_size != batch_size {
            println!("Warning: Reducing batch size from {} to {} for device {} to prevent memory overflow", 
                     batch_size, actual_batch_size, device_id);
        }

        // Estimate memory requirements with safety margin
        let estimated_memory = self.estimate_memory_for_batch(actual_batch_size);
        let available_memory = self.get_device_memory(device_id);
        
        if estimated_memory > (available_memory * 80 / 100) { // Use only 80% of available memory
            let error = GpuError::OutOfMemory {
                device_id,
                batch_size: actual_batch_size,
                available_memory,
                timestamp: current_timestamp(),
            };
            self.error_logger.log_error(&error);
            return Err(error);
        }

        // Execute with retry mechanism
        let mut last_error = None;
        for attempt in 1..=3 {
            match self.execute_batch_internal(
                device_id, start_offset, actual_batch_size, target_address, derivation_path, passphrase
            ) {
                Ok(result) => {
                    let duration = start_time.elapsed().as_millis() as u64;
                    
                    // Update device status on success
                    if let Ok(mut status_vec) = self.device_status.lock() {
                        if let Some(status) = status_vec.iter_mut().find(|s| s.device_id == device_id) {
                            status.increment_batch_count();
                            status.mark_healthy();
                        }
                    }
                    self.error_logger.log_batch_result(device_id, actual_batch_size, true, duration);
                    return Ok(result);
                }
                Err(error) => {
                    last_error = Some(error);
                    println!("Device {} batch attempt {} failed, retrying...", device_id, attempt);
                    
                    // Wait a bit before retry
                    std::thread::sleep(std::time::Duration::from_millis(100 * attempt));
                    
                    // Reset device before retry
                    #[cfg(feature = "cuda")]
                    {
                        use crate::error_handling::cuda_errors::reset_device;
                        if let Err(reset_error) = reset_device(device_id) {
                            println!("Warning: Failed to reset device {} before retry: {:?}", device_id, reset_error);
                        }
                    }
                }
            }
        }

        let duration = start_time.elapsed().as_millis() as u64;
        self.error_logger.log_batch_result(device_id, actual_batch_size, false, duration);
        
        // Convert the last error to GpuError for proper handling
        let gpu_error = if let Some(error) = last_error {
            GpuError::KernelExecutionFailed {
                device_id,
                kernel_name: "batch_execution".to_string(),
                error: error.to_string(),
                timestamp: current_timestamp(),
            }
        } else {
            GpuError::DeviceHardwareFailure {
                device_id,
                device_name: format!("CUDA Device {}", device_id),
                error_code: -1,
                timestamp: current_timestamp(),
            }
        };
        
        self.mark_device_failed(device_id, gpu_error.clone());
        Err(gpu_error)
    }

    /// Calculate maximum safe batch size for device
    fn calculate_max_safe_batch_size(&self, device_id: u32) -> u128 {
        let available_memory = self.get_device_memory(device_id);
        let memory_per_item = 512; // Estimated memory per mnemonic processing (conservative)
        let safety_factor = 0.7; // Use only 70% of available memory for safety
        
        ((available_memory as f64 * safety_factor) / memory_per_item as f64) as u128
    }

    /// Estimate memory requirements for batch
    fn estimate_memory_for_batch(&self, batch_size: u128) -> u64 {
        let base_memory_per_item = 512; // Base memory per item (strings, intermediate results)
        let additional_overhead = batch_size as u64 * 64; // Additional GPU kernel overhead
        
        (batch_size as u64 * base_memory_per_item) + additional_overhead + (100 * 1024 * 1024) // 100MB overhead
    }

    /// Get device memory for a CUDA device
    fn get_device_memory(&self, device_id: u32) -> u64 {
        #[cfg(all(feature = "cuda", cuda_available))]
        {
            use std::mem;
            let mut properties: CudaDeviceProperties = unsafe { mem::zeroed() };
            unsafe {
                if cuda_get_device_properties(&mut properties, device_id as i32) == 0 {
                    return properties.total_global_mem as u64;
                }
            }
        }
        
        // Fallback: get via nvidia-smi
        #[cfg(feature = "cuda")]
        {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(&[
                    "--query-gpu=memory.total", 
                    "--format=csv,noheader,nounits",
                    &format!("--id={}", device_id)
                ])
                .output() 
            {
                if output.status.success() {
                    if let Ok(memory_str) = String::from_utf8(output.stdout) {
                        if let Ok(memory_mb) = memory_str.trim().parse::<u64>() {
                            return memory_mb * 1024 * 1024; // Convert MB to bytes
                        }
                    }
                }
            }
        }
        
        // Conservative fallback
        8 * 1024 * 1024 * 1024 // 8GB
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
            // For VastAI/Docker environments, CUDA runtime may be available even if kernels weren't compiled
            println!("CUDA kernels not compiled, but checking runtime availability...");
            if self.is_cuda_available() {
                println!("CUDA runtime detected via nvidia-smi, but kernels not available.");
                println!("Note: GPU operations will use simplified processing without optimized kernels.");
                return self.execute_batch_fallback_mode(0, start_offset, batch_size, target_address, _derivation_path, passphrase)
                    .map_err(|e| Box::new(e) as Box<dyn Error>);
            } else {
                return Err("CUDA runtime not available. Install NVIDIA drivers and CUDA toolkit.".into());
            }
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
                
                // Initialize device status tracking with real device names
                let mut status_vec = Vec::new();
                for i in 0..count {
                    let (device_name, memory, compute_units) = self.get_device_info(i);
                    println!("  Device {}: {} ({} MB memory, {} compute units)", 
                             i, device_name, memory / (1024 * 1024), compute_units);
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

        use bip39_solver_gpu::gpu_models::SUPPORTED_GPU_MODELS;
        for i in 0..device_count {
            let (device_name, memory, compute_units) = self.get_device_info(i);
            // Try to match against known models
            let matched_name = SUPPORTED_GPU_MODELS.iter()
                .find(|model| device_name.replace(" ", "").to_lowercase().contains(&model.replace(" ", "").to_lowercase()))
                .map(|model| model.to_string())
                .unwrap_or(device_name.clone());
            devices.push(GpuDevice {
                id: i as u32,
                name: matched_name,
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
