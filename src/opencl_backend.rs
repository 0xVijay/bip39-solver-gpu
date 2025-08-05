use crate::gpu_backend::{GpuBackend, GpuBatchResult, GpuDevice};
use crate::word_space::WordSpace;
use std::error::Error;
use std::sync::Arc;

/// OpenCL backend implementation for GPU computation
pub struct OpenClBackend {
    initialized: bool,
    word_space: Option<Arc<WordSpace>>,
}

impl OpenClBackend {
    /// Create a new OpenCL backend instance
    pub fn new() -> Self {
        OpenClBackend {
            initialized: false,
            word_space: None,
        }
    }

    /// Set the word space for mnemonic generation
    pub fn set_word_space(&mut self, word_space: Arc<WordSpace>) {
        self.word_space = Some(word_space);
    }

    /// Execute OpenCL kernel for GPU computation
    #[cfg(feature = "opencl")]
    fn execute_opencl_kernel(
        &self,
        _device_id: u32,
        start_offset: u128,
        batch_size: u128,
        target_address: &str,
    ) -> Result<GpuBatchResult, Box<dyn Error>> {
        use opencl3::context::Context;
        use opencl3::device::{get_all_devices, CL_DEVICE_TYPE_GPU, Device};
        use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
        use opencl3::program::Program;
        use opencl3::kernel::{ExecuteKernel, Kernel};
        use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
        use opencl3::types::{cl_uchar, cl_ulong, cl_bool};
        use std::path::PathBuf;

        const CL_BLOCKING: cl_bool = 1;

        // Get GPU devices
        let devices = get_all_devices(CL_DEVICE_TYPE_GPU)?;
        if devices.is_empty() {
            return Err("No OpenCL GPU devices found".into());
        }

        let device_id = devices[0];
        let device = Device::new(device_id);
        let context = Context::from_device(&device)?;

        // Create command queue
        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)?;

        // Load OpenCL kernel source files
        let kernel_dir = PathBuf::from("cl");
        let mut kernel_source = String::new();

        // Load required kernel files in order
        let kernel_files = [
            "common.cl",
            "sha2.cl", 
            "keccak.cl",
            "secp256k1_common.cl",
            "secp256k1_field.cl",
            "secp256k1_scalar.cl",
            "secp256k1_group.cl",
            "secp256k1.cl",
            "mnemonic_constants.cl",
            "eth_address.cl",
        ];

        for filename in &kernel_files {
            let file_path = kernel_dir.join(filename);
            if file_path.exists() {
                let file_content = std::fs::read_to_string(&file_path)
                    .map_err(|e| format!("Failed to read {}: {}", filename, e))?;
                kernel_source.push_str(&file_content);
                kernel_source.push('\n');
            }
        }

        if kernel_source.is_empty() {
            return Err("No OpenCL kernel source files found in cl/ directory".into());
        }

        // Create program and build
        let program = Program::create_and_build_from_source(&context, &kernel_source, "")?;
        
        // Create kernel
        let kernel = Kernel::create(&program, "mnemonic_to_eth_address")?;

        // Parse target address from hex string
        let target_bytes = if target_address.starts_with("0x") {
            hex::decode(&target_address[2..])?
        } else {
            hex::decode(target_address)?
        };

        if target_bytes.len() != 20 {
            return Err(format!("Invalid Ethereum address length: {} bytes", target_bytes.len()).into());
        }

        // Convert start_offset to hi/lo format (128-bit to 64-bit hi/lo)
        let mnemonic_start_hi = (start_offset >> 64) as u64;
        let mnemonic_start_lo = start_offset as u64;

        // Create buffers
        let batch_size_u32 = batch_size.min(1_000_000) as usize; // Limit batch size for memory
        
        let mut target_buffer = unsafe { 
            Buffer::<cl_uchar>::create(&context, CL_MEM_READ_ONLY, 20, std::ptr::null_mut())?
        };
        let mut found_mnemonic_buffer = unsafe {
            Buffer::<cl_uchar>::create(&context, CL_MEM_READ_WRITE, 256, std::ptr::null_mut())?
        };
        let mut derivation_buffer = unsafe {
            Buffer::<cl_uchar>::create(&context, CL_MEM_READ_ONLY, 64, std::ptr::null_mut())?
        };

        // Write target address to buffer
        let _write_event = unsafe {
            queue.enqueue_write_buffer(&mut target_buffer, CL_BLOCKING, 0, &target_bytes, &[])?
        };

        // Write derivation path (m/44'/60'/0'/0/0)
        let derivation_path = b"m/44'/60'/0'/0/0\0";
        let _write_event2 = unsafe {
            queue.enqueue_write_buffer(&mut derivation_buffer, CL_BLOCKING, 0, derivation_path, &[])?
        };

        // Initialize found_mnemonic buffer to zero
        let zero_buffer = vec![0u8; 256];
        let _write_event3 = unsafe {
            queue.enqueue_write_buffer(&mut found_mnemonic_buffer, CL_BLOCKING, 0, &zero_buffer, &[])?
        };

        // Set kernel arguments and execute
        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&(mnemonic_start_hi as cl_ulong))
                .set_arg(&(mnemonic_start_lo as cl_ulong))
                .set_arg(&target_buffer)
                .set_arg(&found_mnemonic_buffer)
                .set_arg(&derivation_buffer)
                .set_global_work_size(batch_size_u32)
                .enqueue_nd_range(&queue)?;
        }

        // Wait for completion
        queue.finish()?;

        // Read results
        let mut found_data = vec![0u8; 256];
        let _read_event = unsafe {
            queue.enqueue_read_buffer(&mut found_mnemonic_buffer, CL_BLOCKING, 0, &mut found_data, &[])?
        };

        // Check if we found a match
        if found_data[0] == 0x01 {
            // Extract mnemonic string (skip first status byte)
            let mnemonic_bytes = &found_data[1..];
            let mnemonic_len = mnemonic_bytes.iter().position(|&x| x == 0).unwrap_or(255);
            let mnemonic = String::from_utf8_lossy(&mnemonic_bytes[..mnemonic_len]).to_string();
            
            println!("🎯 Found matching mnemonic on GPU: {}", mnemonic);
            
            return Ok(GpuBatchResult {
                mnemonic: Some(mnemonic),
                address: Some(target_address.to_string()),
                offset: Some(start_offset),
                processed_count: batch_size,
            });
        }

        // No match found
        Ok(GpuBatchResult {
            mnemonic: None,
            address: None,
            offset: None,
            processed_count: batch_size,
        })
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
            use opencl3::device::{get_all_devices, CL_DEVICE_TYPE_GPU, Device};

            let gpu_devices = get_all_devices(CL_DEVICE_TYPE_GPU)?;
            let mut devices = Vec::new();

            for (id, device_id) in gpu_devices.iter().enumerate() {
                let device = Device::new(*device_id);
                // Get actual device name
                let device_name = match device.name() {
                    Ok(name) => name,
                    Err(_) => format!("OpenCL GPU Device {}", id),
                };
                
                // Get actual memory size
                let memory = match device.global_mem_size() {
                    Ok(mem) => mem as u64,
                    Err(_) => 4 * 1024 * 1024 * 1024, // 4GB default
                };
                // Get actual compute units
                let compute_units = match device.max_compute_units() {
                    Ok(units) => units as u32,
                    Err(_) => 16, // Default
                };
                devices.push(GpuDevice {
                    id: id as u32,
                    name: device_name,
                    memory,
                    compute_units,
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

        // Try to initialize OpenCL - fail if not available (no CPU fallback)
        self.try_initialize_opencl()?;
        println!("OpenCL backend initialized successfully");

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
        // Only enumerate actual OpenCL GPU devices (no CPU fallback)
        let devices = self.enumerate_opencl_devices()?;
        if devices.is_empty() {
            return Err("No OpenCL GPU devices found".into());
        }
        Ok(devices)
    }

    fn execute_batch(
        &self,
        device_id: u32,
        start_offset: u128,
        batch_size: u128,
        target_address: &str,
        _derivation_path: &str,
        _passphrase: &str,
    ) -> Result<GpuBatchResult, Box<dyn Error>> {
        // Use actual OpenCL GPU kernel processing
        #[cfg(feature = "opencl")]
        {
            self.execute_opencl_kernel(device_id, start_offset, batch_size, target_address)
        }
        
        #[cfg(not(feature = "opencl"))]
        {
            Err("OpenCL support not compiled".into())
        }
    }

    fn is_available(&self) -> bool {
        // Check if OpenCL is actually available (no CPU fallback)
        self.check_opencl_availability().unwrap_or(false)
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