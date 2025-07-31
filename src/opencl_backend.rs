use crate::gpu_backend::{GpuBackend, GpuBatchResult, GpuDevice};
use crate::word_space::WordSpace;
use crate::eth::{derive_ethereum_address, addresses_equal};
use std::error::Error;
use std::sync::Arc;

/// OpenCL backend implementation for GPU computation
/// Currently falls back to CPU processing with OpenCL device enumeration
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
        
        // Try to check for OpenCL availability
        // For now, we'll always succeed and fall back to CPU
        println!("OpenCL backend initialized (CPU fallback mode)");
        
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
        let mut devices = Vec::new();
        
        // Try to enumerate OpenCL devices, but fall back to CPU if failed
        // For now, create a CPU fallback device
        devices.push(GpuDevice {
            id: 0,
            name: "CPU Fallback (OpenCL)".to_string(),
            memory: 8 * 1024 * 1024 * 1024, // 8GB default
            compute_units: std::thread::available_parallelism()
                .map(|p| p.get() as u32)
                .unwrap_or(4),
        });
        
        Ok(devices)
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
        // For now, fall back to CPU processing
        // TODO: Implement actual OpenCL kernel execution
        
        let word_space = self.word_space.as_ref()
            .ok_or("Word space not initialized")?;
        
        let mut processed_count = 0u128;
        
        // Process batch using CPU (fallback implementation)
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
        // OpenCL is always "available" since we fall back to CPU
        true
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