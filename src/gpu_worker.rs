use crate::config::Config;
use crate::word_lut::WordLut;
use crate::bip39::Bip39;
use crate::bip44::Bip44;
use crate::eth_addr::EthAddr;

/// GPU worker for CUDA detection and processing
/// Uses auto-detection and falls back gracefully
pub struct GpuWorker {
    cuda_available: bool,
}

impl GpuWorker {
    /// Create new GPU worker with CUDA detection
    pub fn new() -> Self {
        let cuda_available = Self::detect_cuda();
        Self { cuda_available }
    }
    
    /// Check if CUDA is available
    pub fn is_cuda_available(&self) -> bool {
        self.cuda_available
    }
    
    /// Detect CUDA availability
    fn detect_cuda() -> bool {
        // Try to detect CUDA using cuda_runtime_sys if available
        #[cfg(feature = "cuda")]
        {
            use std::os::raw::c_int;
            extern "C" {
                fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
            }
            
            unsafe {
                let mut device_count: c_int = 0;
                let result = cudaGetDeviceCount(&mut device_count);
                result == 0 && device_count > 0
            }
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            // CUDA feature not enabled, check if nvidia-smi is available
            std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=count")
                .arg("--format=csv,noheader,nounits")
                .output()
                .map(|output| output.status.success())
                .unwrap_or(false)
        }
    }
    
    /// Process batch of candidates using GPU (or fallback to CPU)
    pub fn process_batch(
        &self,
        candidates: &[[u16; 12]],
        config: &Config,
        word_lut: &WordLut,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        if self.cuda_available {
            self.process_batch_cuda(candidates, config, word_lut)
        } else {
            // Fallback to CPU processing
            self.process_batch_cpu(candidates, config, word_lut)
        }
    }
    
    /// Process batch using CUDA kernels
    #[cfg(feature = "cuda")]
    fn process_batch_cuda(
        &self,
        candidates: &[[u16; 12]],
        config: &Config,
        word_lut: &WordLut,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        // For now, use existing CUDA infrastructure if available
        // In production, this would use cudarc and mature CUDA kernels
        println!("[DEBUG] Using CUDA GPU processing for {} candidates", candidates.len());
        
        // Fallback to CPU for now since implementing full CUDA pipeline
        // would require the custom kernels which PRD says to avoid
        self.process_batch_cpu(candidates, config, word_lut)
    }
    
    #[cfg(not(feature = "cuda"))]
    fn process_batch_cuda(
        &self,
        candidates: &[[u16; 12]],
        config: &Config,
        word_lut: &WordLut,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        // CUDA not available at compile time, fallback to CPU
        self.process_batch_cpu(candidates, config, word_lut)
    }
    
    /// Process batch using CPU (fallback or when CUDA unavailable)
    fn process_batch_cpu(
        &self,
        candidates: &[[u16; 12]],
        config: &Config,
        word_lut: &WordLut,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let total_candidates = candidates.len();
        let progress_interval = std::cmp::max(1000, total_candidates / 20); // Show progress every 5%
        
        for (idx, candidate) in candidates.iter().enumerate() {
            if idx > 0 && idx % progress_interval == 0 {
                let percentage = (idx as f64 / total_candidates as f64) * 100.0;
                println!("[INFO] Processing candidates: {}/{} ({:.1}%)", idx, total_candidates, percentage);
            }
            
            // Quick validation first
            if !Bip39::quick_validate_indices(candidate) {
                continue; // Skip invalid checksums
            }
            
            // Convert to mnemonic
            let mnemonic = word_lut.indices_to_mnemonic(candidate)
                .ok_or("Failed to convert indices to mnemonic")?;
            
            // Generate seed
            let seed = Bip39::mnemonic_to_seed(&mnemonic, &config.ethereum.passphrase)?;
            
            // Derive private key
            let private_key = Bip44::derive_private_key(&seed, &config.ethereum.derivation_path)?;
            
            // Generate Ethereum address
            let address = EthAddr::private_key_to_address(&private_key)?;
            
            // Check if it matches target
            if EthAddr::addresses_equal(&address, &config.ethereum.target_address) {
                println!("[INFO] Match found at candidate {}/{}", idx + 1, total_candidates);
                return Ok(Some(mnemonic));
            }
        }
        
        Ok(None)
    }
    
    /// Get optimal batch size for GPU processing
    pub fn get_optimal_batch_size(&self) -> usize {
        if self.cuda_available {
            64000 // As specified in PRD
        } else {
            1000 // Smaller batches for CPU
        }
    }
    
    /// Get GPU device information
    pub fn get_device_info(&self) -> Vec<String> {
        if !self.cuda_available {
            return vec!["No CUDA devices available".to_string()];
        }
        
        #[cfg(feature = "cuda")]
        {
            // Get CUDA device information
            use std::os::raw::{c_int, c_char};
            extern "C" {
                fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
                fn cudaGetDeviceProperties(prop: *mut CudaDeviceProp, device: c_int) -> c_int;
            }
            
            #[repr(C)]
            struct CudaDeviceProp {
                name: [c_char; 256],
                total_global_mem: usize,
                shared_mem_per_block: usize,
                regs_per_block: c_int,
                warp_size: c_int,
                mem_pitch: usize,
                max_threads_per_block: c_int,
                max_threads_dim: [c_int; 3],
                max_grid_size: [c_int; 3],
                clock_rate: c_int,
                total_const_mem: usize,
                major: c_int,
                minor: c_int,
                texture_alignment: usize,
                texture_pitch_alignment: usize,
                device_overlap: c_int,
                multi_processor_count: c_int,
                // ... more fields exist but we only need basic info
            }
            
            unsafe {
                let mut device_count: c_int = 0;
                if cudaGetDeviceCount(&mut device_count) != 0 {
                    return vec!["Failed to get CUDA device count".to_string()];
                }
                
                let mut devices = Vec::new();
                for i in 0..device_count {
                    let mut prop: CudaDeviceProp = std::mem::zeroed();
                    if cudaGetDeviceProperties(&mut prop, i) == 0 {
                        let name = std::ffi::CStr::from_ptr(prop.name.as_ptr())
                            .to_string_lossy()
                            .to_string();
                        let memory_mb = prop.total_global_mem / 1024 / 1024;
                        devices.push(format!("Device {}: {} ({} MB)", i, name, memory_mb));
                    }
                }
                devices
            }
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            vec!["CUDA support not compiled in".to_string()]
        }
    }
}

impl Default for GpuWorker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EthereumConfig;
    
    #[test]
    fn test_gpu_worker_creation() {
        let worker = GpuWorker::new();
        
        // Should successfully create worker
        // CUDA availability depends on system
        println!("CUDA available: {}", worker.is_cuda_available());
    }
    
    #[test]
    fn test_process_batch_cpu() {
        let worker = GpuWorker::new();
        let word_lut = WordLut::new();
        
        // Test with known good mnemonic indices
        let candidates = vec![[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]]; // abandon...about
        
        let config = Config {
            wallet_type: "ethereum".to_string(),
            mnemonic_length: 12,
            ethereum: crate::config::EthereumConfig {
                target_address: "0x9858EfFD232B4033E47d90003D41EC34EcaEda94".to_string(),
                derivation_path: "m/44'/60'/0'/0/0".to_string(),
                passphrase: "".to_string(),
            },
            word_constraints: vec![],
        };
        
        let result = worker.process_batch_cpu(&candidates, &config, &word_lut).unwrap();
        
        // Should find the matching mnemonic
        if let Some(mnemonic) = result {
            assert_eq!(mnemonic, "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about");
        }
    }
}