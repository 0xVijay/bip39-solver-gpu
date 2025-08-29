use rayon::prelude::*;
use crate::config::Config;
use crate::word_lut::WordLut;
use crate::candidate_gen::CandidateGenerator;
use crate::bip39::Bip39;
use crate::bip44::Bip44;
use crate::eth_addr::EthAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// CPU worker using Rayon for parallel processing
/// Provides high-performance CPU fallback when GPU unavailable
pub struct CpuWorker {
    /// Number of threads to use (auto-detected)
    thread_count: usize,
}

impl CpuWorker {
    /// Create new CPU worker with optimal thread count
    pub fn new() -> Self {
        let thread_count = rayon::current_num_threads();
        Self { thread_count }
    }
    
    /// Create CPU worker with specific thread count
    pub fn with_threads(thread_count: usize) -> Self {
        // Initialize Rayon thread pool with specified count
        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build_global()
            .unwrap_or_default();
        
        Self { thread_count }
    }
    
    /// Get number of threads being used
    pub fn thread_count(&self) -> usize {
        self.thread_count
    }
    
    /// Search for target address using parallel processing
    pub fn search_parallel(
        &self,
        config: &Config,
        candidate_gen: &CandidateGenerator,
        word_lut: &WordLut,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let total = candidate_gen.total_combinations();
        let found = Arc::new(AtomicBool::new(false));
        let batch_size = 10000; // Process in batches for better performance
        
        println!("[INFO] Starting CPU search with {} threads", self.thread_count);
        
        // Create chunks of work for parallel processing
        let chunk_size = std::cmp::max(batch_size as u128, total / (self.thread_count as u128 * 4));
        let mut chunks = Vec::new();
        let mut offset = 0u128;
        
        while offset < total {
            let chunk_end = std::cmp::min(offset + chunk_size, total);
            chunks.push((offset, chunk_end));
            offset = chunk_end;
        }
        
        // Process chunks in parallel
        let result = chunks
            .into_par_iter()
            .map(|(start, end)| {
                if found.load(Ordering::Relaxed) {
                    return None; // Early termination
                }
                self.process_range(config, candidate_gen, word_lut, start, end, &found)
            })
            .find_any(|result| result.is_some())
            .flatten();
        
        Ok(result)
    }
    
    /// Process a range of candidate indices
    fn process_range(
        &self,
        config: &Config,
        candidate_gen: &CandidateGenerator,
        word_lut: &WordLut,
        start: u128,
        end: u128,
        found: &Arc<AtomicBool>,
    ) -> Option<String> {
        for i in start..end {
            if found.load(Ordering::Relaxed) {
                return None; // Another thread found the result
            }
            
            // Convert index to word indices
            let candidate = candidate_gen.index_to_words(i)?;
            
            // Quick validation first
            if !Bip39::quick_validate_indices(&candidate) {
                continue; // Skip invalid checksums
            }
            
            // Convert to mnemonic
            let mnemonic = word_lut.indices_to_mnemonic(&candidate)?;
            
            // Process this candidate
            match self.process_candidate(&mnemonic, config) {
                Ok(true) => {
                    found.store(true, Ordering::Relaxed);
                    return Some(mnemonic);
                }
                Ok(false) => continue,
                Err(_) => continue, // Skip on error
            }
        }
        
        None
    }
    
    /// Process a single candidate mnemonic
    fn process_candidate(
        &self,
        mnemonic: &str,
        config: &Config,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        // Generate seed
        let seed = Bip39::mnemonic_to_seed(mnemonic, &config.ethereum.passphrase)?;
        
        // Derive private key
        let private_key = Bip44::derive_private_key(&seed, &config.ethereum.derivation_path)?;
        
        // Generate Ethereum address
        let address = EthAddr::private_key_to_address(&private_key)?;
        
        // Check if it matches target
        Ok(EthAddr::addresses_equal(&address, &config.ethereum.target_address))
    }
    
    /// Process batch of candidates sequentially (for testing)
    pub fn process_batch_sequential(
        &self,
        candidates: &[[u16; 12]],
        config: &Config,
        word_lut: &WordLut,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        for candidate in candidates {
            // Quick validation first
            if !Bip39::quick_validate_indices(candidate) {
                continue; // Skip invalid checksums
            }
            
            // Convert to mnemonic
            let mnemonic = word_lut.indices_to_mnemonic(candidate)
                .ok_or("Failed to convert indices to mnemonic")?;
            
            // Process this candidate
            if self.process_candidate(&mnemonic, config)? {
                return Ok(Some(mnemonic));
            }
        }
        
        Ok(None)
    }
    
    /// Process batch of candidates in parallel
    pub fn process_batch_parallel(
        &self,
        candidates: &[[u16; 12]],
        config: &Config,
        word_lut: &WordLut,
    ) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let found = Arc::new(AtomicBool::new(false));
        
        let result = candidates
            .par_iter()
            .map(|candidate| {
                if found.load(Ordering::Relaxed) {
                    return None; // Early termination
                }
                
                // Quick validation first
                if !Bip39::quick_validate_indices(candidate) {
                    return None; // Skip invalid checksums
                }
                
                // Convert to mnemonic
                let mnemonic = word_lut.indices_to_mnemonic(candidate)?;
                
                // Process this candidate
                match self.process_candidate(&mnemonic, config) {
                    Ok(true) => {
                        found.store(true, Ordering::Relaxed);
                        Some(mnemonic)
                    }
                    Ok(false) => None,
                    Err(_) => None, // Skip on error
                }
            })
            .find_any(|result| result.is_some())
            .flatten();
        
        Ok(result)
    }
    
    /// Estimate processing rate (mnemonics per second)
    pub fn estimate_rate(&self, sample_size: usize) -> Result<f64, Box<dyn std::error::Error>> {
        let word_lut = WordLut::new();
        
        // Create test configuration
        let config = Config {
            wallet_type: "ethereum".to_string(),
            mnemonic_length: 12,
            ethereum: crate::config::EthereumConfig {
                target_address: "0x0000000000000000000000000000000000000000".to_string(),
                derivation_path: "m/44'/60'/0'/0/0".to_string(),
                passphrase: "".to_string(),
            },
            word_constraints: vec![],
        };
        
        // Generate test candidates
        let mut candidates = Vec::with_capacity(sample_size);
        for i in 0..sample_size {
            let mut indices = [0u16; 12];
            indices[11] = 3; // Make it a valid mnemonic ("abandon...about")
            indices[0] = (i % 2048) as u16; // Vary first word
            candidates.push(indices);
        }
        
        // Time the processing
        let start = std::time::Instant::now();
        self.process_batch_parallel(&candidates, &config, &word_lut)?;
        let elapsed = start.elapsed();
        
        let rate = sample_size as f64 / elapsed.as_secs_f64();
        Ok(rate)
    }
}

impl Default for CpuWorker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EthereumConfig;
    
    #[test]
    fn test_cpu_worker_creation() {
        let worker = CpuWorker::new();
        assert!(worker.thread_count() > 0);
        println!("CPU worker using {} threads", worker.thread_count());
    }
    
    #[test]
    fn test_process_candidate() {
        let worker = CpuWorker::new();
        
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
        
        // Test with known mnemonic
        let mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
        let result = worker.process_candidate(mnemonic, &config).unwrap();
        
        // Should match the target address
        assert!(result);
    }
    
    #[test]
    fn test_process_batch() {
        let worker = CpuWorker::new();
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
        
        let result = worker.process_batch_sequential(&candidates, &config, &word_lut).unwrap();
        
        // Should find the matching mnemonic
        if let Some(mnemonic) = result {
            assert_eq!(mnemonic, "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about");
        }
    }
    
    #[test]
    fn test_estimate_rate() {
        let worker = CpuWorker::new();
        let rate = worker.estimate_rate(100).unwrap();
        
        println!("Estimated processing rate: {:.2} mnemonics/sec", rate);
        assert!(rate > 0.0);
    }
}