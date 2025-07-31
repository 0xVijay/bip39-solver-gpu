use std::error::Error;

/// Result type returned by GPU backends for batch execution
#[derive(Debug, Clone)]
pub struct GpuBatchResult {
    /// The mnemonic that produced a match (if any)
    pub mnemonic: Option<String>,
    /// The Ethereum address that was found (if any)  
    pub address: Option<String>,
    /// The offset/index in the search space where the match was found
    pub offset: Option<u128>,
    /// Number of mnemonics processed in this batch
    pub processed_count: u128,
}

/// Information about a GPU device
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Device ID/index
    pub id: u32,
    /// Device name
    pub name: String,
    /// Available memory in bytes
    pub memory: u64,
    /// Maximum compute units
    pub compute_units: u32,
}

/// GPU backend trait for modular GPU computation support
pub trait GpuBackend: Send + Sync {
    /// Get the name of this backend (e.g., "OpenCL", "CUDA")
    fn backend_name(&self) -> &'static str;

    /// Initialize the GPU backend
    fn initialize(&mut self) -> Result<(), Box<dyn Error>>;

    /// Shutdown the GPU backend and cleanup resources
    fn shutdown(&mut self) -> Result<(), Box<dyn Error>>;

    /// Enumerate available GPU devices
    fn enumerate_devices(&self) -> Result<Vec<GpuDevice>, Box<dyn Error>>;

    /// Execute a batch of mnemonic candidates on the specified device
    ///
    /// # Arguments
    /// * `device_id` - The GPU device to use
    /// * `start_offset` - Starting offset in the search space
    /// * `batch_size` - Number of candidates to process
    /// * `target_address` - The Ethereum address to search for
    /// * `derivation_path` - BIP44 derivation path
    /// * `passphrase` - BIP39 passphrase
    ///
    /// # Returns
    /// Result containing batch execution results
    fn execute_batch(
        &self,
        device_id: u32,
        start_offset: u128,
        batch_size: u128,
        target_address: &str,
        derivation_path: &str,
        passphrase: &str,
    ) -> Result<GpuBatchResult, Box<dyn Error>>;

    /// Check if the backend is available on this system
    fn is_available(&self) -> bool;
}
