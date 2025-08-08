/// GPU memory management utilities for optimal batch sizing
use crate::gpu_backend::GpuDevice;

/// Memory requirements per mnemonic processing (in bytes)
const MEMORY_PER_MNEMONIC: usize = 256; // Estimated for all crypto operations

/// Maximum GPU memory utilization percentage
const MAX_GPU_MEMORY_UTILIZATION: f32 = 0.80; // 80%

/// Minimum batch size to ensure efficiency
const MIN_BATCH_SIZE: u128 = 1000;

/// Maximum batch size to prevent timeout issues
const MAX_BATCH_SIZE: u128 = 1_000_000;

/// Calculate optimal batch size based on GPU device memory
pub fn calculate_optimal_batch_size(device: &GpuDevice) -> u128 {
    // Calculate available memory (80% of total)
    let available_memory = (device.memory as f32 * MAX_GPU_MEMORY_UTILIZATION) as usize;
    
    // Calculate batch size based on memory per mnemonic
    let memory_based_batch = (available_memory / MEMORY_PER_MNEMONIC) as u128;
    
    // Apply min/max constraints
    let optimal_batch = memory_based_batch
        .max(MIN_BATCH_SIZE)
        .min(MAX_BATCH_SIZE);
    
    println!(
        "Device {}: {} MB memory → {} optimal batch size", 
        device.id, 
        device.memory / (1024 * 1024), 
        optimal_batch
    );
    
    optimal_batch
}

/// Get memory info for debugging
pub fn get_memory_info(device: &GpuDevice, batch_size: u128) -> String {
    let memory_usage = (batch_size as usize * MEMORY_PER_MNEMONIC) as u64;
    let memory_usage_mb = memory_usage / (1024 * 1024);
    let total_memory_mb = device.memory / (1024 * 1024);
    let utilization = (memory_usage as f32 / device.memory as f32) * 100.0;
    
    format!(
        "Memory: {}/{} MB ({:.1}% utilization)", 
        memory_usage_mb, 
        total_memory_mb, 
        utilization
    )
}

/// Validate batch size is within device limits
pub fn validate_batch_size(device: &GpuDevice, batch_size: u128) -> Result<(), String> {
    let memory_needed = (batch_size as usize * MEMORY_PER_MNEMONIC) as u64;
    let max_memory = (device.memory as f32 * MAX_GPU_MEMORY_UTILIZATION) as u64;
    
    if memory_needed > max_memory {
        return Err(format!(
            "Batch size {} requires {} MB but device {} only has {} MB available",
            batch_size,
            memory_needed / (1024 * 1024),
            device.id,
            max_memory / (1024 * 1024)
        ));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_size_calculation() {
        let device = GpuDevice {
            id: 0,
            name: "Test GPU".to_string(),
            memory: 8 * 1024 * 1024 * 1024, // 8 GB
            compute_units: 80,
        };
        
        let batch_size = calculate_optimal_batch_size(&device);
        
        // Should be between min and max
        assert!(batch_size >= MIN_BATCH_SIZE);
        assert!(batch_size <= MAX_BATCH_SIZE);
        
        // Should not exceed memory constraints
        assert!(validate_batch_size(&device, batch_size).is_ok());
    }
}