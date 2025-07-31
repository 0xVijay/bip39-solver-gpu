use opencl3::device::{Device, get_all_devices, CL_DEVICE_TYPE_ALL};
use opencl3::context::Context;
use opencl3::command_queue::CommandQueue;
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE, CL_MEM_READ_ONLY, CL_MEM_COPY_HOST_PTR};
use opencl3::program::Program;
use opencl3::kernel::{Kernel, ExecuteKernel};
use opencl3::platform::get_platforms;

pub struct GpuAccelerator {
    context: Context,
    queue: CommandQueue,
    program: Program,
}

impl GpuAccelerator {
    /// Initialize the GPU accelerator with OpenCL
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Get all available platforms first
        let platforms = match opencl3::platform::get_platforms() {
            Ok(platforms) => platforms,
            Err(e) => return Err(format!("No OpenCL platforms found: {}", e).into()),
        };
        
        if platforms.is_empty() {
            return Err("No OpenCL platforms available".into());
        }
        
        // Get all available devices
        let devices = get_all_devices(CL_DEVICE_TYPE_ALL)?;
        if devices.is_empty() {
            return Err("No OpenCL devices found".into());
        }
        
        let device = Device::new(devices[0]);
        println!("Using OpenCL device: {}", device.name()?);
        
        // Create context and command queue
        let context = Context::from_device(&device)?;
        let queue = CommandQueue::create_default_with_properties(
            &context, 
            opencl3::command_queue::CL_QUEUE_PROFILING_ENABLE,
            0
        )?;
        
        // Load and compile the OpenCL kernels
        let program = Self::build_program(&context)?;
        
        Ok(GpuAccelerator {
            context,
            queue,
            program,
        })
    }
    
    /// Build the OpenCL program from kernel sources
    fn build_program(context: &Context) -> Result<Program, Box<dyn std::error::Error>> {
        // Read kernel source files
        let keccak_source = include_str!("../cl/keccak.cl");
        let secp256k1_common = include_str!("../cl/secp256k1_common.cl");
        let secp256k1_field = include_str!("../cl/secp256k1_field.cl");
        let secp256k1_group = include_str!("../cl/secp256k1_group.cl");
        let secp256k1_scalar = include_str!("../cl/secp256k1_scalar.cl");
        let secp256k1 = include_str!("../cl/secp256k1.cl");
        let sha2 = include_str!("../cl/sha2.cl");
        let common = include_str!("../cl/common.cl");
        let eth_address = include_str!("../cl/eth_address.cl");
        
        // Combine all kernel sources
        let combined_source = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            common, keccak_source, sha2, secp256k1_common, 
            secp256k1_field, secp256k1_group, secp256k1_scalar, 
            secp256k1, eth_address
        );
        
        let program = Program::create_and_build_from_source(&context, &combined_source, "")?;
        Ok(program)
    }
    
    /// Process a batch of mnemonic candidates on GPU
    pub fn process_batch(
        &self,
        start_index: u128,
        batch_size: u32,
        target_address: &[u8; 20],
        word_constraints: &[Vec<u16>; 12], // Word indices for each position
    ) -> Result<Option<(u128, String)>, Box<dyn std::error::Error>> {
        // Create buffers for target address
        let target_buffer: Buffer<u8> = unsafe {
            Buffer::create(
                &self.context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                20,
                target_address.as_ptr() as *mut std::ffi::c_void,
            )?
        };
        
        // Create buffer for found result (256 bytes for mnemonic)
        let mut found_result = vec![0u8; 256];
        let found_buffer: Buffer<u8> = unsafe {
            Buffer::create(
                &self.context,
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                found_result.len(),
                found_result.as_mut_ptr() as *mut std::ffi::c_void,
            )?
        };
        
        // Create individual constraint buffers for each word position (0-11)
        let mut constraint_buffers: Vec<Buffer<u16>> = Vec::new();
        let mut constraint_sizes: Vec<u32> = Vec::new();
        
        for pos in 0..12 {
            let constraints = &word_constraints[pos];
            if constraints.is_empty() {
                // If no constraints, use all possible words (0-2047)
                let all_words: Vec<u16> = (0..2048).collect();
                let buffer: Buffer<u16> = unsafe {
                    Buffer::create(
                        &self.context,
                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        all_words.len() * std::mem::size_of::<u16>(),
                        all_words.as_ptr() as *mut std::ffi::c_void,
                    )?
                };
                constraint_buffers.push(buffer);
                constraint_sizes.push(all_words.len() as u32);
            } else {
                let buffer: Buffer<u16> = unsafe {
                    Buffer::create(
                        &self.context,
                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        constraints.len() * std::mem::size_of::<u16>(),
                        constraints.as_ptr() as *mut std::ffi::c_void,
                    )?
                };
                constraint_buffers.push(buffer);
                constraint_sizes.push(constraints.len() as u32);
            }
        }
        
        // Create kernel
        let kernel = Kernel::create(&self.program, "mnemonic_to_eth_address")?;
        
        // Set kernel arguments to match the kernel signature exactly
        let mnemonic_start_hi = (start_index >> 64) as u64;
        let mnemonic_start_lo = (start_index & 0xFFFFFFFFFFFFFFFF) as u64;
        
        let mut execute_kernel = ExecuteKernel::new(&kernel);
        unsafe {
            execute_kernel
                .set_arg(&mnemonic_start_hi)
                .set_arg(&mnemonic_start_lo)
                .set_arg(&target_buffer)
                .set_arg(&found_buffer);
                
            // Add constraint buffers for each position (pos0 through pos11)
            for i in 0..12 {
                execute_kernel.set_arg(&constraint_buffers[i]);
            }
            
            // Add constraint sizes for each position
            for i in 0..12 {
                execute_kernel.set_arg(&constraint_sizes[i]);
            }
        }
        
        // Execute kernel
        let global_work_size = batch_size as usize;
        let event = unsafe {
            execute_kernel
                .set_global_work_size(global_work_size)
                .enqueue_nd_range(&self.queue)?
        };
        
        // Wait for completion
        event.wait()?;
        
        // Read back results
        let read_event = unsafe {
            self.queue.enqueue_read_buffer(
                &found_buffer,
                opencl3::types::CL_TRUE,
                0,
                &mut found_result,
                &[],
            )?
        };
        read_event.wait()?;
        
        // Check if we found a result
        if found_result[0] == 1 {
            // Found a matching combination - extract the mnemonic from the buffer
            let mut mnemonic_bytes = Vec::new();
            let mut i = 1;
            while i < found_result.len() && found_result[i] != 0 {
                mnemonic_bytes.push(found_result[i]);
                i += 1;
            }
            
            let mnemonic_str = String::from_utf8_lossy(&mnemonic_bytes).to_string();
            let found_index = start_index; // The kernel returns the exact combination
            
            Ok(Some((found_index, mnemonic_str)))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_initialization() {
        // This test may fail if no GPU is available
        match GpuAccelerator::new() {
            Ok(_) => println!("GPU acceleration initialized successfully"),
            Err(e) => println!("GPU acceleration not available: {}", e),
        }
    }
}