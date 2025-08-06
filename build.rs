use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(cuda_available)");
    println!("cargo::rustc-check-cfg=cfg(opencl_available)");
    
    let features = env::var("CARGO_CFG_TARGET_FEATURES").unwrap_or_default();
    println!("cargo:rustc-cfg=features=\"{}\"", features);
    
    let out_dir = env::var("OUT_DIR").unwrap();

    // Check if CUDA feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:warning=Building CUDA kernels...");
        
        // Build all CUDA kernels
        let cuda_sources = [
            "cuda/pbkdf2.cu",
            "cuda/bip32.cu", 
            "cuda/secp256k1.cu",
            "cuda/keccak256.cu",
            "cuda/gpu_pipeline.cu",
        ];
        
        // CUDA header files that may change
        let cuda_headers = [
            "cuda/hmac_sha512.cuh",
            "cuda/sha512.cuh",
        ];
        
        for source in &cuda_sources {
            println!("cargo:rerun-if-changed={}", source);
        }
        
        for header in &cuda_headers {
            println!("cargo:rerun-if-changed={}", header);
        }
        
        // Check if CUDA toolkit is available before attempting compilation
        match find_nvcc() {
            Ok(_) => {
                // Compile CUDA kernels - linking happens only if this succeeds
                match compile_cuda_kernels(&cuda_sources) {
                    Ok(()) => {
                        println!("cargo:warning=CUDA kernels compiled successfully");
                        
                        // Link CUDA libraries only after successful compilation
                        println!("cargo:rustc-link-lib=cudart");
                        println!("cargo:rustc-link-lib=cuda");
                        
                        // Add CUDA library search paths
                        if let Ok(cuda_path) = env::var("CUDA_PATH") {
                            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
                            println!("cargo:rustc-link-search=native={}/lib", cuda_path);
                        }
                        
                        // Common CUDA installation paths
                        let cuda_lib_paths = [
                            "/usr/local/cuda/lib64",
                            "/usr/local/cuda/lib",
                            "/usr/local/cuda-12/lib64",
                            "/usr/local/cuda-11/lib64",
                            "/opt/cuda/lib64",
                            "/usr/lib/x86_64-linux-gnu",
                        ];
                        
                        for path in &cuda_lib_paths {
                            if std::path::Path::new(path).exists() {
                                println!("cargo:rustc-link-search=native={}", path);
                            }
                        }
                        
                        // Set compile-time flag to indicate CUDA is available
                        println!("cargo:rustc-cfg=cuda_available");
                    },
                    Err(e) => {
                        println!("cargo:warning=CUDA kernel compilation failed: {}", e);
                        println!("cargo:warning=Building without CUDA support. GPU operations will not be available.");
                        // Explicitly do not link CUDA libraries when compilation fails
                    }
                }
            },
            Err(e) => {
                println!("cargo:warning=CUDA toolkit not found: {}", e);
                println!("cargo:warning=Building without CUDA support. Install CUDA toolkit for GPU acceleration.");
                // No CUDA linking when nvcc is not available
            }
        }
    }

    // Check if OpenCL feature is enabled  
    if env::var("CARGO_FEATURE_OPENCL").is_ok() {
        println!("cargo:warning=Building with OpenCL support...");
        
        // Check if OpenCL libraries are available
        if is_opencl_available() {
            println!("cargo:rustc-cfg=opencl_available");
            println!("cargo:warning=OpenCL libraries found successfully");
        } else {
            println!("cargo:warning=OpenCL libraries not found. Install OpenCL drivers for GPU support.");
            println!("cargo:warning=Building without OpenCL support. GPU operations will not be available.");
            
            // Create stub OpenCL library to prevent linking failures
            // This is a workaround for opencl-sys always trying to link OpenCL
            if let Err(e) = create_stub_opencl_library(&out_dir) {
                println!("cargo:warning=Failed to create OpenCL stub: {}", e);
                println!("cargo:warning=OpenCL compilation may fail due to missing libraries");
            }
        }
    }
}

fn compile_cuda_kernels(sources: &[&str]) -> Result<(), String> {
    let out_dir = env::var("OUT_DIR").map_err(|e| format!("OUT_DIR not set: {}", e))?;
    let out_path = PathBuf::from(&out_dir);
    
    // Find nvcc compiler
    let nvcc = find_nvcc()?;
    
    let gpu_architectures = detect_gpu_architectures();
    println!("cargo:warning=Detected GPU architectures: {:?}", gpu_architectures);
    
    // Create a combined source file that includes all CUDA sources
    // This ensures all inline functions are available during compilation
    let combined_source = out_path.join("combined_kernels.cu");
    let mut combined_content = String::new();
    
    // Add common includes first
    combined_content.push_str("#include <cuda_runtime.h>\n");
    combined_content.push_str("#include <stdint.h>\n");
    combined_content.push_str("#include <cstring>\n");
    combined_content.push_str("#include <algorithm>\n\n");
    
    // Include header file contents directly instead of using relative paths
    // This avoids path issues when compiling in OUT_DIR
    
    // Include sha512.cuh content
    let sha512_header = std::fs::read_to_string("cuda/sha512.cuh")
        .map_err(|e| format!("Failed to read cuda/sha512.cuh: {}", e))?;
    combined_content.push_str("// ========== Content from cuda/sha512.cuh ==========\n");
    combined_content.push_str(&sha512_header);
    combined_content.push_str("\n");
    
    // Include hmac_sha512.cuh content (but skip the sha512.cuh include)
    let hmac_header = std::fs::read_to_string("cuda/hmac_sha512.cuh")
        .map_err(|e| format!("Failed to read cuda/hmac_sha512.cuh: {}", e))?;
    let hmac_filtered = hmac_header.lines()
        .filter(|line| !line.trim().starts_with("#include \"sha512.cuh\""))
        .collect::<Vec<_>>()
        .join("\n");
    combined_content.push_str("// ========== Content from cuda/hmac_sha512.cuh ==========\n");
    combined_content.push_str(&hmac_filtered);
    combined_content.push_str("\n\n");
    
    // Include all source files
    for source in sources {
        println!("cargo:warning=Including CUDA source: {}", source);
        
        // Read the source file and extract its content (excluding includes and guards)
        let source_content = std::fs::read_to_string(source)
            .map_err(|e| format!("Failed to read {}: {}", source, e))?;
            
        // Filter out include guards and duplicate includes
        let mut filtered_content = String::new();
        let mut in_include_guard = false;
        let mut skip_until_endif = 0;
        
        for line in source_content.lines() {
            let trimmed = line.trim();
            
            // Skip include guards
            if trimmed.starts_with("#ifndef") && (trimmed.contains("_CU") || trimmed.contains("_CUH")) {
                in_include_guard = true;
                continue;
            }
            if trimmed.starts_with("#define") && in_include_guard {
                in_include_guard = false;
                continue;
            }
            if trimmed == "#endif" && (skip_until_endif > 0 || line.contains("// ") || source_content.matches("#ifndef").count() == source_content.matches("#endif").count()) {
                if skip_until_endif > 0 {
                    skip_until_endif -= 1;
                }
                continue;
            }
            
            // Skip duplicate includes that we've already added
            if trimmed.starts_with("#include") {
                if trimmed.contains("cuda_runtime.h") || 
                   trimmed.contains("stdint.h") ||
                   trimmed.contains("cstring") ||
                   trimmed.contains("algorithm") ||
                   trimmed.contains("sha512.cuh") ||
                   trimmed.contains("hmac_sha512.cuh") {
                    continue;
                }
            }
            
            filtered_content.push_str(line);
            filtered_content.push('\n');
        }
        
        combined_content.push_str(&format!("// ========== Content from {} ==========\n", source));
        combined_content.push_str(&filtered_content);
        combined_content.push_str("\n");
    }
    
    // Write the combined source file
    std::fs::write(&combined_source, combined_content)
        .map_err(|e| format!("Failed to write combined source: {}", e))?;
    
    // Compile the combined source file
    println!("cargo:warning=Compiling combined CUDA kernels");
    let obj_path = out_path.join("cuda_kernels_combined.o");
    let mut nvcc_cmd = std::process::Command::new(&nvcc);
    nvcc_cmd.args(&[
        "-c", 
        combined_source.to_str().unwrap(), 
        "-o", 
        obj_path.to_str().unwrap(), 
        "--compiler-options", 
        "-fPIC", 
        "-O3", 
        "--std=c++11", 
        "-Xptxas", 
        "-O3", 
        "-lineinfo", 
        "-Wno-deprecated-gpu-targets", 
        "--disable-warnings"
    ]);
    
    for arch in &gpu_architectures {
        nvcc_cmd.arg(&format!("-arch={}", arch));
    }
    
    let output = nvcc_cmd.output().map_err(|e| format!("Failed to execute nvcc: {}", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stderr.contains("warning") && !stderr.contains("error") && !stderr.contains("fatal") {
            println!("cargo:warning=CUDA compilation warnings: {}", stderr);
        } else {
            return Err(format!("nvcc compilation failed: {}{}", stderr, stdout));
        }
    } else {
        println!("cargo:warning=Compiled combined CUDA kernels to {}", obj_path.display());
    }
    
    // Create static library from the single object file
    let combined_lib = out_path.join("libcuda_kernels.a");
    let mut ar_cmd = std::process::Command::new("ar");
    ar_cmd.arg("rcs").arg(combined_lib.to_str().unwrap()).arg(obj_path.to_str().unwrap());
    
    let ar_output = ar_cmd.output()
        .map_err(|e| format!("Failed to create static library: {}", e))?;
    
    if !ar_output.status.success() {
        let stderr = String::from_utf8_lossy(&ar_output.stderr);
        return Err(format!("ar failed for cuda_kernels: {}", stderr));
    }
    
    // Tell cargo to link the static library
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=cuda_kernels");
    
    Ok(())
}

fn detect_gpu_architectures() -> Vec<String> {
    let mut architectures = Vec::new();
    
    // Try to detect using nvidia-smi first (most reliable for VastAI/Docker)
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=name", "--format=csv,noheader,nounits"])
        .output()
    {
        if output.status.success() {
            let gpu_names = String::from_utf8_lossy(&output.stdout);
            for gpu_name in gpu_names.lines() {
                let gpu_name = gpu_name.trim();
                if let Some(arch) = map_gpu_name_to_architecture(gpu_name) {
                    if !architectures.contains(&arch) {
                        architectures.push(arch);
                    }
                }
            }
        }
    }
    
    // If nvidia-smi detection failed, try using deviceQuery or direct CUDA API
    if architectures.is_empty() {
        if let Ok(arch) = detect_via_device_query() {
            architectures.push(arch);
        }
    }
    
    // Fallback to common modern architectures if detection fails
    if architectures.is_empty() {
        println!("cargo:warning=Could not detect GPU architecture, using common fallbacks");
        architectures = vec![
            "sm_86".to_string(), // RTX 30XX, RTX 40XX (Ampere/Ada Lovelace)
            "sm_80".to_string(), // A100 (Ampere)
            "sm_75".to_string(), // RTX 20XX, V100 (Turing/Volta)
        ];
    }
    
    architectures
}

fn map_gpu_name_to_architecture(gpu_name: &str) -> Option<String> {
    let gpu_name_lower = gpu_name.to_lowercase();
    
    // RTX 40XX series (Ada Lovelace) - sm_89
    if gpu_name_lower.contains("rtx 40") || gpu_name_lower.contains("rtx40") {
        return Some("sm_89".to_string());
    }
    
    // RTX 30XX series (Ampere) - sm_86
    if gpu_name_lower.contains("rtx 30") || gpu_name_lower.contains("rtx30") || 
       gpu_name_lower.contains("rtx 3060") || gpu_name_lower.contains("rtx 3070") ||
       gpu_name_lower.contains("rtx 3080") || gpu_name_lower.contains("rtx 3090") {
        return Some("sm_86".to_string());
    }
    
    // RTX 20XX series (Turing) - sm_75
    if gpu_name_lower.contains("rtx 20") || gpu_name_lower.contains("rtx20") ||
       gpu_name_lower.contains("rtx 2060") || gpu_name_lower.contains("rtx 2070") ||
       gpu_name_lower.contains("rtx 2080") {
        return Some("sm_75".to_string());
    }
    
    // GTX 16XX series (Turing) - sm_75
    if gpu_name_lower.contains("gtx 16") || gpu_name_lower.contains("gtx16") ||
       gpu_name_lower.contains("gtx 1660") || gpu_name_lower.contains("gtx 1650") {
        return Some("sm_75".to_string());
    }
    
    // Tesla V100 (Volta) - sm_70
    if gpu_name_lower.contains("v100") {
        return Some("sm_70".to_string());
    }
    
    // Tesla T4 (Turing) - sm_75
    if gpu_name_lower.contains("t4") {
        return Some("sm_75".to_string());
    }
    
    // Tesla A100 (Ampere) - sm_80
    if gpu_name_lower.contains("a100") {
        return Some("sm_80".to_string());
    }
    
    // Tesla A10/A30/A40 (Ampere) - sm_86
    if gpu_name_lower.contains("a10") || gpu_name_lower.contains("a30") || gpu_name_lower.contains("a40") {
        return Some("sm_86".to_string());
    }
    
    // GTX 10XX series (Pascal) - sm_61/sm_60
    if gpu_name_lower.contains("gtx 10") || gpu_name_lower.contains("gtx10") ||
       gpu_name_lower.contains("gtx 1080") || gpu_name_lower.contains("gtx 1070") ||
       gpu_name_lower.contains("gtx 1060") {
        return Some("sm_61".to_string());
    }
    
    // Titan series
    if gpu_name_lower.contains("titan") {
        if gpu_name_lower.contains("rtx") {
            return Some("sm_75".to_string()); // Titan RTX
        } else {
            return Some("sm_61".to_string()); // Older Titans
        }
    }
    
    // Quadro series (approximate based on generation)
    if gpu_name_lower.contains("quadro") {
        if gpu_name_lower.contains("rtx") {
            return Some("sm_75".to_string()); // Quadro RTX series
        } else {
            return Some("sm_61".to_string()); // Older Quadro
        }
    }
    
    println!("cargo:warning=Unknown GPU: {}, using sm_75 as fallback", gpu_name);
    Some("sm_75".to_string()) // Safe fallback for unknown GPUs
}

fn detect_via_device_query() -> Result<String, String> {
    // Try to find deviceQuery utility (often included with CUDA samples)
    let device_query_paths = [
        "deviceQuery",
        "/usr/local/cuda/extras/demo_suite/deviceQuery",
        "/usr/local/cuda/bin/deviceQuery",
        "/opt/cuda/extras/demo_suite/deviceQuery",
    ];
    
    for path in &device_query_paths {
        if let Ok(output) = std::process::Command::new(path).output() {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                
                // Parse compute capability from deviceQuery output
                for line in output_str.lines() {
                    if line.contains("CUDA Capability Major/Minor version number") {
                        if let Some(capability) = extract_compute_capability(line) {
                            return Ok(format!("sm_{}", capability.replace(".", "")));
                        }
                    }
                }
            }
        }
    }
    
    Err("Could not detect via deviceQuery".to_string())
}

fn extract_compute_capability(line: &str) -> Option<String> {
    // Extract version like "7.5" from line containing capability info
    let parts: Vec<&str> = line.split_whitespace().collect();
    for part in parts {
        if part.contains('.') && part.len() <= 4 {
            if let Ok(_) = part.parse::<f32>() {
                return Some(part.to_string());
            }
        }
    }
    None
}

fn find_nvcc() -> Result<String, String> {
    // Try different common locations for nvcc
    let nvcc_paths = [
        "nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/opt/cuda/bin/nvcc",
    ];
    
    for path in &nvcc_paths {
        if std::process::Command::new(path)
            .arg("--version")
            .output()
            .is_ok()
        {
            return Ok(path.to_string());
        }
    }
    
    // Check CUDA_PATH environment variable
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc_path = format!("{}/bin/nvcc", cuda_path);
        if std::process::Command::new(&nvcc_path)
            .arg("--version")
            .output()
            .is_ok()
        {
            return Ok(nvcc_path);
        }
    }
    
    Err("nvcc compiler not found. Please install CUDA toolkit or set CUDA_PATH environment variable.".to_string())
}

fn is_opencl_available() -> bool {
    // Check for common OpenCL library locations
    let opencl_paths = [
        "/usr/lib/x86_64-linux-gnu/libOpenCL.so",
        "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
        "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0",
        "/usr/lib/libOpenCL.so",
        "/usr/lib/libOpenCL.so.1",
        "/usr/local/lib/libOpenCL.so",
        "/usr/local/lib/libOpenCL.so.1",
        "/opt/intel/opencl/lib64/libOpenCL.so",
        "/opt/intel/opencl/lib64/libOpenCL.so.1",
        "/opt/amdgpu/lib64/libOpenCL.so",
        "/opt/amdgpu/lib64/libOpenCL.so.1",
        "/usr/lib64/libOpenCL.so",
        "/usr/lib64/libOpenCL.so.1",
    ];
    
    for path in &opencl_paths {
        if std::path::Path::new(path).exists() {
            println!("cargo:warning=Found OpenCL library at: {}", path);
            return true;
        }
    }
    
    // Try to use pkg-config to find OpenCL
    if let Ok(output) = std::process::Command::new("pkg-config")
        .args(&["--exists", "OpenCL"])
        .output()
    {
        if output.status.success() {
            println!("cargo:warning=Found OpenCL via pkg-config");
            return true;
        }
    }
    
    // Try ldconfig to search for OpenCL libraries
    if let Ok(output) = std::process::Command::new("ldconfig")
        .args(&["-p"])
        .output()
    {
        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if output_str.contains("libOpenCL.so") {
                println!("cargo:warning=Found OpenCL via ldconfig");
                return true;
            }
        }
    }
    
    // Check OpenCL headers as well (for development packages)
    let header_paths = [
        "/usr/include/CL/cl.h",
        "/usr/include/OpenCL/opencl.h",
        "/usr/local/include/CL/cl.h",
    ];
    
    for path in &header_paths {
        if std::path::Path::new(path).exists() {
            println!("cargo:warning=Found OpenCL headers at: {}", path);
            // Only return true if we also have libraries
            continue;
        }
    }
    
    false
}

fn create_stub_opencl_library(out_dir: &str) -> Result<(), String> {
    let out_path = PathBuf::from(out_dir);
    
    // Create a stub library that provides empty OpenCL symbols
    let stub_c_file = out_path.join("opencl_stub.c");
    let stub_content = r#"
// Stub OpenCL implementation to prevent linking errors when OpenCL is not available
// These functions will return error codes indicating OpenCL is not available

typedef int cl_int;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_event;
typedef unsigned int cl_uint;
typedef unsigned long size_t;
typedef unsigned long cl_ulong;
typedef size_t cl_size_t;
typedef cl_uint cl_device_info;
typedef cl_uint cl_platform_info;

#define CL_SUCCESS 0
#define CL_PLATFORM_NOT_FOUND_KHR -1001

// Essential OpenCL functions that opencl-sys expects
cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) {
    if (num_platforms) *num_platforms = 0;
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clGetDeviceIDs(cl_platform_id platform, unsigned long device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices) {
    if (num_devices) *num_devices = 0;
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (param_value_size_ret) *param_value_size_ret = 0;
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (param_value_size_ret) *param_value_size_ret = 0;
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_context clCreateContext(const cl_uint* properties, cl_uint num_devices, const cl_device_id* devices, void* pfn_notify, void* user_data, cl_int* errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_PLATFORM_NOT_FOUND_KHR;
    return 0;
}

cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_ulong properties, cl_int* errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_PLATFORM_NOT_FOUND_KHR;
    return 0;
}

cl_mem clCreateBuffer(cl_context context, cl_ulong flags, size_t size, void* host_ptr, cl_int* errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_PLATFORM_NOT_FOUND_KHR;
    return 0;
}

cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char** strings, const size_t* lengths, cl_int* errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_PLATFORM_NOT_FOUND_KHR;
    return 0;
}

cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options, void* pfn_notify, void* user_data) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_kernel clCreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_PLATFORM_NOT_FOUND_KHR;
    return 0;
}

cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_uint blocking_read, size_t offset, size_t size, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_uint blocking_write, size_t offset, size_t size, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clFinish(cl_command_queue command_queue) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clReleaseKernel(cl_kernel kernel) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clReleaseProgram(cl_program program) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clReleaseMemObject(cl_mem memobj) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clReleaseContext(cl_context context) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clReleaseEvent(cl_event event) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

// Additional functions needed by opencl3 crate
cl_int clGetProgramInfo(cl_program program, cl_uint param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (param_value_size_ret) *param_value_size_ret = 0;
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clGetKernelInfo(cl_kernel kernel, cl_uint param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (param_value_size_ret) *param_value_size_ret = 0;
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_uint param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (param_value_size_ret) *param_value_size_ret = 0;
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clGetContextInfo(cl_context context, cl_uint param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (param_value_size_ret) *param_value_size_ret = 0;
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clGetMemObjectInfo(cl_mem memobj, cl_uint param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (param_value_size_ret) *param_value_size_ret = 0;
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clGetCommandQueueInfo(cl_command_queue command_queue, cl_uint param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (param_value_size_ret) *param_value_size_ret = 0;
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clGetEventInfo(cl_event event, cl_uint param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (param_value_size_ret) *param_value_size_ret = 0;
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clGetEventProfilingInfo(cl_event event, cl_uint param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (param_value_size_ret) *param_value_size_ret = 0;
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clWaitForEvents(cl_uint num_events, const cl_event* event_list) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clRetainEvent(cl_event event) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clRetainKernel(cl_kernel kernel) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clRetainProgram(cl_program program) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clRetainMemObject(cl_mem memobj) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clRetainCommandQueue(cl_command_queue command_queue) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clRetainContext(cl_context context) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clRetainDevice(cl_device_id device) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}

cl_int clReleaseDevice(cl_device_id device) {
    return CL_PLATFORM_NOT_FOUND_KHR;
}
"#;
    
    std::fs::write(&stub_c_file, stub_content)
        .map_err(|e| format!("Failed to write OpenCL stub: {}", e))?;
    
    // Compile the stub into a static library
    let stub_lib = out_path.join("libOpenCL.a");
    let mut cc_cmd = std::process::Command::new("gcc");
    cc_cmd.args(&["-c", stub_c_file.to_str().unwrap(), "-o", &format!("{}/opencl_stub.o", out_dir)]);
    
    let cc_output = cc_cmd.output()
        .map_err(|e| format!("Failed to compile OpenCL stub: {}", e))?;
    
    if !cc_output.status.success() {
        let stderr = String::from_utf8_lossy(&cc_output.stderr);
        return Err(format!("gcc failed for OpenCL stub: {}", stderr));
    }
    
    // Create static library
    let mut ar_cmd = std::process::Command::new("ar");
    ar_cmd.args(&["rcs", stub_lib.to_str().unwrap(), &format!("{}/opencl_stub.o", out_dir)]);
    
    let ar_output = ar_cmd.output()
        .map_err(|e| format!("Failed to create OpenCL stub library: {}", e))?;
    
    if !ar_output.status.success() {
        let stderr = String::from_utf8_lossy(&ar_output.stderr);
        return Err(format!("ar failed for OpenCL stub: {}", stderr));
    }
    
    // Tell cargo to use our stub library instead of system OpenCL
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=OpenCL");
    println!("cargo:warning=Created stub OpenCL library to prevent linking failures");
    
    Ok(())
}
