use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(cuda_available)");
    println!("cargo::rustc-check-cfg=cfg(opencl_available)");
    
    let features = env::var("CARGO_CFG_TARGET_FEATURES").unwrap_or_default();
    println!("cargo:rustc-cfg=features=\"{}\"", features);

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
        
        for source in &cuda_sources {
            println!("cargo:rerun-if-changed={}", source);
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
            
            // Only link OpenCL when actually available
            println!("cargo:rustc-link-lib=OpenCL");
            
            // Add OpenCL library search paths
            let opencl_lib_paths = [
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib",
                "/usr/local/lib",
                "/opt/intel/opencl/lib64",
            ];
            
            for path in &opencl_lib_paths {
                if std::path::Path::new(path).exists() {
                    println!("cargo:rustc-link-search=native={}", path);
                }
            }
        } else {
            println!("cargo:warning=OpenCL libraries not found. Install OpenCL drivers for GPU support.");
            println!("cargo:warning=Building without OpenCL support. GPU operations will not be available.");
            // Explicitly do not link OpenCL when not available
        }
    }
}

fn compile_cuda_kernels(sources: &[&str]) -> Result<(), String> {
    let out_dir = env::var("OUT_DIR").map_err(|e| format!("OUT_DIR not set: {}", e))?;
    let out_path = PathBuf::from(&out_dir);
    
    // Find nvcc compiler
    let nvcc = find_nvcc()?;
    
    // Compile all CUDA sources together to resolve cross-file dependencies
    let combined_object = out_path.join("cuda_kernels.o");
    let combined_lib = out_path.join("libcuda_kernels.a");
    
    println!("cargo:warning=Compiling CUDA kernels together to resolve dependencies...");
    for source in sources {
        println!("cargo:warning=Including CUDA source: {}", source);
    }
    
    // Prepare nvcc command with all source files
    let mut nvcc_cmd = std::process::Command::new(&nvcc);
    nvcc_cmd.args(&[
        "-c",
        "-o", combined_object.to_str().unwrap(),
        "--compiler-options", "-fPIC",
        "-arch=sm_75",  // Modern GPU architecture (RTX 20XX+, V100+) - avoids deprecated warnings
        "-O3",          // Optimize for performance
        "--std=c++11",  // C++11 standard
        "-Xptxas", "-O3", // PTX optimization
        "-lineinfo",    // Debug info
        "-Wno-deprecated-gpu-targets", // Suppress deprecated architecture warnings
        "--disable-warnings", // Disable warnings being treated as errors
    ]);
    
    // Add all source files to the command
    for source in sources {
        nvcc_cmd.arg(source);
    }
    
    let output = nvcc_cmd
        .output()
        .map_err(|e| format!("Failed to execute nvcc: {}", e))?;
    
    // Check if compilation succeeded (warnings are OK)
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        // If it's just warnings, continue
        if stderr.contains("warning") && !stderr.contains("error") && !stderr.contains("fatal") {
            println!("cargo:warning=CUDA compilation warnings: {}", stderr);
        } else {
            return Err(format!("nvcc compilation failed: {}{}", stderr, stdout));
        }
    } else {
        println!("cargo:warning=CUDA kernels compiled successfully");
    }
    
    // Create a static library from the combined object file
    let ar_output = std::process::Command::new("ar")
        .args(&["rcs", combined_lib.to_str().unwrap(), combined_object.to_str().unwrap()])
        .output()
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
