use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
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
        
        // Compile CUDA kernels
        if let Err(e) = compile_cuda_kernels(&cuda_sources) {
            println!("cargo:warning=CUDA kernel compilation failed: {}", e);
            // Continue without CUDA support
        } else {
            println!("cargo:warning=CUDA kernels compiled successfully");
            
            // Link CUDA libraries
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cuda");
            
            // Add CUDA library search paths
            if let Ok(cuda_path) = env::var("CUDA_PATH") {
                println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
                println!("cargo:rustc-link-search=native={}/lib", cuda_path);
            }
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib");
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        }
    }

    // Check if OpenCL feature is enabled  
    if env::var("CARGO_FEATURE_OPENCL").is_ok() {
        println!("cargo:warning=Building with OpenCL support...");
        
        // Link OpenCL libraries
        println!("cargo:rustc-link-lib=OpenCL");
    }
}

fn compile_cuda_kernels(sources: &[&str]) -> Result<(), String> {
    let out_dir = env::var("OUT_DIR").map_err(|e| format!("OUT_DIR not set: {}", e))?;
    let out_path = PathBuf::from(&out_dir);
    
    // Find nvcc compiler
    let nvcc = find_nvcc()?;
    
    for source in sources {
        let source_path = PathBuf::from(source);
        let object_name = source_path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("Invalid source file: {}", source))?;
        let object_path = out_path.join(format!("{}.o", object_name));
        
        println!("cargo:warning=Compiling CUDA kernel: {}", source);
        
        let output = std::process::Command::new(&nvcc)
            .args(&[
                "-c",
                source,
                "-o", object_path.to_str().unwrap(),
                "--compiler-options", "-fPIC",
                "-arch=sm_50",  // Compatible with most modern GPUs
                "-O3",          // Optimize for performance
                "--std=c++11",  // C++11 standard
                "-Xptxas", "-O3", // PTX optimization
                "-lineinfo",    // Debug info
            ])
            .output()
            .map_err(|e| format!("Failed to execute nvcc: {}", e))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("nvcc compilation failed for {}: {}", source, stderr));
        }
        
        // Create a static library from the object file
        let lib_name = format!("lib{}.a", object_name);
        let lib_path = out_path.join(&lib_name);
        
        let ar_output = std::process::Command::new("ar")
            .args(&["rcs", lib_path.to_str().unwrap(), object_path.to_str().unwrap()])
            .output()
            .map_err(|e| format!("Failed to create static library: {}", e))?;
        
        if !ar_output.status.success() {
            let stderr = String::from_utf8_lossy(&ar_output.stderr);
            return Err(format!("ar failed for {}: {}", object_name, stderr));
        }
        
        // Tell cargo to link the static library
        println!("cargo:rustc-link-search=native={}", out_dir);
        println!("cargo:rustc-link-lib=static={}", object_name);
    }
    
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
