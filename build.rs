use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/");
    
    // Only build CUDA kernels if CUDA is available and requested
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() || check_cuda_available() {
        build_cuda_kernels();
    } else {
        println!("cargo:warning=CUDA not available or not requested, skipping CUDA kernel compilation");
    }
}

fn check_cuda_available() -> bool {
    // Check if nvcc is available
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn build_cuda_kernels() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let cuda_dir = PathBuf::from("cuda");
    
    println!("cargo:warning=Building CUDA kernels...");
    
    // Compile CUDA kernels to shared library
    let kernel_files = ["pbkdf2.cu", "secp256k1.cu", "keccak256.cu"];
    let mut nvcc_args = vec![
        "-shared",
        "-fPIC",
        "-O3",
        "--compiler-options",
        "-fPIC",
        "-o",
    ];
    
    let lib_path = format!("{}/libcuda_kernels.so", out_dir);
    nvcc_args.push(&lib_path);
    
    // Create kernel paths as owned strings
    let kernel_paths: Vec<String> = kernel_files.iter()
        .map(|kernel| cuda_dir.join(kernel).to_string_lossy().to_string())
        .collect();
    
    // Add paths to args
    for path in &kernel_paths {
        nvcc_args.push(path);
    }
    
    let output = Command::new("nvcc")
        .args(&nvcc_args)
        .output();
    
    match output {
        Ok(result) => {
            if result.status.success() {
                println!("cargo:warning=CUDA kernels compiled successfully");
                println!("cargo:rustc-link-search=native={}", out_dir);
                println!("cargo:rustc-link-lib=dylib=cuda_kernels");
                
                // Also link CUDA runtime
                println!("cargo:rustc-link-lib=dylib=cudart");
            } else {
                let stderr = String::from_utf8_lossy(&result.stderr);
                println!("cargo:warning=CUDA compilation failed: {}", stderr);
            }
        }
        Err(e) => {
            println!("cargo:warning=Failed to run nvcc: {}", e);
        }
    }
}