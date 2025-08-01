use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/");

    // Only build CUDA kernels if CUDA is available and requested
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() || check_cuda_available() {
        build_cuda_kernels();
    } else {
        println!(
            "cargo:warning=CUDA not available or not requested, skipping CUDA kernel compilation"
        );
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

    // Compile CUDA kernels directly into object files and link them manually
    let kernel_files = ["pbkdf2.cu", "secp256k1.cu", "keccak256.cu"];
    
    // Link CUDA runtime libraries first
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    
    // Add CUDA library search paths
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-search=native={}/lib", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib");
        println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
        println!("cargo:rustc-link-search=native=/opt/cuda/lib");
    }
    
    // Compile all CUDA files together into a single object file
    let obj_path = format!("{}/cuda_kernels.o", out_dir);
    let mut nvcc_args = vec![
        "-c",
        "-O3", 
        "--compiler-options", "-fPIC",
        "-o", &obj_path
    ];
    
    // Add all kernel files
    for kernel in &kernel_files {
        let kernel_path = cuda_dir.join(kernel);
        nvcc_args.push(kernel_path.to_str().unwrap());
    }
    
    let result = Command::new("nvcc").args(&nvcc_args).output();
    
    match result {
        Ok(output) if output.status.success() => {
            println!("cargo:warning=CUDA kernels compiled successfully");
            
            // Use cc crate to compile and link the object file
            let mut build = cc::Build::new();
            build.object(&obj_path);
            build.compile("cuda_kernels");
            
        },
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("cargo:warning=CUDA compilation failed: {}", stderr);
        },
        Err(e) => {
            println!("cargo:warning=Failed to run nvcc: {}", e);
        }
    }
}
