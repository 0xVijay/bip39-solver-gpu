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

    // Compile CUDA kernels to object files, then archive into static library
    let kernel_files = ["pbkdf2.cu", "secp256k1.cu", "keccak256.cu"];
    
    // First compile to object files
    let mut obj_files = Vec::new();
    for kernel in &kernel_files {
        let kernel_path = cuda_dir.join(kernel);
        let obj_name = kernel.replace(".cu", ".o");
        let obj_path = format!("{}/{}", out_dir, obj_name);
        
        let nvcc_args = vec![
            "-c",
            "-O3",
            "--compiler-options", "-fPIC",
            kernel_path.to_str().unwrap(),
            "-o", &obj_path
        ];
        
        let result = Command::new("nvcc").args(&nvcc_args).output();
        match result {
            Ok(output) if output.status.success() => {
                obj_files.push(obj_path);
            },
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!("cargo:warning=CUDA compilation failed for {}: {}", kernel, stderr);
                return;
            },
            Err(e) => {
                println!("cargo:warning=Failed to run nvcc for {}: {}", kernel, e);
                return;
            }
        }
    }
    
    // Create static library using ar
    let lib_path = format!("{}/libcuda_kernels.a", out_dir);
    let mut ar_args = vec!["rcs", &lib_path];
    for obj in &obj_files {
        ar_args.push(obj);
    }

    let ar_result = Command::new("ar").args(&ar_args).output();
    
    match ar_result {
        Ok(result) if result.status.success() => {
            println!("cargo:warning=CUDA kernels compiled successfully");
            println!("cargo:rustc-link-search=native={}", out_dir);
            println!("cargo:rustc-link-lib=static=cuda_kernels");

            // Link CUDA runtime and driver libraries
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rustc-link-lib=dylib=cuda");
            
            // Try to find CUDA library paths
            if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
                println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
                println!("cargo:rustc-link-search=native={}/lib", cuda_path);
            } else {
                // Default CUDA paths
                println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
                println!("cargo:rustc-link-search=native=/usr/local/cuda/lib");
                println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
                println!("cargo:rustc-link-search=native=/opt/cuda/lib");
            }
        },
        Ok(result) => {
            let stderr = String::from_utf8_lossy(&result.stderr);
            println!("cargo:warning=Static library creation failed: {}", stderr);
        },
        Err(e) => {
            println!("cargo:warning=Failed to run ar: {}", e);
        }
    }
}
