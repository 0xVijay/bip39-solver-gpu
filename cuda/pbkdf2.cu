// CUDA kernel for PBKDF2-HMAC-SHA512 derivation
// High-performance GPU implementation for mnemonic seed derivation

#ifndef PBKDF2_CU
#define PBKDF2_CU

#include <cuda_runtime.h>
#include <stdint.h>
#include "sha512.cuh"

// Constants for PBKDF2
#define PBKDF2_ITERATIONS 2048
#define HMAC_SHA512_BLOCK_SIZE 128
#define HMAC_SHA512_DIGEST_SIZE 64
#define SHA512_BLOCK_SIZE 128
#define SHA512_DIGEST_SIZE 64

/**
 * GPU-optimized HMAC-SHA512 implementation
 */
// ...existing code...
#include "hmac_sha512.cuh"

/**
 * High-performance CUDA kernel for batch PBKDF2-HMAC-SHA512 computation
 * 
 * @param mnemonics     Array of mnemonic strings (input)
 * @param passphrases   Array of passphrase strings (input) 
 * @param seeds         Array of output seeds (64 bytes each)
 * @param count         Number of mnemonics to process
 */
__global__ void cuda_pbkdf2_batch(
    const char** mnemonics,
    const char** passphrases,
    uint8_t* seeds,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= count) {
        return;
    }
    
    // Get mnemonic and passphrase for this thread
    const char* mnemonic = mnemonics[idx];
    const char* passphrase = passphrases[idx];
    uint8_t* seed = &seeds[idx * 64];
    
    // Calculate lengths
    size_t mnemonic_len = 0;
    while (mnemonic[mnemonic_len] != '\0' && mnemonic_len < 256) {
        mnemonic_len++;
    }
    
    size_t passphrase_len = 0;
    while (passphrase[passphrase_len] != '\0' && passphrase_len < 256) {
        passphrase_len++;
    }
    
    // Create salt: "mnemonic" + passphrase
    uint8_t salt[256 + 8];
    const char* prefix = "mnemonic";
    for (int i = 0; i < 8; i++) {
        salt[i] = prefix[i];
    }
    for (size_t i = 0; i < passphrase_len; i++) {
        salt[8 + i] = (uint8_t)passphrase[i];
    }
    size_t salt_len = 8 + passphrase_len;
    
    // PBKDF2 implementation
    uint8_t u[SHA512_DIGEST_SIZE];
    uint8_t result[SHA512_DIGEST_SIZE];
    
    // Initialize result to zeros
    for (int i = 0; i < SHA512_DIGEST_SIZE; i++) {
        result[i] = 0;
    }
    
    // PBKDF2 only needs one block since output is 64 bytes
    uint8_t salt_with_counter[256 + 8 + 4];
    for (size_t i = 0; i < salt_len; i++) {
        salt_with_counter[i] = salt[i];
    }
    // Add counter (big-endian 1)
    salt_with_counter[salt_len] = 0;
    salt_with_counter[salt_len + 1] = 0; 
    salt_with_counter[salt_len + 2] = 0;
    salt_with_counter[salt_len + 3] = 1;
    
    // First iteration: U1 = HMAC(password, salt || counter)
    cuda_hmac_sha512((const uint8_t*)mnemonic, mnemonic_len, 
                     salt_with_counter, salt_len + 4, u);
    
    // Copy U1 to result
    for (int i = 0; i < SHA512_DIGEST_SIZE; i++) {
        result[i] = u[i];
    }
    
    // Remaining iterations: U_i = HMAC(password, U_{i-1})
    for (int iter = 1; iter < PBKDF2_ITERATIONS; iter++) {
        cuda_hmac_sha512((const uint8_t*)mnemonic, mnemonic_len, u, SHA512_DIGEST_SIZE, u);
        
        // XOR with result
        for (int i = 0; i < SHA512_DIGEST_SIZE; i++) {
            result[i] ^= u[i];
        }
    }
    
    // Copy result to output seed
    for (int i = 0; i < 64; i++) {
        seed[i] = result[i];
    }
}

/**
 * Host function to launch PBKDF2 kernel with optimized memory management
 */
extern "C" int cuda_pbkdf2_batch_host(
    const char** mnemonics,
    const char** passphrases, 
    uint8_t* seeds,
    uint32_t count
) {
    // Calculate optimal thread configuration
    int block_size = 256;  // Optimal for most GPUs
    int grid_size = (count + block_size - 1) / block_size;
    
    // Allocate GPU memory
    char** d_mnemonics;
    char** d_passphrases;
    uint8_t* d_seeds;
    
    // Allocate device memory for pointer arrays
    cudaMalloc(&d_mnemonics, count * sizeof(char*));
    cudaMalloc(&d_passphrases, count * sizeof(char*));
    cudaMalloc(&d_seeds, count * 64);
    
    // Allocate and copy string data
    for (uint32_t i = 0; i < count; i++) {
        size_t mnem_len = strlen(mnemonics[i]) + 1;
        size_t pass_len = strlen(passphrases[i]) + 1;
        
        char* d_mnem;
        char* d_pass;
        
        cudaMalloc(&d_mnem, mnem_len);
        cudaMalloc(&d_pass, pass_len);
        
        cudaMemcpy(d_mnem, mnemonics[i], mnem_len, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pass, passphrases[i], pass_len, cudaMemcpyHostToDevice);
        
        cudaMemcpy(&d_mnemonics[i], &d_mnem, sizeof(char*), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_passphrases[i], &d_pass, sizeof(char*), cudaMemcpyHostToDevice);
    }
    
    // Launch kernel
    cuda_pbkdf2_batch<<<grid_size, block_size>>>(
        (const char**)d_mnemonics, (const char**)d_passphrases, d_seeds, count
    );
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(seeds, d_seeds, count * 64, cudaMemcpyDeviceToHost);
    
    // Cleanup
    for (uint32_t i = 0; i < count; i++) {
        char* d_mnem;
        char* d_pass;
        cudaMemcpy(&d_mnem, &d_mnemonics[i], sizeof(char*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_pass, &d_passphrases[i], sizeof(char*), cudaMemcpyDeviceToHost);
        cudaFree(d_mnem);
        cudaFree(d_pass);
    }
    
    cudaFree(d_mnemonics);
    cudaFree(d_passphrases);
    cudaFree(d_seeds);
    
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

#endif // PBKDF2_CU