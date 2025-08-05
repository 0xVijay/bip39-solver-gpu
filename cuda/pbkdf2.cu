// CUDA kernel for PBKDF2-HMAC-SHA512 derivation
// High-performance GPU implementation for mnemonic seed derivation

#ifndef PBKDF2_CU
#define PBKDF2_CU

#include <cuda_runtime.h>
#include <stdint.h>

// Constants for PBKDF2
#define PBKDF2_ITERATIONS 2048
#define HMAC_SHA512_BLOCK_SIZE 128
#define HMAC_SHA512_DIGEST_SIZE 64
#define SHA512_BLOCK_SIZE 128
#define SHA512_DIGEST_SIZE 64

// SHA-512 constants
__constant__ uint64_t sha512_k[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

// GPU-optimized rotate right
__device__ inline uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

// SHA-512 core functions
__device__ inline uint64_t ch(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ (~x & z);
}

__device__ inline uint64_t maj(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ inline uint64_t sigma0(uint64_t x) {
    return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39);
}

__device__ inline uint64_t sigma1(uint64_t x) {
    return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41);
}

__device__ inline uint64_t gamma0(uint64_t x) {
    return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7);
}

__device__ inline uint64_t gamma1(uint64_t x) {
    return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6);
}

/**
 * GPU-optimized SHA-512 implementation
 */
__device__ void cuda_sha512(const uint8_t* message, size_t len, uint8_t* digest) {
    uint64_t h[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };
    
    uint64_t w[80];
    uint64_t msg_len = len * 8;
    
    // Process message in 1024-bit chunks
    for (size_t chunk_start = 0; chunk_start < len; chunk_start += 128) {
        // Initialize w[0..15] with message chunk
        for (int i = 0; i < 16; i++) {
            w[i] = 0;
            for (int j = 0; j < 8; j++) {
                size_t byte_pos = chunk_start + i * 8 + j;
                if (byte_pos < len) {
                    w[i] = (w[i] << 8) | message[byte_pos];
                }
            }
        }
        
        // Extend w[16..79]
        for (int i = 16; i < 80; i++) {
            w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
        }
        
        // Initialize working variables
        uint64_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint64_t e = h[4], f = h[5], g = h[6], h7 = h[7];
        
        // Main loop
        for (int i = 0; i < 80; i++) {
            uint64_t t1 = h7 + sigma1(e) + ch(e, f, g) + sha512_k[i] + w[i];
            uint64_t t2 = sigma0(a) + maj(a, b, c);
            
            h7 = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        
        // Add to hash
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += h7;
    }
    
    // Output digest
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            digest[i * 8 + j] = (h[i] >> (56 - j * 8)) & 0xff;
        }
    }
}

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