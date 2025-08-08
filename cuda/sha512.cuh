// Shared SHA-512 implementation for CUDA kernels
// This file contains the device-side SHA-512 function that can be used by all CUDA modules

#ifndef SHA512_CUH
#define SHA512_CUH

#include <stdint.h>
#include <stddef.h>

// SHA-512 constants (external declaration)
extern __constant__ uint64_t sha512_k_shared[80];

// GPU-optimized rotate right
__device__ __inline__ uint64_t rotr64_shared(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

// SHA-512 core functions
__device__ __inline__ uint64_t ch_shared(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __inline__ uint64_t maj_shared(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __inline__ uint64_t sigma0_shared(uint64_t x) {
    return rotr64_shared(x, 28) ^ rotr64_shared(x, 34) ^ rotr64_shared(x, 39);
}

__device__ __inline__ uint64_t sigma1_shared(uint64_t x) {
    return rotr64_shared(x, 14) ^ rotr64_shared(x, 18) ^ rotr64_shared(x, 41);
}

__device__ __inline__ uint64_t gamma0_shared(uint64_t x) {
    return rotr64_shared(x, 1) ^ rotr64_shared(x, 8) ^ (x >> 7);
}

__device__ __inline__ uint64_t gamma1_shared(uint64_t x) {
    return rotr64_shared(x, 19) ^ rotr64_shared(x, 61) ^ (x >> 6);
}

/**
 * GPU-optimized SHA-512 implementation (shared across all CUDA modules)
 */
__device__ __inline__ void cuda_sha512(const uint8_t* message, size_t len, uint8_t* digest) {
    uint64_t h[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };
    
    uint64_t w[80];
    
    // Process message in 1024-bit chunks
    size_t total_chunks = (len + 128 + 16) / 128; // Account for padding
    
    for (size_t chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++) {
        size_t chunk_start = chunk_idx * 128;
        
        // Initialize w[0..15] with message chunk
        for (int i = 0; i < 16; i++) {
            w[i] = 0;
            for (int j = 0; j < 8; j++) {
                size_t byte_pos = chunk_start + i * 8 + j;
                if (byte_pos < len) {
                    w[i] = (w[i] << 8) | message[byte_pos];
                } else if (byte_pos == len) {
                    // Add padding bit
                    w[i] = (w[i] << 8) | 0x80;
                } else if (chunk_idx == total_chunks - 1 && i >= 14) {
                    // Add length in last 16 bytes (big-endian)
                    uint64_t bit_len = len * 8;
                    if (i == 14) {
                        w[i] = 0; // High 64 bits of length (always 0 for our use case)
                    } else { // i == 15
                        w[i] = bit_len; // Low 64 bits of length
                    }
                } else {
                    w[i] = w[i] << 8; // Padding with zeros
                }
            }
        }
        
        // Extend w[16..79]
        for (int i = 16; i < 80; i++) {
            w[i] = gamma1_shared(w[i-2]) + w[i-7] + gamma0_shared(w[i-15]) + w[i-16];
        }
        
        // Initialize working variables
        uint64_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint64_t e = h[4], f = h[5], g = h[6], h7 = h[7];
        
        // Main loop
        for (int i = 0; i < 80; i++) {
            uint64_t t1 = h7 + sigma1_shared(e) + ch_shared(e, f, g) + sha512_k_shared[i] + w[i];
            uint64_t t2 = sigma0_shared(a) + maj_shared(a, b, c);
            
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
    
    // Output digest (big-endian)
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            digest[i * 8 + j] = (h[i] >> (56 - j * 8)) & 0xff;
        }
    }
}

#endif // SHA512_CUH