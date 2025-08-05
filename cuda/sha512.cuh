// Shared SHA-512 implementation for CUDA kernels
// This file contains the device-side SHA-512 function that can be used by all CUDA modules

#ifndef SHA512_CUH
#define SHA512_CUH

#include <stdint.h>
#include <stddef.h>

// SHA-512 constants
__constant__ uint64_t sha512_k_shared[80] = {
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