// Keccak-256 implementation for OpenCL
// Based on the FIPS-202 specification

#define KECCAK_ROUNDS 24

// Keccak-f[1600] round constants
__constant ulong keccak_round_constants[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL, 0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL, 0x0000000080008009UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL, 0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL,
    0x0000000080008082UL, 0x0000000080000081UL, 0x8000000080008080UL, 0x8000000000008082UL
};

// Rotation offsets for Keccak
__constant int keccak_rotation_offsets[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

// Rotate left function
ulong rotl(ulong x, int n) {
    return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600] permutation
void keccak_f1600(ulong state[25]) {
    for (int round = 0; round < KECCAK_ROUNDS; round++) {
        // Theta step
        ulong C[5];
        for (int x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        
        ulong D[5];
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl(C[(x + 1) % 5], 1);
        }
        
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                state[y * 5 + x] ^= D[x];
            }
        }
        
        // Rho and Pi steps
        ulong current = state[1];
        for (int t = 0; t < 24; t++) {
            int next_pos = ((t + 1) * (t + 2) / 2) % 25;
            ulong temp = state[next_pos];
            state[next_pos] = rotl(current, keccak_rotation_offsets[t]);
            current = temp;
        }
        
        // Chi step
        for (int y = 0; y < 5; y++) {
            ulong temp[5];
            for (int x = 0; x < 5; x++) {
                temp[x] = state[y * 5 + x];
            }
            for (int x = 0; x < 5; x++) {
                state[y * 5 + x] = temp[x] ^ ((~temp[(x + 1) % 5]) & temp[(x + 2) % 5]);
            }
        }
        
        // Iota step
        state[0] ^= keccak_round_constants[round];
    }
}

// Keccak-256 hash function
void keccak256(uchar *input, uint input_len, uchar *output) {
    ulong state[25] = {0};
    
    // Absorption phase
    uint rate = 136; // 1088 bits / 8 = 136 bytes for Keccak-256
    uint offset = 0;
    
    while (offset < input_len) {
        uint chunk_size = min(rate, input_len - offset);
        
        // XOR input chunk into state
        for (uint i = 0; i < chunk_size; i++) {
            uint byte_pos = i;
            uint word_pos = byte_pos / 8;
            uint byte_in_word = byte_pos % 8;
            
            if (word_pos < 25) {
                ulong byte_val = (ulong)input[offset + i];
                state[word_pos] ^= byte_val << (byte_in_word * 8);
            }
        }
        
        if (chunk_size == rate) {
            keccak_f1600(state);
        }
        
        offset += chunk_size;
    }
    
    // Padding
    uint pad_pos = input_len % rate;
    uint pad_word = pad_pos / 8;
    uint pad_byte = pad_pos % 8;
    
    if (pad_word < 25) {
        state[pad_word] ^= 0x01UL << (pad_byte * 8); // Keccak padding
        state[rate / 8 - 1] ^= 0x80UL << 56; // Domain separation
    }
    
    keccak_f1600(state);
    
    // Squeeze phase - extract 256 bits (32 bytes)
    for (int i = 0; i < 32; i++) {
        uint word_pos = i / 8;
        uint byte_in_word = i % 8;
        output[i] = (uchar)((state[word_pos] >> (byte_in_word * 8)) & 0xFF);
    }
}

// Convert bytes to little-endian format for Ethereum addresses
void bytes_to_little_endian(uchar *bytes, uint len) {
    for (uint i = 0; i < len / 2; i++) {
        uchar temp = bytes[i];
        bytes[i] = bytes[len - 1 - i];
        bytes[len - 1 - i] = temp;
    }
}