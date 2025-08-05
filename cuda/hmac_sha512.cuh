// Shared inline device implementation for cuda_hmac_sha512
#ifndef HMAC_SHA512_CUH
#define HMAC_SHA512_CUH

#include <stdint.h>
#include <stddef.h>

__device__ __inline__ void cuda_hmac_sha512(const uint8_t* key, size_t key_len,
                                            const uint8_t* message, size_t msg_len,
                                            uint8_t* digest) {
    // ...implementation copied from pbkdf2.cu...
    uint8_t ipad[128];
    uint8_t opad[128];
    uint8_t inner_digest[64];
    uint8_t key_buf[128];
    if (key_len > 128) {
        cuda_sha512(key, key_len, key_buf);
        key_len = 64;
    } else {
        for (size_t i = 0; i < key_len; i++) {
            key_buf[i] = key[i];
        }
    }
    for (size_t i = key_len; i < 128; i++) {
        key_buf[i] = 0;
    }
    for (int i = 0; i < 128; i++) {
        ipad[i] = key_buf[i] ^ 0x36;
        opad[i] = key_buf[i] ^ 0x5c;
    }
    uint8_t inner_msg[128 + 256];
    for (int i = 0; i < 128; i++) {
        inner_msg[i] = ipad[i];
    }
    for (size_t i = 0; i < msg_len && i < 256; i++) {
        inner_msg[128 + i] = message[i];
    }
    cuda_sha512(inner_msg, 128 + msg_len, inner_digest);
    uint8_t outer_msg[128 + 64];
    for (int i = 0; i < 128; i++) {
        outer_msg[i] = opad[i];
    }
    for (int i = 0; i < 64; i++) {
        outer_msg[128 + i] = inner_digest[i];
    }
    cuda_sha512(outer_msg, 128 + 64, digest);
}

#endif // HMAC_SHA512_CUH
