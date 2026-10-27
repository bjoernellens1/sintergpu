// SPDX-License-Identifier: Apache-2.0
// Fixed256-lane tree, expanded over caller-owned groupshared storage.
// Caller initializes all lanes/channels and performs the initial group barrier.
// Keep all lanes active through the eight per-stride barriers.
#ifndef SINTERGPU_BLOCK_REDUCE_SLANG
#define SINTERGPU_BLOCK_REDUCE_SLANG
#define SINTERGPU_ADD(A, B) ((A) + (B))
#define SINTERGPU_REDUCE_256(BUFFER, LANE, CHANNELS, COMBINE) \
    for (uint sintergpuStride = 128; sintergpuStride > 0; sintergpuStride >>= 1) { \
        if ((LANE) < sintergpuStride) { \
            for (uint sintergpuChannel = 0; sintergpuChannel < (CHANNELS); ++sintergpuChannel) { \
                uint sintergpuOffset = sintergpuChannel * 256 + (LANE); \
                (BUFFER)[sintergpuOffset] = COMBINE((BUFFER)[sintergpuOffset], (BUFFER)[sintergpuOffset + sintergpuStride]); \
            } \
        } \
        GroupMemoryBarrierWithGroupSync(); \
    }
#endif
