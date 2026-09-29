#pragma once

// Exact decode of eight consecutive stored weights from a row_split_k128_v1 matrix or a contiguous
// BF16 matrix, for kernels whose geometry is a runtime value.

#include "core/weight.h"
#include "ops/common/math.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct RowPlanes {
    const std::uint8_t* codes  = nullptr; // or BF16 values for QType::BF16
    const std::uint8_t* high   = nullptr;
    const std::uint8_t* scales = nullptr;
    int qtype                  = 0;
    int groups_per_row         = 0; // row-split groups per row; K for BF16
};

inline RowPlanes row_planes(const Weight& w) {
    RowPlanes p;
    p.codes  = static_cast<const std::uint8_t*>(w.qdata);
    p.high   = static_cast<const std::uint8_t*>(w.qhigh);
    p.scales = static_cast<const std::uint8_t*>(w.scales);
    p.qtype  = static_cast<int>(w.qtype);
    if (w.qtype == QType::BF16) {
        p.groups_per_row = w.k;
    } else {
        const int group  = w.qtype == QType::Q8_G32_FP16 ? 32 : 64;
        p.groups_per_row = (w.k + 127) / 128 * 128 / group;
    }
    return p;
}

// Weights for logical columns [k0, k0+8) of `row`; k0 % 8 == 0.
__device__ __forceinline__ void load_eight(const RowPlanes& p, std::int64_t row, int k0,
                                           float (&w)[8]) {
    if (p.qtype == static_cast<int>(QType::BF16)) {
        const uint4 packed = *reinterpret_cast<const uint4*>(
            reinterpret_cast<const __nv_bfloat16*>(p.codes) + row * p.groups_per_row + k0);
        const float2 a = bf16x2_bits_to_float2(packed.x);
        const float2 b = bf16x2_bits_to_float2(packed.y);
        const float2 c = bf16x2_bits_to_float2(packed.z);
        const float2 d = bf16x2_bits_to_float2(packed.w);
        w[0] = a.x, w[1] = a.y, w[2] = b.x, w[3] = b.y;
        w[4] = c.x, w[5] = c.y, w[6] = d.x, w[7] = d.y;
        return;
    }
    if (p.qtype == static_cast<int>(QType::Q8_G32_FP16)) {
        const std::int64_t gi = row * p.groups_per_row + (k0 >> 5);
        const float scale     = __half2float(
            __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(p.scales + gi * 2)));
        const uint2 bytes = *reinterpret_cast<const uint2*>(p.codes + gi * 32 + (k0 & 31));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[i]     = static_cast<float>(static_cast<std::int8_t>(bytes.x >> (8 * i))) * scale;
            w[i + 4] = static_cast<float>(static_cast<std::int8_t>(bytes.y >> (8 * i))) * scale;
        }
        return;
    }
    const std::int64_t gi = row * p.groups_per_row + (k0 >> 6);
    const int chunk       = (k0 & 63) >> 3;
    const std::uint32_t word =
        *reinterpret_cast<const std::uint32_t*>(p.codes + gi * 32 + chunk * 4);
    const auto scale = *reinterpret_cast<const std::uint16_t*>(p.scales + gi * 2);
    if (p.qtype == static_cast<int>(QType::Q4_G64_FP16)) {
        Q4SimtDecodeAtom::decode_eight(word, scale, w);
    } else if (p.qtype == static_cast<int>(QType::Q5_G64_FP16)) {
        Q5SimtDecodeAtom::decode_eight(word, p.high[gi * 8 + chunk], scale, w);
    } else {
        Q6SimtDecodeAtom::decode_eight(
            word, *reinterpret_cast<const std::uint16_t*>(p.high + gi * 16 + chunk * 2), scale,
            w);
    }
}

} // namespace ninfer::ops::detail
