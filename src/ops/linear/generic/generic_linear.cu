#include "ops/linear/generic/generic_linear.h"

#include "ops/common/rowsplit_decode.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarps       = 8;
constexpr int kTokenChunk  = 8;

// One warp owns one output row for up to kTokenChunk token columns: every decoded weight group is
// reused across those columns. Lanes cover disjoint eight-value spans of K.
template <int Tokens>
__global__ void __launch_bounds__(kWarps * 32)
    generic_linear_kernel(RowPlanes w, const __nv_bfloat16* __restrict__ x,
                          __nv_bfloat16* __restrict__ out, int n, int k, int t_total) {
    const int row  = static_cast<int>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int t0   = static_cast<int>(blockIdx.y) * kTokenChunk;
    if (row >= n) return;
    const int tokens = min(Tokens, t_total - t0);
    float acc[Tokens];
#pragma unroll
    for (int t = 0; t < Tokens; ++t) acc[t] = 0.0F;
    for (int k0 = lane * 8; k0 < k; k0 += 256) {
        float wv[8];
        load_eight(w, row, k0, wv);
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            if (t >= tokens) break;
            const uint4 packed = *reinterpret_cast<const uint4*>(
                x + static_cast<std::int64_t>(t0 + t) * k + k0);
            const float2 a = bf16x2_bits_to_float2(packed.x);
            const float2 b = bf16x2_bits_to_float2(packed.y);
            const float2 c = bf16x2_bits_to_float2(packed.z);
            const float2 d = bf16x2_bits_to_float2(packed.w);
            float s = acc[t];
            s = fmaf(wv[0], a.x, s);
            s = fmaf(wv[1], a.y, s);
            s = fmaf(wv[2], b.x, s);
            s = fmaf(wv[3], b.y, s);
            s = fmaf(wv[4], c.x, s);
            s = fmaf(wv[5], c.y, s);
            s = fmaf(wv[6], d.x, s);
            s = fmaf(wv[7], d.y, s);
            acc[t] = s;
        }
    }
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        if (t >= tokens) break;
        const float v = warp_reduce_sum(acc[t]);
        if (lane == 0) out[static_cast<std::int64_t>(t0 + t) * n + row] = __float2bfloat16_rn(v);
    }
}

} // namespace

bool generic_linear_supported(const Weight& w) noexcept {
    const bool codec = w.qtype == QType::BF16 ? w.layout == QuantLayout::Contiguous
                                              : w.layout == QuantLayout::RowSplit &&
                                                    (w.qtype == QType::Q4_G64_FP16 ||
                                                     w.qtype == QType::Q5_G64_FP16 ||
                                                     w.qtype == QType::Q6_G64_FP16 ||
                                                     w.qtype == QType::Q8_G32_FP16);
    return codec && w.n > 0 && w.k > 0 && w.k % 8 == 0 && w.qdata != nullptr;
}

void generic_linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (!generic_linear_supported(w)) {
        throw std::invalid_argument("generic linear: unsupported weight representation");
    }
    const int t = x.ne[1];
    const dim3 grid((w.n + kWarps - 1) / kWarps, (t + kTokenChunk - 1) / kTokenChunk);
    const auto planes = row_planes(w);
    const auto* in    = static_cast<const __nv_bfloat16*>(x.data);
    auto* dst         = static_cast<__nv_bfloat16*>(out.data);
    if (t == 1) {
        generic_linear_kernel<1><<<grid, kWarps * 32, 0, stream>>>(planes, in, dst, w.n, w.k, t);
    } else {
        generic_linear_kernel<kTokenChunk>
            <<<grid, kWarps * 32, 0, stream>>>(planes, in, dst, w.n, w.k, t);
    }
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("generic linear launch: ") +
                                 cudaGetErrorString(status));
    }
}

} // namespace ninfer::ops::detail
