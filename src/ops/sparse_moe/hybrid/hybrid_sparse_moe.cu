#include "ninfer/ops/hybrid_sparse_moe.h"

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/sparse_moe/hybrid/hybrid_host_runtime.h"

#include <cuda.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("hybrid sparse MoE: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

void check(CUresult status, const char* what) {
    if (status != CUDA_SUCCESS) {
        const char* text = nullptr;
        cuGetErrorString(status, &text);
        throw std::runtime_error(std::string("hybrid sparse MoE: ") + what + ": " +
                                 (text ? text : "unknown"));
    }
}

// Device view of a row-split matrix.
struct Planes {
    const std::uint8_t* codes;
    const std::uint8_t* high;
    const std::uint8_t* scales;
    int qtype; // QType value
    int groups_per_row;
};

Planes planes(const Weight& w) {
    const int group = w.qtype == QType::Q8_G32_FP16 ? 32 : 64;
    return {static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.qhigh),
            static_cast<const std::uint8_t*>(w.scales), static_cast<int>(w.qtype),
            (w.k + 127) / 128 * 128 / group};
}

// Eight consecutive stored weights starting at logical column k0 (k0 % 8 == 0), exactly decoded.
__device__ __forceinline__ void load_eight(const Planes& p, std::int64_t row, int k0,
                                           float (&w)[8]) {
    using namespace ninfer::ops::detail;
    if (p.qtype == static_cast<int>(QType::Q8_G32_FP16)) {
        const std::int64_t gi = row * p.groups_per_row + (k0 >> 5);
        const float scale =
            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(p.scales + gi * 2)));
        const uint2 bytes = *reinterpret_cast<const uint2*>(p.codes + gi * 32 + (k0 & 31));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[i]     = static_cast<float>(static_cast<std::int8_t>(bytes.x >> (8 * i))) * scale;
            w[i + 4] = static_cast<float>(static_cast<std::int8_t>(bytes.y >> (8 * i))) * scale;
        }
        return;
    }
    const std::int64_t gi    = row * p.groups_per_row + (k0 >> 6);
    const int chunk          = (k0 & 63) >> 3;
    const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(p.codes + gi * 32 + chunk * 4);
    const auto scale         = *reinterpret_cast<const std::uint16_t*>(p.scales + gi * 2);
    if (p.qtype == static_cast<int>(QType::Q4_G64_FP16)) {
        Q4SimtDecodeAtom::decode_eight(word, scale, w);
    } else if (p.qtype == static_cast<int>(QType::Q5_G64_FP16)) {
        Q5SimtDecodeAtom::decode_eight(word, p.high[gi * 8 + chunk], scale, w);
    } else {
        Q6SimtDecodeAtom::decode_eight(
            word, *reinterpret_cast<const std::uint16_t*>(p.high + gi * 16 + chunk * 2), scale, w);
    }
}

// Warp dot product of one stored row with a BF16 vector of length k (k % 8 == 0).
__device__ __forceinline__ float warp_dot_bf16(const Planes& p, std::int64_t row,
                                               const __nv_bfloat16* x, int k) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float acc      = 0.0F;
    for (int k0 = lane * 8; k0 < k; k0 += 256) {
        float w[8];
        load_eight(p, row, k0, w);
        const uint4 packed = *reinterpret_cast<const uint4*>(x + k0);
        const float2 a     = bf16x2_bits_to_float2(packed.x);
        const float2 b     = bf16x2_bits_to_float2(packed.y);
        const float2 c     = bf16x2_bits_to_float2(packed.z);
        const float2 d     = bf16x2_bits_to_float2(packed.w);
        acc = fmaf(w[0], a.x, acc);
        acc = fmaf(w[1], a.y, acc);
        acc = fmaf(w[2], b.x, acc);
        acc = fmaf(w[3], b.y, acc);
        acc = fmaf(w[4], c.x, acc);
        acc = fmaf(w[5], c.y, acc);
        acc = fmaf(w[6], d.x, acc);
        acc = fmaf(w[7], d.y, acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float warp_dot_f32(const Planes& p, std::int64_t row, const float* x,
                                              int k) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float acc      = 0.0F;
    for (int k0 = lane * 8; k0 < k; k0 += 256) {
        float w[8];
        load_eight(p, row, k0, w);
        const float4 a = *reinterpret_cast<const float4*>(x + k0);
        const float4 b = *reinterpret_cast<const float4*>(x + k0 + 4);
        acc = fmaf(w[0], a.x, acc);
        acc = fmaf(w[1], a.y, acc);
        acc = fmaf(w[2], a.z, acc);
        acc = fmaf(w[3], a.w, acc);
        acc = fmaf(w[4], b.x, acc);
        acc = fmaf(w[5], b.y, acc);
        acc = fmaf(w[6], b.z, acc);
        acc = fmaf(w[7], b.w, acc);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float silu(float v) { return v / (1.0F + __expf(-v)); }

struct SlotTable {
    std::int16_t slot[kHybridMoeMaxExperts];
};

struct RouteArgs {
    const __nv_bfloat16* x;
    const __nv_bfloat16* router; // [E+1, H]
    int experts, top_k, hidden;
    int* ids;             // device [T][K]
    float* alpha;         // device [T][K]
    float* shared_scale;  // device [T]
    int* host_ids;        // mapped [T][K]
    float* host_alpha;    // mapped [T][K]
    __nv_bfloat16* host_x; // mapped [T][H]
};

// One block per token: router logits, exact top-K with lower-id tie break, normalized weights.
__global__ void __launch_bounds__(kThreads) route_kernel(RouteArgs a) {
    __shared__ float logits[kHybridMoeMaxExperts + 1];
    __shared__ float red_value[kWarps];
    __shared__ int red_index[kWarps];
    __shared__ float chosen[kHybridMoeMaxTopK];
    __shared__ int chosen_id[kHybridMoeMaxTopK];
    const int t    = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const __nv_bfloat16* xt = a.x + static_cast<std::int64_t>(t) * a.hidden;

    for (int r = warp; r <= a.experts; r += kWarps) {
        const __nv_bfloat16* row = a.router + static_cast<std::int64_t>(r) * a.hidden;
        float acc = 0.0F;
        for (int k = lane * 2; k < a.hidden; k += 64) {
            const float2 w = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(row + k));
            const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xt + k));
            acc            = fmaf(w.x, v.x, fmaf(w.y, v.y, acc));
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) logits[r] = acc;
    }
    // Publish the represented input for the host share while the router finishes.
    for (int k = static_cast<int>(threadIdx.x); k < a.hidden; k += kThreads) {
        a.host_x[static_cast<std::int64_t>(t) * a.hidden + k] = xt[k];
    }
    __syncthreads();

    for (int s = 0; s < a.top_k; ++s) {
        float best = -INFINITY;
        int best_i = 0x7fffffff;
        for (int e = static_cast<int>(threadIdx.x); e < a.experts; e += kThreads) {
            const float v = logits[e];
            if (v > best || (v == best && e < best_i)) {
                best   = v;
                best_i = e;
            }
        }
        for (int off = 16; off; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffU, best, off);
            const int oi   = __shfl_xor_sync(0xffffffffU, best_i, off);
            if (ov > best || (ov == best && oi < best_i)) {
                best   = ov;
                best_i = oi;
            }
        }
        if (lane == 0) {
            red_value[warp] = best;
            red_index[warp] = best_i;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float v = red_value[0];
            int i   = red_index[0];
            for (int w = 1; w < kWarps; ++w) {
                if (red_value[w] > v || (red_value[w] == v && red_index[w] < i)) {
                    v = red_value[w];
                    i = red_index[w];
                }
            }
            chosen[s]    = v;
            chosen_id[s] = i;
            logits[i]    = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // softmax(p)[e] / sum over selected = exp(l_e - l_max) / sum exp(l_sel - l_max).
        const float top = chosen[0];
        float sum       = 0.0F;
        for (int s = 0; s < a.top_k; ++s) sum += expf(chosen[s] - top);
        for (int s = 0; s < a.top_k; ++s) {
            const int o      = t * a.top_k + s;
            const float w    = expf(chosen[s] - top) / sum;
            a.ids[o]         = chosen_id[s];
            a.alpha[o]       = w;
            a.host_ids[o]    = chosen_id[s];
            a.host_alpha[o]  = w;
        }
        a.shared_scale[t] = 1.0F / (1.0F + expf(-logits[a.experts]));
    }
}

struct ExpertArgs {
    const __nv_bfloat16* x;
    Planes device_gate_up, shared_gate_up;
    SlotTable slots;
    const int* ids;
    int top_k, hidden, intermediate, shared_intermediate, act_stride;
    float* act; // [T][K+1][act_stride]
};

// grid (ceil(max(I,Is)/kWarps), T*(K+1)): gate/up rows and SwiGLU for resident and shared paths.
__global__ void __launch_bounds__(kThreads) expert_act_kernel(ExpertArgs a) {
    const int path  = static_cast<int>(blockIdx.y) % (a.top_k + 1);
    const int t     = static_cast<int>(blockIdx.y) / (a.top_k + 1);
    const int j     = static_cast<int>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5);
    const bool shared = path == a.top_k;
    const int width   = shared ? a.shared_intermediate : a.intermediate;
    if (j >= width) return;
    const __nv_bfloat16* xt = a.x + static_cast<std::int64_t>(t) * a.hidden;
    float g;
    float u;
    if (shared) {
        g = warp_dot_bf16(a.shared_gate_up, j, xt, a.hidden);
        u = warp_dot_bf16(a.shared_gate_up, a.shared_intermediate + j, xt, a.hidden);
    } else {
        const int slot = a.slots.slot[a.ids[t * a.top_k + path]];
        if (slot < 0) return;
        const std::int64_t base = static_cast<std::int64_t>(slot) * 2 * a.intermediate;
        g = warp_dot_bf16(a.device_gate_up, base + j, xt, a.hidden);
        u = warp_dot_bf16(a.device_gate_up, base + a.intermediate + j, xt, a.hidden);
    }
    if ((threadIdx.x & 31) == 0) {
        a.act[(static_cast<std::int64_t>(t) * (a.top_k + 1) + path) * a.act_stride + j] =
            silu(g) * u;
    }
}

struct DownArgs {
    Planes device_down, shared_down;
    SlotTable slots;
    const int* ids;
    const float* alpha;
    const float* shared_scale;
    const float* act;
    int top_k, hidden, intermediate, shared_intermediate, act_stride;
    float* partial; // [T][H]
};

// grid (ceil(H/kWarps), T): device share of the routed sum plus the gated shared expert.
__global__ void __launch_bounds__(kThreads) expert_down_kernel(DownArgs a) {
    const int t = static_cast<int>(blockIdx.y);
    const int r = static_cast<int>(blockIdx.x) * kWarps + (static_cast<int>(threadIdx.x) >> 5);
    if (r >= a.hidden) return;
    const float* act_t = a.act + static_cast<std::int64_t>(t) * (a.top_k + 1) * a.act_stride;
    float sum          = a.shared_scale[t] *
                warp_dot_f32(a.shared_down, r, act_t + static_cast<std::int64_t>(a.top_k) * a.act_stride,
                             a.shared_intermediate);
    for (int s = 0; s < a.top_k; ++s) {
        const int slot = a.slots.slot[a.ids[t * a.top_k + s]];
        if (slot < 0) continue;
        sum += a.alpha[t * a.top_k + s] *
               warp_dot_f32(a.device_down, static_cast<std::int64_t>(slot) * a.hidden + r,
                            act_t + static_cast<std::int64_t>(s) * a.act_stride, a.intermediate);
    }
    if ((threadIdx.x & 31) == 0) a.partial[static_cast<std::int64_t>(t) * a.hidden + r] = sum;
}

// destination <- BF16(destination + device partial + host partial), one rounding.
__global__ void combine_kernel(__nv_bfloat16* destination, const float* partial,
                               const float* host_partial, std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    destination[i] = __float2bfloat16_rn(__bfloat162float(destination[i]) + partial[i] +
                                         host_partial[i]);
}

int act_stride(const HybridSparseMoeWeights& w) {
    const int width = w.intermediate > w.shared_intermediate ? w.intermediate : w.shared_intermediate;
    return (width + 63) / 64 * 64;
}

} // namespace

std::size_t hybrid_sparse_moe_workspace_bytes(const HybridSparseMoeWeights& w,
                                              std::int32_t max_tokens) {
    const std::size_t T = static_cast<std::size_t>(max_tokens);
    const std::size_t K = static_cast<std::size_t>(w.top_k);
    std::size_t bytes   = 0;
    const auto add      = [&](std::size_t n) { bytes += (n + 255) / 256 * 256; };
    add(T * K * sizeof(int));
    add(T * K * sizeof(float));
    add(T * sizeof(float));
    add(T * (K + 1) * static_cast<std::size_t>(act_stride(w)) * sizeof(float));
    add(T * static_cast<std::size_t>(w.hidden) * sizeof(float));
    return bytes;
}

void hybrid_sparse_moe(const Tensor& x, const HybridSparseMoeWeights& w,
                       HybridMoeHostRuntime& runtime, Tensor& destination,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    const int T = x.ne[1];
    if (x.dtype != DType::BF16 || destination.dtype != DType::BF16 || x.ne[0] != w.hidden ||
        destination.ne[0] != w.hidden || destination.ne[1] != T || T <= 0) {
        throw std::invalid_argument("hybrid sparse MoE: invalid activation geometry");
    }
    if (w.experts > kHybridMoeMaxExperts || w.top_k > kHybridMoeMaxTopK || w.hidden % 256 ||
        w.intermediate % 8 || w.shared_intermediate % 8) {
        throw std::invalid_argument("hybrid sparse MoE: unsupported geometry");
    }
    auto& host = runtime.impl();
    if (T > host.max_tokens) throw std::invalid_argument("hybrid sparse MoE: too many tokens");

    auto scope          = workspace.scope();
    const int K         = w.top_k;
    Tensor ids          = workspace.alloc(DType::I32, {K, T});
    Tensor alpha        = workspace.alloc(DType::FP32, {K, T});
    Tensor shared_scale = workspace.alloc(DType::FP32, {T});
    const int stride    = act_stride(w);
    Tensor act          = workspace.alloc(DType::FP32, {stride, (K + 1) * T});
    Tensor partial      = workspace.alloc(DType::FP32, {w.hidden, T});

    SlotTable slots{};
    for (int e = 0; e < kHybridMoeMaxExperts; ++e) {
        slots.slot[e] = e < w.experts ? w.slot_of_expert[static_cast<std::size_t>(e)] : -1;
    }
    const auto mailbox = host.device_mailbox();

    RouteArgs route{static_cast<const __nv_bfloat16*>(x.data),
                    static_cast<const __nv_bfloat16*>(w.router_shared_gate.qdata),
                    w.experts,
                    K,
                    w.hidden,
                    static_cast<int*>(ids.data),
                    static_cast<float*>(alpha.data),
                    static_cast<float*>(shared_scale.data),
                    mailbox.ids,
                    mailbox.alpha,
                    mailbox.x};
    route_kernel<<<T, kThreads, 0, stream>>>(route);
    check(cudaGetLastError(), "route launch");
    // Header words are baked into the captured graph as constants for this layer and T.
    check(cuStreamWriteValue32(stream, mailbox.layer_word, static_cast<cuuint32_t>(w.layer), 0),
          "publish layer");
    check(cuStreamWriteValue32(stream, mailbox.tokens_word, static_cast<cuuint32_t>(T), 0),
          "publish tokens");
    check(cuStreamWriteValue32(stream, mailbox.request_word, 1, 0), "publish request");

    const bool any_resident = w.device_gate_up.qdata != nullptr;
    ExpertArgs ea{static_cast<const __nv_bfloat16*>(x.data),
                  any_resident ? planes(w.device_gate_up) : planes(w.shared_gate_up),
                  planes(w.shared_gate_up),
                  slots,
                  static_cast<const int*>(ids.data),
                  K,
                  w.hidden,
                  w.intermediate,
                  w.shared_intermediate,
                  stride,
                  static_cast<float*>(act.data)};
    const int width = w.intermediate > w.shared_intermediate ? w.intermediate : w.shared_intermediate;
    expert_act_kernel<<<dim3((width + kWarps - 1) / kWarps, T * (K + 1)), kThreads, 0, stream>>>(
        ea);
    check(cudaGetLastError(), "expert act launch");
    DownArgs da{any_resident ? planes(w.device_down) : planes(w.shared_down),
                planes(w.shared_down),
                slots,
                static_cast<const int*>(ids.data),
                static_cast<const float*>(alpha.data),
                static_cast<const float*>(shared_scale.data),
                static_cast<const float*>(act.data),
                K,
                w.hidden,
                w.intermediate,
                w.shared_intermediate,
                stride,
                static_cast<float*>(partial.data)};
    expert_down_kernel<<<dim3((w.hidden + kWarps - 1) / kWarps, T), kThreads, 0, stream>>>(da);
    check(cudaGetLastError(), "expert down launch");

    check(cuStreamWaitValue32(stream, mailbox.done_word, 1, CU_STREAM_WAIT_VALUE_EQ),
          "wait for host experts");
    check(cuStreamWriteValue32(stream, mailbox.done_word, 0, 0), "reset host completion");
    const std::int64_t n = static_cast<std::int64_t>(T) * w.hidden;
    combine_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(destination.data), static_cast<const float*>(partial.data),
        mailbox.out, n);
    check(cudaGetLastError(), "combine launch");
}

} // namespace ninfer::ops
