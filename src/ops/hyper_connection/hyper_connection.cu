#include "ninfer/ops/hyper_connection.h"

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kThreads = 256;

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

__device__ __forceinline__ float sigmoidf(float v) { return 1.0F / (1.0F + __expf(-v)); }

__device__ float block_sum(float v) {
    __shared__ float partial[kThreads / 32];
    v = warp_reduce_sum(v);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    float total = 0.0F;
    if (warp == 0) {
        total = lane < kThreads / 32 ? partial[lane] : 0.0F;
        total = warp_reduce_sum(total);
        if (lane == 0) partial[0] = total;
    }
    __syncthreads();
    total = partial[0];
    __syncthreads();
    return total;
}

// grid (count, T)
__global__ void grouped_rmsnorm_kernel(const __nv_bfloat16* __restrict__ x,
                                       const __nv_bfloat16* __restrict__ w,
                                       __nv_bfloat16* __restrict__ out, int group, int width,
                                       float eps) {
    const int s              = static_cast<int>(blockIdx.x);
    const std::int64_t base  = static_cast<std::int64_t>(blockIdx.y) * width + s * group;
    float sum                = 0.0F;
    for (int h = static_cast<int>(threadIdx.x); h < group; h += kThreads) {
        const float v = __bfloat162float(x[base + h]);
        sum += v * v;
    }
    const float inv = rsqrtf(block_sum(sum) / static_cast<float>(group) + eps);
    for (int h = static_cast<int>(threadIdx.x); h < group; h += kThreads) {
        const float v = __bfloat162float(x[base + h]);
        out[base + h] = __float2bfloat16_rn(v * inv * (1.0F + __bfloat162float(w[s * group + h])));
    }
}

__global__ void scaled_silu_kernel(__nv_bfloat16* y, float scale, std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= n) return;
    const float v = __bfloat162float(y[i]) * scale;
    y[i]          = __float2bfloat16_rn(v * sigmoidf(v));
}

// out [H,T]
__global__ void gated_mean_kernel(const __nv_bfloat16* __restrict__ gate,
                                  const __nv_bfloat16* __restrict__ xn,
                                  __nv_bfloat16* __restrict__ out, int group, int count,
                                  std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= n) return;
    const std::int64_t t = i / group;
    const int h          = static_cast<int>(i - t * group);
    const std::int64_t base = t * group * count + h;
    float sum               = 0.0F;
    for (int s = 0; s < count; ++s) {
        const std::int64_t j = base + static_cast<std::int64_t>(s) * group;
        sum += sigmoidf(__bfloat162float(gate[j])) * __bfloat162float(xn[j]);
    }
    out[i] = __float2bfloat16_rn(sum / static_cast<float>(count));
}

__global__ void injection_weights_kernel(const __nv_bfloat16* inject, float* weights, int count,
                                         std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= n) return;
    weights[i] = 2.0F * sigmoidf(__bfloat162float(inject[i]) / static_cast<float>(count));
}

__global__ void combine_kernel(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ block,
                               const float* __restrict__ weights, int group, int count,
                               std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= n) return;
    const std::int64_t width = static_cast<std::int64_t>(group) * count;
    const std::int64_t t     = i / width;
    const std::int64_t r     = i - t * width;
    const int s              = static_cast<int>(r / group);
    const int h              = static_cast<int>(r - static_cast<std::int64_t>(s) * group);
    x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) +
                               __bfloat162float(block[t * group + h]) * weights[t * count + s]);
}

__global__ void expand_kernel(const __nv_bfloat16* __restrict__ e, __nv_bfloat16* __restrict__ x,
                              int group, int count, std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= n) return;
    const std::int64_t width = static_cast<std::int64_t>(group) * count;
    const std::int64_t t     = i / width;
    const int h              = static_cast<int>((i - t * width) % group);
    x[i]                     = e[t * group + h];
}

// grid (count, T)
__global__ void ple_gate_kernel(const __nv_bfloat16* __restrict__ key,
                                const __nv_bfloat16* __restrict__ query,
                                const __nv_bfloat16* __restrict__ value,
                                __nv_bfloat16* __restrict__ out, int group, int count) {
    const int s              = static_cast<int>(blockIdx.x);
    const std::int64_t t     = blockIdx.y;
    const std::int64_t base  = t * group * count + static_cast<std::int64_t>(s) * group;
    float dot                = 0.0F;
    for (int h = static_cast<int>(threadIdx.x); h < group; h += kThreads) {
        dot += __bfloat162float(key[base + h]) * __bfloat162float(query[base + h]);
    }
    float g = block_sum(dot) / sqrtf(static_cast<float>(group));
    g       = copysignf(sqrtf(fmaxf(fabsf(g), 1e-6F)), g);
    const float gate = sigmoidf(g);
    for (int h = static_cast<int>(threadIdx.x); h < group; h += kThreads) {
        out[base + h] = __float2bfloat16_rn(gate * __bfloat162float(value[t * group + h]));
    }
}

__device__ __forceinline__ std::int32_t* history_words(const PleStateView& st, int slot) {
    return reinterpret_cast<std::int32_t*>(static_cast<char*>(st.base) +
                                           static_cast<std::int64_t>(slot) * st.slot_pitch_bytes +
                                           st.history_offset_bytes);
}

// One thread per sequence: token order within a sequence is sequential.
__global__ void ngram_rows_kernel(const std::int32_t* __restrict__ ids,
                                  const std::int32_t* __restrict__ valid, PleHash hash,
                                  PleStateView state, const std::int32_t* __restrict__ src,
                                  const std::int32_t* __restrict__ dst,
                                  std::int32_t* __restrict__ rows, int width, int batch) {
    const int b = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const int context = hash.ngram_size - 1;
    // prev[0] is the token immediately before the current one.
    std::int64_t prev[4];
    const std::int32_t* in = history_words(state, src[b]);
    for (int k = 0; k < context; ++k) prev[k] = in[2 * k] ? in[2 * k + 1] : hash.eos;
    const int heads = context * hash.heads_per_ngram;
    const int count = valid ? valid[b] : width;
    for (int t = 0; t < width; ++t) {
        const std::int64_t token = ids[static_cast<std::int64_t>(b) * width + t];
        // shifted[k] for k = 1..context: eos when an eos occurs among prev[0..k-2].
        std::uint64_t shifted[4];
        bool eos_seen = false;
        for (int k = 0; k < context; ++k) {
            shifted[k] = static_cast<std::uint64_t>(eos_seen ? hash.eos : prev[k]);
            eos_seen   = eos_seen || prev[k] == hash.eos;
        }
        std::uint64_t mixed = static_cast<std::uint64_t>(token) * hash.multipliers[0];
        for (int order = 2; order <= hash.ngram_size; ++order) {
            mixed ^= shifted[order - 2] * hash.multipliers[order - 1];
            for (int j = 0; j < hash.heads_per_ngram; ++j) {
                const int head = (order - 2) * hash.heads_per_ngram + j;
                rows[(static_cast<std::int64_t>(b) * width + t) * heads + head] =
                    static_cast<std::int32_t>(mixed % hash.vocab[head] + hash.offset[head]);
            }
        }
        if (t < count) {
            for (int k = context - 1; k > 0; --k) prev[k] = prev[k - 1];
            prev[0] = token;
        }
    }
    // An absent entry reads as eos, so storing (present, eos) for it is equivalent. The history was
    // fully read above, which makes src == dst safe. No dst: record mode, state is not written.
    if (dst == nullptr) return;
    std::int32_t* out = history_words(state, dst[b]);
    for (int k = 0; k < context; ++k) {
        out[2 * k]     = 1;
        out[2 * k + 1] = static_cast<std::int32_t>(prev[k]);
    }
}

constexpr int kMaxHistory = 16;

// grid (ceil(C/kThreads), B); one thread per channel
__global__ void dilated_conv_silu_kernel(const __nv_bfloat16* __restrict__ x,
                                         const __nv_bfloat16* __restrict__ weight,
                                         const std::int32_t* __restrict__ valid, PleStateView state,
                                         const std::int32_t* __restrict__ src,
                                         const std::int32_t* __restrict__ dst,
                                         __nv_bfloat16* __restrict__ out, int channels, int width,
                                         int taps, int dilation, int hist) {
    const int c = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
    const int b = static_cast<int>(blockIdx.y);
    if (c >= channels) return;
    const auto conv = [&](int slot) {
        return reinterpret_cast<__nv_bfloat16*>(static_cast<char*>(state.base) +
                                                static_cast<std::int64_t>(slot) *
                                                    state.slot_pitch_bytes +
                                                state.conv_offset_bytes);
    };
    float old[kMaxHistory];
    const __nv_bfloat16* in = conv(src[b]);
    for (int j = 0; j < hist; ++j) old[j] = __bfloat162float(in[static_cast<std::int64_t>(j) * channels + c]);
    const std::int64_t xb = static_cast<std::int64_t>(b) * width;
    const auto xcat = [&](int index) -> float {
        return index < hist ? old[index]
                            : __bfloat162float(x[(xb + index - hist) * channels + c]);
    };
    for (int t = 0; t < width; ++t) {
        float acc = 0.0F;
        for (int k = 0; k < taps; ++k) {
            acc += __bfloat162float(weight[static_cast<std::int64_t>(k) * channels + c]) *
                   xcat(hist + t - (taps - 1 - k) * dilation);
        }
        out[(xb + t) * channels + c] = __float2bfloat16_rn(acc * sigmoidf(acc));
    }
    if (dst == nullptr) return; // record mode: the caller keeps the inputs for a later fold
    const int count     = valid ? valid[b] : width;
    __nv_bfloat16* next = conv(dst[b]);
    for (int j = 0; j < hist; ++j) {
        next[static_cast<std::int64_t>(j) * channels + c] = __float2bfloat16_rn(xcat(count + j));
    }
}

unsigned blocks(std::int64_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

struct PleFoldRows {
    PleReplayFoldRow row[8];
};

// grid (ceil(C/kThreads), rows); one thread per channel; channel 0 also folds the token history.
__global__ void ple_replay_fold_kernel(const __nv_bfloat16* __restrict__ conv_record,
                                       const std::int32_t* __restrict__ id_record, PleStateView state,
                                       PleFoldRows rows, int channels, int width, int hist,
                                       int context) {
    const int c                = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
    const int r                = static_cast<int>(blockIdx.y);
    const PleReplayFoldRow row = rows.row[r];
    const int commit           = row.commit_columns;
    if (c >= channels || commit <= 0) return;
    const auto slot = [&](int index) {
        return static_cast<char*>(state.base) + static_cast<std::int64_t>(index) * state.slot_pitch_bytes;
    };
    // tail_hist(source history || records[0:commit]); everything is read before any write.
    const auto* in = reinterpret_cast<const __nv_bfloat16*>(slot(row.source_slot) + state.conv_offset_bytes);
    __nv_bfloat16 next[kMaxHistory];
    for (int j = 0; j < hist; ++j) {
        const int index = commit + j;
        next[j] = index < hist ? in[static_cast<std::int64_t>(index) * channels + c]
                               : conv_record[(static_cast<std::int64_t>(r) * width + index - hist) *
                                                 channels + c];
    }
    auto* out = reinterpret_cast<__nv_bfloat16*>(slot(row.destination_slot) + state.conv_offset_bytes);
    for (int j = 0; j < hist; ++j) out[static_cast<std::int64_t>(j) * channels + c] = next[j];
    if (c != 0) return;
    const auto* words = reinterpret_cast<const std::int32_t*>(slot(row.source_slot) +
                                                              state.history_offset_bytes);
    std::int32_t present[4];
    std::int32_t token[4];
    for (int k = 0; k < context; ++k) {
        present[k] = words[2 * k];
        token[k]   = words[2 * k + 1];
    }
    for (int t = 0; t < commit; ++t) {
        for (int k = context - 1; k > 0; --k) {
            present[k] = present[k - 1];
            token[k]   = token[k - 1];
        }
        present[0] = 1;
        token[0]   = id_record[static_cast<std::int64_t>(r) * width + t];
    }
    auto* dst = reinterpret_cast<std::int32_t*>(slot(row.destination_slot) + state.history_offset_bytes);
    for (int k = 0; k < context; ++k) {
        dst[2 * k]     = present[k];
        dst[2 * k + 1] = token[k];
    }
}

} // namespace

void hc_grouped_rmsnorm(const Tensor& x, const Tensor& weight, std::int32_t group, float eps,
                        Tensor& out, cudaStream_t stream) {
    const int width = x.ne[0];
    require(group > 0 && width % group == 0 && weight.ne[0] == width && out.ne[0] == width &&
                out.ne[1] == x.ne[1],
            "hc_grouped_rmsnorm: invalid geometry");
    grouped_rmsnorm_kernel<<<dim3(width / group, x.ne[1]), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<__nv_bfloat16*>(out.data), group, width, eps);
    check_launch("hc_grouped_rmsnorm");
}

void hc_scaled_silu(Tensor& y, float scale, cudaStream_t stream) {
    const std::int64_t n = y.numel();
    scaled_silu_kernel<<<blocks(n), kThreads, 0, stream>>>(static_cast<__nv_bfloat16*>(y.data),
                                                            scale, n);
    check_launch("hc_scaled_silu");
}

void hc_gated_mean(const Tensor& gate, const Tensor& xn, std::int32_t count, Tensor& out,
                   cudaStream_t stream) {
    const int group = out.ne[0];
    require(gate.ne[0] == group * count && xn.ne[0] == group * count && gate.ne[1] == out.ne[1],
            "hc_gated_mean: invalid geometry");
    const std::int64_t n = out.numel();
    gated_mean_kernel<<<blocks(n), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.data), static_cast<const __nv_bfloat16*>(xn.data),
        static_cast<__nv_bfloat16*>(out.data), group, count, n);
    check_launch("hc_gated_mean");
}

void hc_injection_weights(const Tensor& inject, std::int32_t count, Tensor& weights,
                          cudaStream_t stream) {
    require(inject.ne[0] == count && weights.ne[0] == count && weights.dtype == DType::FP32,
            "hc_injection_weights: invalid geometry");
    const std::int64_t n = inject.numel();
    injection_weights_kernel<<<blocks(n), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(inject.data), static_cast<float*>(weights.data), count, n);
    check_launch("hc_injection_weights");
}

void hc_combine(Tensor& x, const Tensor& block, const Tensor& weights, cudaStream_t stream) {
    const int group = block.ne[0];
    const int count = weights.ne[0];
    require(x.ne[0] == group * count && x.ne[1] == block.ne[1] && weights.dtype == DType::FP32,
            "hc_combine: invalid geometry");
    const std::int64_t n = x.numel();
    combine_kernel<<<blocks(n), kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(block.data),
        static_cast<const float*>(weights.data), group, count, n);
    check_launch("hc_combine");
}

void hc_expand(const Tensor& e, std::int32_t count, Tensor& x, cudaStream_t stream) {
    const int group = e.ne[0];
    require(x.ne[0] == group * count && x.ne[1] == e.ne[1], "hc_expand: invalid geometry");
    const std::int64_t n = x.numel();
    expand_kernel<<<blocks(n), kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(e.data),
                                                       static_cast<__nv_bfloat16*>(x.data), group,
                                                       count, n);
    check_launch("hc_expand");
}

void ple_gate(const Tensor& key_n, const Tensor& query_n, const Tensor& value, std::int32_t count,
              Tensor& out, cudaStream_t stream) {
    const int group = value.ne[0];
    require(key_n.ne[0] == group * count && query_n.ne[0] == group * count &&
                out.ne[0] == group * count && key_n.ne[1] == value.ne[1],
            "ple_gate: invalid geometry");
    ple_gate_kernel<<<dim3(count, value.ne[1]), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(key_n.data),
        static_cast<const __nv_bfloat16*>(query_n.data),
        static_cast<const __nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(out.data), group,
        count);
    check_launch("ple_gate");
}

void ple_ngram_rows(const Tensor& ids, const Tensor& valid, const PleHash& hash,
                    const PleStateView& state, const Tensor& src_slots, const Tensor& dst_slots,
                    Tensor& rows, cudaStream_t stream) {
    const int width   = ids.ne[0];
    const int batch   = ids.ne[1];
    const int context = hash.ngram_size - 1;
    require(context >= 1 && context <= 4 && hash.heads_per_ngram > 0 &&
                context * hash.heads_per_ngram <= 32 &&
                rows.ne[0] == context * hash.heads_per_ngram && rows.ne[1] == width &&
                src_slots.ne[0] == batch && (!dst_slots.data || dst_slots.ne[0] == batch),
            "ple_ngram_rows: invalid geometry");
    ngram_rows_kernel<<<(batch + 63) / 64, 64, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data),
        valid.data ? static_cast<const std::int32_t*>(valid.data) : nullptr, hash, state,
        static_cast<const std::int32_t*>(src_slots.data),
        static_cast<const std::int32_t*>(dst_slots.data), static_cast<std::int32_t*>(rows.data),
        width, batch);
    check_launch("ple_ngram_rows");
}

void ple_dilated_conv_silu(const Tensor& x, const Tensor& weight, std::int32_t dilation,
                           const Tensor& valid, const PleStateView& state,
                           const Tensor& src_slots, const Tensor& dst_slots, Tensor& out,
                           cudaStream_t stream) {
    const int channels = x.ne[0];
    const int width    = x.ne[1];
    const int batch    = x.ne[2];
    const int taps     = weight.ne[1];
    const int hist     = (taps - 1) * dilation;
    require(weight.ne[0] == channels && hist <= kMaxHistory && out.ne[0] == channels &&
                src_slots.ne[0] == batch && (!dst_slots.data || dst_slots.ne[0] == batch),
            "ple_dilated_conv_silu: invalid geometry");
    dilated_conv_silu_kernel<<<dim3(blocks(channels), batch), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        valid.data ? static_cast<const std::int32_t*>(valid.data) : nullptr, state,
        static_cast<const std::int32_t*>(src_slots.data),
        static_cast<const std::int32_t*>(dst_slots.data), static_cast<__nv_bfloat16*>(out.data),
        channels, width, taps, dilation, hist);
    check_launch("ple_dilated_conv_silu");
}

void ple_replay_fold(const Tensor& conv_record, const Tensor& id_record, std::int32_t taps,
                     std::int32_t dilation, std::int32_t ngram_size, const PleStateView& state,
                     std::span<const PleReplayFoldRow> rows, cudaStream_t stream) {
    const int channels = conv_record.ne[0];
    const int width    = conv_record.ne[1];
    const int hist     = (taps - 1) * dilation;
    const int context  = ngram_size - 1;
    require(conv_record.dtype == DType::BF16 && id_record.dtype == DType::I32 &&
                id_record.ne[0] == width && id_record.ne[1] == conv_record.ne[2] &&
                hist > 0 && hist <= kMaxHistory && context >= 1 && context <= 4 &&
                !rows.empty() && rows.size() <= 8 &&
                static_cast<std::int64_t>(rows.size()) <= conv_record.ne[2],
            "ple_replay_fold: invalid geometry");
    PleFoldRows packed{};
    for (std::size_t i = 0; i < rows.size(); ++i) {
        require(rows[i].commit_columns >= 0 && rows[i].commit_columns <= width,
                "ple_replay_fold: commit extent outside the record width");
        packed.row[i] = rows[i];
    }
    ple_replay_fold_kernel<<<dim3(blocks(channels), static_cast<unsigned>(rows.size())), kThreads,
                             0, stream>>>(static_cast<const __nv_bfloat16*>(conv_record.data),
                                          static_cast<const std::int32_t*>(id_record.data), state,
                                          packed, channels, width, hist, context);
    check_launch("ple_replay_fold");
}

} // namespace ninfer::ops
