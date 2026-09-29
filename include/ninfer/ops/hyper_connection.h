#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Qwen4-Exp hyper-connection pieces. A wide activation is BF16 [count*H, T]: stream s of token t
 * occupies rows [s*H, (s+1)*H). The complete gated residual is
 *
 *   xn     = grouped_rmsnorm(x, w_norm)                      (per stream, unit-offset weight)
 *   mix    = sigmoid(W_up SiLU((W_down xn) / count))          (the linears are ops::linear calls)
 *   input  = (1/count) sum_s mix_s * xn_s                     -> block input [H,T]
 *   inject = 2 * sigmoid((W_inject xn) / count)               -> per-stream weights [count,T]
 *   x      = x + inject_s * block_output                      (for every stream s)
 *
 * The oracle evaluates each formula naively in FP64 from the represented BF16 inputs; outputs are
 * compared after their single BF16 (or FP32 for weights) storage rounding.
 */

// out[s*H+h,t] = x * rsqrt(mean_h(x_s^2) + eps) * (1 + weight[s*H+h]).
void hc_grouped_rmsnorm(const Tensor& x, const Tensor& weight, std::int32_t group, float eps,
                        Tensor& out, cudaStream_t stream);

// y <- SiLU(y * scale), elementwise in place (BF16).
void hc_scaled_silu(Tensor& y, float scale, cudaStream_t stream);

// out[h,t] = (1/count) * sum_s sigmoid(gate[s*H+h,t]) * xn[s*H+h,t]; out BF16 [H,T].
void hc_gated_mean(const Tensor& gate, const Tensor& xn, std::int32_t count, Tensor& out,
                   cudaStream_t stream);

// weights[s,t] = 2 * sigmoid(inject[s,t] / count); inject BF16 [count,T], weights FP32 [count,T].
void hc_injection_weights(const Tensor& inject, std::int32_t count, Tensor& weights,
                          cudaStream_t stream);

// x[s*H+h,t] += block[h,t] * weights[s,t] with one BF16 rounding per element.
void hc_combine(Tensor& x, const Tensor& block, const Tensor& weights, cudaStream_t stream);

// x[s*H+h,t] = e[h,t] for every stream s.
void hc_expand(const Tensor& e, std::int32_t count, Tensor& x, cudaStream_t stream);

/**
 * Qwen4-Exp PLE gate: for stream s and token t,
 *   g      = sum_h key_n[s*H+h,t] * query_n[s*H+h,t] / sqrt(H)
 *   g      = sign(g) * sqrt(max(|g|, 1e-6))
 *   out[s*H+h,t] = sigmoid(g) * value[h,t]
 * key_n/query_n BF16 [count*H,T], value BF16 [H,T], out BF16 [count*H,T].
 */
void ple_gate(const Tensor& key_n, const Tensor& query_n, const Tensor& value, std::int32_t count,
              Tensor& out, cudaStream_t stream);

// PLE per-slot state inside caller storage: slot i starts at base + i * slot_pitch_bytes and holds
// the BF16 convolution history [hist, C] at conv_offset_bytes and 2*(ngram-1) I32 history words
// at history_offset_bytes. All-zero state means "no history": zero convolution padding and an
// eos token context.
struct PleStateView {
    void* base                        = nullptr;
    std::int64_t slot_pitch_bytes     = 0;
    std::int64_t conv_offset_bytes    = 0;
    std::int64_t history_offset_bytes = 0;
};

/**
 * Hashed n-gram row ids for W tokens of B sequences. ids I32 [W,B]; `valid` I32 [B] or empty.
 * The slot history holds, for each of the (ngram-1) previous tokens, a (present, token) I32 pair;
 * absent tokens read as eos. Row b reads slot src_slots[b] and writes the
 * updated history to dst_slots[b] after its valid columns. For token t with previous tokens
 * p1 (t-1), p2 (t-2), ...: shifted_k = eos when an eos occurs among p1..p(k-1), else p_k.
 * Head j of order n (n = 2..ngram) hashes (tok*m0) ^ (shifted_1*m1) ^ ... ^ (shifted_{n-1}*m_{n-1})
 * modulo its prime vocabulary and adds its offset. rows I32 [heads, W, B].
 */
struct PleHash {
    std::int32_t ngram_size      = 0;
    std::int32_t heads_per_ngram = 0;
    std::int32_t eos             = 0;
    std::uint64_t multipliers[4] = {};
    std::uint64_t vocab[32]      = {};
    std::uint64_t offset[32]     = {};
};

void ple_ngram_rows(const Tensor& ids, const Tensor& valid, const PleHash& hash,
                    const PleStateView& state, const Tensor& src_slots, const Tensor& dst_slots,
                    Tensor& rows, cudaStream_t stream);

/**
 * Dilated depthwise causal convolution with SiLU: x BF16 [C,W,B], weight BF16 [K,C], slot
 * history BF16 [hist,C] (hist = (K-1)*dilation, time-major),
 *   out[c,t] = SiLU(sum_k weight[k,c] * xcat[c, hist + t - (K-1-k)*dilation]),
 * where xcat is the slot history followed by x. The last `hist` columns of xcat after the valid
 * columns are written to dst_slots[b]. out BF16 [C,W,B].
 */
void ple_dilated_conv_silu(const Tensor& x, const Tensor& weight, std::int32_t dilation,
                           const Tensor& valid, const PleStateView& state,
                           const Tensor& src_slots, const Tensor& dst_slots, Tensor& out,
                           cudaStream_t stream);

} // namespace ninfer::ops
