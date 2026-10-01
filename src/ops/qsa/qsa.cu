#include "ninfer/ops/qsa.h"

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kIndexDim    = 128;
constexpr int kIndexHeads  = 4;
constexpr int kRotaryHalf  = 32;
constexpr int kCompress    = 4;
constexpr int kBlockTopk   = 512;
constexpr int kHeadDim     = 256;
constexpr int kQHeads      = 24;
constexpr int kKvHeads     = 2;
constexpr int kGroup       = kQHeads / kKvHeads;
constexpr int kPageTokens  = 64;
constexpr int kScoreBlocks = 64;  // blocks scored by one CTA
constexpr int kColumnChunk = 128; // columns scored per pass (bounds the score workspace)
constexpr int kTile        = 16;  // tokens per attention tile

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa: ") + message); }
}

struct InvFreq {
    float v[kRotaryHalf];
};

struct Plane {
    const char* base;
    std::int64_t nb0, nb1, nb2, nb3;
};

struct Rows {
    const std::int32_t* positions; // [W,B]
    const std::int32_t* valid;     // [B] or null
    const std::int32_t* table_rows;
    const std::int32_t* tables;
    std::int32_t table_stride;
    std::int32_t width;
};

Plane plane_of(const Tensor& t) {
    return Plane{static_cast<const char*>(t.data), t.nb[0], t.nb[1], t.nb[2], t.nb[3]};
}

__device__ __forceinline__ float round_bf16(float v) {
    return __bfloat162float(__float2bfloat16_rn(v));
}

__device__ __forceinline__ bool live_column(const Rows& r, int column) {
    const int b = column / r.width;
    return r.valid == nullptr || column % r.width < r.valid[b];
}

__device__ __forceinline__ const char* page_row(const Plane& plane, const Rows& r, int b, int pos,
                                                int head) {
    const int g = r.tables[static_cast<std::int64_t>(r.table_rows[b]) * r.table_stride +
                           pos / kPageTokens];
    return plane.base + static_cast<std::int64_t>(pos % kPageTokens) * plane.nb1 +
           static_cast<std::int64_t>(head) * plane.nb2 + static_cast<std::int64_t>(g) * plane.nb3;
}

// Last live position of row b, or -1 for an empty row.
__device__ __forceinline__ int last_position(const Rows& r, int b) {
    const int live = r.valid == nullptr ? r.width : r.valid[b];
    return live <= 0 ? -1 : r.positions[b * r.width + live - 1];
}

// Split-half NeoX RoPE of dimension d of a normalized head held in `x`, rounded to BF16.
__device__ __forceinline__ float rope_dim(const float* x, int d, int pos, const InvFreq& f) {
    if (d >= 2 * kRotaryHalf) { return x[d]; }
    const int pair = d % kRotaryHalf;
    float s;
    float c;
    sincosf(static_cast<float>(pos) * f.v[pair], &s, &c);
    const float v = d < kRotaryHalf ? x[d] * c - x[d + kRotaryHalf] * s
                                    : x[d] * c + x[d - kRotaryHalf] * s;
    return round_bf16(v);
}

// ---------------------------------------------------------------------------------------------
// grid (columns), block 128
__global__ void index_append_kernel(const __nv_bfloat16* __restrict__ key,
                                    std::int64_t key_stride, Rows r, Plane plane) {
    const int c = static_cast<int>(blockIdx.x);
    if (!live_column(r, c)) { return; }
    const int b   = c / r.width;
    const int d   = static_cast<int>(threadIdx.x);
    char* row     = const_cast<char*>(page_row(plane, r, b, r.positions[c], 0));
    *reinterpret_cast<__nv_bfloat16*>(row + d * plane.nb0) = key[c * key_stride + d];
}

// Block keys of every row that needs selection. grid (ceil(nbcap/8), B), block 256: warp = block.
__global__ void block_keys_kernel(Rows r, Plane plane, const __nv_bfloat16* __restrict__ key_norm,
                                  InvFreq f, float eps, int nbcap,
                                  __nv_bfloat16* __restrict__ keys) {
    const int b    = static_cast<int>(blockIdx.y);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int i    = static_cast<int>(blockIdx.x) * 8 + warp;
    const int nb   = (last_position(r, b) + 1) / kCompress;
    if (nb <= kBlockTopk || i >= nb) { return; }
    __shared__ float kn[8][kIndexDim];
    const char* page = page_row(plane, r, b, i * kCompress, 0);
    float u[4];
    float ss = 0.0F;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int d = lane * 4 + k;
        float sum   = 0.0F;
#pragma unroll
        for (int t = 0; t < kCompress; ++t) {
            sum += __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(page + t * plane.nb1 + d * plane.nb0));
        }
        u[k] = round_bf16(sum / static_cast<float>(kCompress));
        ss += u[k] * u[k];
    }
    ss              = warp_sum(ss);
    const float inv = rsqrtf(ss / static_cast<float>(kIndexDim) + eps);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int d  = lane * 4 + k;
        kn[warp][d] = round_bf16(u[k] * inv * (1.0F + __bfloat162float(key_norm[d])));
    }
    __syncwarp();
    __nv_bfloat16* out = keys + (static_cast<std::int64_t>(b) * nbcap + i) * kIndexDim;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int d = lane * 4 + k;
        out[d]      = __float2bfloat16_rn(rope_dim(kn[warp], d, i * kCompress, f));
    }
}

// grid (ceil(nbcap/kScoreBlocks), chunk columns), block 256.
__global__ void scores_kernel(const __nv_bfloat16* __restrict__ raw_query,
                              std::int64_t query_stride, const __nv_bfloat16* __restrict__ query_norm,
                              Rows r, InvFreq f, float eps, int nbcap, int column_begin,
                              const __nv_bfloat16* __restrict__ keys, float* __restrict__ scores) {
    const int local = static_cast<int>(blockIdx.y);
    const int c     = column_begin + local;
    if (!live_column(r, c)) { return; }
    const int b   = c / r.width;
    const int p   = r.positions[c];
    const int nb  = (p + 1) / kCompress;
    const int i0  = static_cast<int>(blockIdx.x) * kScoreBlocks;
    if (nb <= kBlockTopk || i0 >= nb) { return; }
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    __shared__ float normalized[kIndexHeads][kIndexDim];
    __shared__ float q[kIndexHeads][kIndexDim];
    if (warp < kIndexHeads) {
        const __nv_bfloat16* x = raw_query + c * query_stride + warp * kIndexDim;
        float v[4];
        float ss = 0.0F;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[k] = __bfloat162float(x[lane * 4 + k]);
            ss += v[k] * v[k];
        }
        ss              = warp_sum(ss);
        const float inv = rsqrtf(ss / static_cast<float>(kIndexDim) + eps);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int d          = lane * 4 + k;
            normalized[warp][d] = round_bf16(v[k] * inv * (1.0F + __bfloat162float(query_norm[d])));
        }
    }
    __syncthreads();
    for (int idx = static_cast<int>(threadIdx.x); idx < kIndexHeads * kIndexDim;
         idx += blockDim.x) {
        const int h = idx / kIndexDim;
        q[h][idx % kIndexDim] = rope_dim(normalized[h], idx % kIndexDim, p, f);
    }
    __syncthreads();
    const float inv_sqrt = rsqrtf(static_cast<float>(kIndexDim));
    for (int bi = warp; bi < kScoreBlocks; bi += 8) {
        const int i = i0 + bi;
        if (i >= nb) { break; }
        const __nv_bfloat16* k =
            keys + (static_cast<std::int64_t>(b) * nbcap + i) * kIndexDim + lane * 4;
        float dot[kIndexHeads] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float kv = __bfloat162float(k[e]);
#pragma unroll
            for (int h = 0; h < kIndexHeads; ++h) { dot[h] += kv * q[h][lane * 4 + e]; }
        }
        float total = 0.0F;
#pragma unroll
        for (int h = 0; h < kIndexHeads; ++h) { total += fmaxf(warp_sum(dot[h]), 0.0F); }
        if (lane == 0) { scores[static_cast<std::int64_t>(local) * nbcap + i] = total * inv_sqrt; }
    }
}

__device__ __forceinline__ std::uint32_t order_key(float v) {
    const std::uint32_t bits = __float_as_uint(v);
    return (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
}

// Exclusive prefix of `flag` over the 256-thread block plus the block total.
__device__ __forceinline__ int block_exclusive(bool flag, int* warp_totals, int* total) {
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int warp          = static_cast<int>(threadIdx.x) >> 5;
    const unsigned ballot   = __ballot_sync(kFullWarpMask, flag);
    const int within        = __popc(ballot & ((1U << lane) - 1U));
    if (lane == 0) { warp_totals[warp] = __popc(ballot); }
    __syncthreads();
    int before = 0;
    int sum    = 0;
    for (int w = 0; w < 8; ++w) {
        if (w < warp) { before += warp_totals[w]; }
        sum += warp_totals[w];
    }
    __syncthreads();
    *total = sum;
    return before + within;
}

// grid (chunk columns), block 256.
__global__ void topk_kernel(Rows r, int nbcap, int column_begin, const float* __restrict__ scores,
                            std::int32_t* __restrict__ selected) {
    const int local      = static_cast<int>(blockIdx.x);
    const int c          = column_begin + local;
    std::int32_t* out    = selected + static_cast<std::int64_t>(c) * kBlockTopk;
    const int tid        = static_cast<int>(threadIdx.x);
    const int nb         = live_column(r, c) ? (r.positions[c] + 1) / kCompress : 0;
    if (nb <= kBlockTopk) {
        for (int k = tid; k < kBlockTopk; k += blockDim.x) { out[k] = k < nb ? k : -1; }
        return;
    }
    const float* s = scores + static_cast<std::int64_t>(local) * nbcap;
    __shared__ int histogram[256];
    __shared__ std::uint32_t prefix_shared;
    __shared__ int remaining_shared;
    __shared__ int warp_totals[8];
    std::uint32_t prefix = 0;
    std::uint32_t mask   = 0;
    int remaining        = kBlockTopk;
    for (int shift = 24; shift >= 0; shift -= 8) {
        histogram[tid] = 0;
        __syncthreads();
        for (int i = tid; i < nb; i += blockDim.x) {
            const std::uint32_t key = order_key(s[i]);
            if ((key & mask) == prefix) { atomicAdd(&histogram[(key >> shift) & 255U], 1); }
        }
        __syncthreads();
        if (tid == 0) {
            int above = 0;
            int bin   = 255;
            for (; bin > 0; --bin) {
                if (above + histogram[bin] >= remaining) { break; }
                above += histogram[bin];
            }
            prefix_shared    = prefix | (static_cast<std::uint32_t>(bin) << shift);
            remaining_shared = remaining - above;
        }
        __syncthreads();
        prefix    = prefix_shared;
        remaining = remaining_shared;
        mask |= 255U << shift;
    }
    // `prefix` is the K-th largest key; take every larger key and the first `remaining` equal ones.
    int equal_seen = 0;
    int written    = 0;
    for (int base = 0; base < nb; base += blockDim.x) {
        const int i                = base + tid;
        const std::uint32_t key    = i < nb ? order_key(s[i]) : 0U;
        const bool greater         = i < nb && key > prefix;
        const bool equal           = i < nb && key == prefix;
        int equal_total            = 0;
        const int equal_rank       = equal_seen + block_exclusive(equal, warp_totals, &equal_total);
        const bool take            = greater || (equal && equal_rank < remaining);
        int take_total             = 0;
        const int slot             = written + block_exclusive(take, warp_totals, &take_total);
        if (take) { out[slot] = i; }
        equal_seen += equal_total;
        written += take_total;
    }
}

// grid (columns), block 256: append K bit for bit and V as FP16_RNE.
__global__ void kv_append_kernel(const __nv_bfloat16* __restrict__ k,
                                 const __nv_bfloat16* __restrict__ v, Rows r, Plane kp, Plane vp) {
    const int c = static_cast<int>(blockIdx.x);
    if (!live_column(r, c)) { return; }
    const int b   = c / r.width;
    const int pos = r.positions[c];
    const int d   = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int h = 0; h < kKvHeads; ++h) {
        const std::int64_t src = (static_cast<std::int64_t>(c) * kKvHeads + h) * kHeadDim + d;
        char* krow             = const_cast<char*>(page_row(kp, r, b, pos, h));
        char* vrow             = const_cast<char*>(page_row(vp, r, b, pos, h));
        *reinterpret_cast<__nv_bfloat16*>(krow + d * kp.nb0) = k[src];
        *reinterpret_cast<__half*>(vrow + d * vp.nb0) = __float2half_rn(__bfloat162float(v[src]));
    }
}

// Token splits per column: few columns (decode) spread their selected tokens over more CTAs.
int attention_splits(int columns) {
    const int ctas = columns * kKvHeads;
    return ctas >= 64 ? 1 : std::min(32, std::max(1, 64 / ctas));
}

// grid (columns, kv heads, splits), block 32*kGroup: warp = query head of the group. With one
// split the normalized output is written directly; otherwise each split writes its unnormalized
// accumulator and (max, sum) for combine_kernel.
__global__ void attention_kernel(const __nv_bfloat16* __restrict__ q, Rows r, Plane kp, Plane vp,
                                 const std::int32_t* __restrict__ selected, float scale_log2,
                                 int columns, __nv_bfloat16* __restrict__ out,
                                 float* __restrict__ part_acc, float2* __restrict__ part_ml) {
    const int c      = static_cast<int>(blockIdx.x);
    const int kvh    = static_cast<int>(blockIdx.y);
    const int split  = static_cast<int>(blockIdx.z);
    const int splits = static_cast<int>(gridDim.z);
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int head   = kvh * kGroup + warp;
    __nv_bfloat16* o = out + (static_cast<std::int64_t>(c) * kQHeads + head) * kHeadDim + lane * 8;
    if (!live_column(r, c)) {
        if (splits == 1) {
#pragma unroll
            for (int e = 0; e < 8; ++e) { o[e] = __float2bfloat16_rn(0.0F); }
        }
        return;
    }
    const int b      = c / r.width;
    const int p      = r.positions[c];
    const int nb     = (p + 1) / kCompress;
    const int chosen = min(nb, kBlockTopk);
    const int total  = chosen * kCompress + (p + 1 - nb * kCompress);
    const int per    = ((total + splits - 1) / splits + kTile - 1) / kTile * kTile;
    const int first  = split * per;
    const int count  = min(total, first + per);
    const std::int32_t* blocks = selected + static_cast<std::int64_t>(c) * kBlockTopk;

    float qv[8];
    const __nv_bfloat16* qh =
        q + (static_cast<std::int64_t>(c) * kQHeads + head) * kHeadDim + lane * 8;
#pragma unroll
    for (int e = 0; e < 8; ++e) { qv[e] = __bfloat162float(qh[e]) * scale_log2; }

    __shared__ __align__(16) __nv_bfloat16 ktile[kTile][kHeadDim];
    __shared__ __align__(16) __half vtile[kTile][kHeadDim];
    float m = -INFINITY;
    float l = 0.0F;
    float acc[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (int base = first; base < count; base += kTile) {
        __syncthreads();
        // 16 tokens x 256 dims, 8 elements (16 bytes) per vector.
        for (int idx = static_cast<int>(threadIdx.x); idx < kTile * (kHeadDim / 8);
             idx += blockDim.x) {
            const int tt = idx / (kHeadDim / 8);
            const int d  = (idx % (kHeadDim / 8)) * 8;
            const int t  = base + tt;
            uint4 kval   = make_uint4(0, 0, 0, 0);
            uint4 vval   = make_uint4(0, 0, 0, 0);
            if (t < count) {
                const int pos = t < chosen * kCompress
                                    ? blocks[t / kCompress] * kCompress + t % kCompress
                                    : nb * kCompress + (t - chosen * kCompress);
                kval = *reinterpret_cast<const uint4*>(page_row(kp, r, b, pos, kvh) + d * kp.nb0);
                vval = *reinterpret_cast<const uint4*>(page_row(vp, r, b, pos, kvh) + d * vp.nb0);
            }
            *reinterpret_cast<uint4*>(&ktile[tt][d]) = kval;
            *reinterpret_cast<uint4*>(&vtile[tt][d]) = vval;
        }
        __syncthreads();
        float s[kTile];
        float tile_max = -INFINITY;
#pragma unroll
        for (int tt = 0; tt < kTile; ++tt) {
            float partial = 0.0F;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                partial += qv[e] * __bfloat162float(ktile[tt][lane * 8 + e]);
            }
            partial = warp_sum(partial);
            s[tt]   = base + tt < count ? partial : -INFINITY;
            tile_max = fmaxf(tile_max, s[tt]);
        }
        const float m_new = fmaxf(m, tile_max);
        const float alpha = exp2f(m - m_new);
        l *= alpha;
#pragma unroll
        for (int e = 0; e < 8; ++e) { acc[e] *= alpha; }
#pragma unroll
        for (int tt = 0; tt < kTile; ++tt) {
            const float w = exp2f(s[tt] - m_new);
            l += w;
#pragma unroll
            for (int e = 0; e < 8; ++e) { acc[e] += w * __half2float(vtile[tt][lane * 8 + e]); }
        }
        m = m_new;
    }
    if (splits == 1) {
        const float inv = 1.0F / l;
#pragma unroll
        for (int e = 0; e < 8; ++e) { o[e] = __float2bfloat16_rn(acc[e] * inv); }
        return;
    }
    const std::int64_t slot = (static_cast<std::int64_t>(split) * columns + c) * kQHeads + head;
#pragma unroll
    for (int e = 0; e < 8; ++e) { part_acc[slot * kHeadDim + lane * 8 + e] = acc[e]; }
    if (lane == 0) { part_ml[slot] = make_float2(m, l); }
}

// grid (columns, q heads), block kHeadDim.
__global__ void combine_kernel(Rows r, int columns, int splits, const float* __restrict__ part_acc,
                               const float2* __restrict__ part_ml,
                               __nv_bfloat16* __restrict__ out) {
    const int c    = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int d    = static_cast<int>(threadIdx.x);
    float value    = 0.0F;
    if (live_column(r, c)) {
        float m_max = -INFINITY;
        for (int s = 0; s < splits; ++s) {
            const float2 ml = part_ml[(static_cast<std::int64_t>(s) * columns + c) * kQHeads + head];
            if (ml.y != 0.0F) { m_max = fmaxf(m_max, ml.x); }
        }
        float num = 0.0F;
        float den = 0.0F;
        for (int s = 0; s < splits; ++s) {
            const std::int64_t slot = (static_cast<std::int64_t>(s) * columns + c) * kQHeads + head;
            const float2 ml         = part_ml[slot];
            if (ml.y == 0.0F) { continue; } // empty split
            const float w = exp2f(ml.x - m_max);
            num += w * part_acc[slot * kHeadDim + d];
            den += w * ml.y;
        }
        value = num / den;
    }
    out[(static_cast<std::int64_t>(c) * kQHeads + head) * kHeadDim + d] = __float2bfloat16_rn(value);
}

void check_geometry(const QsaGeometry& g) {
    require(g.index_heads == kIndexHeads && g.index_dim == kIndexDim &&
                g.rotary_dim == 2 * kRotaryHalf && g.compress == kCompress &&
                g.block_topk == kBlockTopk,
            "unregistered geometry");
    require(std::isfinite(g.theta) && g.theta > 0.0F && std::isfinite(g.eps) && g.eps > 0.0F,
            "theta and eps must be positive and finite");
}

InvFreq inv_freq(const QsaGeometry& g) {
    InvFreq f{};
    for (int i = 0; i < kRotaryHalf; ++i) {
        f.v[i] = static_cast<float>(
            std::pow(static_cast<double>(g.theta), -2.0 * i / static_cast<double>(g.rotary_dim)));
    }
    return f;
}

Rows rows_of(const Tensor& positions, const Tensor& valid, const Tensor& table_rows,
             const Tensor& block_tables) {
    require(positions.dtype == DType::I32 && positions.is_contiguous() && positions.ne[2] == 1 &&
                positions.ne[3] == 1,
            "positions must be contiguous I32 [W,B]");
    const int batch = positions.ne[1];
    require(table_rows.dtype == DType::I32 && table_rows.numel() == batch,
            "kv_table_rows must be I32 [B]");
    require(valid.data == nullptr || (valid.dtype == DType::I32 && valid.numel() == batch),
            "valid_columns must be empty or I32 [B]");
    require(block_tables.dtype == DType::I32 && block_tables.nb[0] == 4,
            "block tables must be I32 [L,rows]");
    return Rows{static_cast<const std::int32_t*>(positions.data),
                static_cast<const std::int32_t*>(valid.data),
                static_cast<const std::int32_t*>(table_rows.data),
                static_cast<const std::int32_t*>(block_tables.data), block_tables.ne[0],
                positions.ne[0]};
}

void require_plane(const Tensor& plane, DType dtype, int leading, int heads, const char* message) {
    require(plane.dtype == dtype && plane.ne[0] == leading && plane.ne[1] == kPageTokens &&
                plane.ne[2] == heads && plane.nb[0] == static_cast<std::int64_t>(dtype_size(dtype)),
            message);
}

std::int64_t column_stride(const Tensor& x, int rows, int columns, const char* message) {
    require(x.dtype == DType::BF16 && x.ne[0] == rows && x.ne[1] == columns && x.ne[2] == 1 &&
                x.ne[3] == 1 && x.nb[0] == 2 && x.nb[1] % 2 == 0 && x.nb[1] >= 2 * rows,
            message);
    return x.nb[1] / 2;
}

} // namespace

void qsa_index_append(const Tensor& key, const Tensor& positions, const Tensor& valid_columns,
                      const Tensor& kv_table_rows, const Tensor& block_tables, Tensor& pages,
                      cudaStream_t stream) {
    const Rows r        = rows_of(positions, valid_columns, kv_table_rows, block_tables);
    const int columns   = positions.ne[0] * positions.ne[1];
    const auto stride   = column_stride(key, kIndexDim, columns, "key must be BF16 [128,W*B]");
    require_plane(pages, DType::BF16, kIndexDim, 1, "index plane must be BF16 [128,64,1,N]");
    if (columns == 0) { return; }
    index_append_kernel<<<columns, kIndexDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(key.data), stride, r, plane_of(pages));
    check_launch("qsa_index_append");
}

std::size_t qsa_select_workspace_bytes(const QsaGeometry& geometry, std::uint32_t max_visible,
                                       std::int32_t rows, std::int32_t columns) {
    check_geometry(geometry);
    require(max_visible > 0 && rows > 0 && columns >= rows, "invalid workspace envelope");
    const std::size_t nbcap = std::max<std::size_t>(max_visible / kCompress, 1);
    // Block keys of every row plus one chunk of scores, each 256-byte aligned.
    const std::size_t keys   = static_cast<std::size_t>(rows) * nbcap * kIndexDim * 2;
    const std::size_t scores = static_cast<std::size_t>(std::min(columns, kColumnChunk)) * nbcap * 4;
    return ((keys + 255) / 256 + (scores + 255) / 256) * 256 + 512;
}

void qsa_select(const Tensor& raw_query, const Tensor& query_norm, const Tensor& key_norm,
                const Tensor& positions, const Tensor& valid_columns, const Tensor& kv_table_rows,
                const Tensor& block_tables, const Tensor& pages, const QsaGeometry& geometry,
                std::uint32_t max_visible, WorkspaceArena& workspace, Tensor& selected,
                cudaStream_t stream) {
    check_geometry(geometry);
    const Rows r      = rows_of(positions, valid_columns, kv_table_rows, block_tables);
    const int width   = positions.ne[0];
    const int batch   = positions.ne[1];
    const int columns = width * batch;
    const auto stride = column_stride(raw_query, kIndexHeads * kIndexDim, columns,
                                      "raw_query must be BF16 [512,W*B]");
    require(query_norm.dtype == DType::BF16 && query_norm.numel() == kIndexDim &&
                key_norm.dtype == DType::BF16 && key_norm.numel() == kIndexDim,
            "norm weights must be BF16 [128]");
    require_plane(pages, DType::BF16, kIndexDim, 1, "index plane must be BF16 [128,64,1,N]");
    require(selected.dtype == DType::I32 && selected.is_contiguous() &&
                selected.ne[0] == kBlockTopk && selected.ne[1] == width && selected.ne[2] == batch,
            "selected must be contiguous I32 [512,W,B]");
    require(max_visible > 0, "max_visible must be positive");
    if (columns == 0) { return; }
    // The launch sequence depends only on the column count so that CUDA Graphs captured for
    // different visibility envelopes share one topology; columns that see at most K complete
    // blocks skip scoring inside the kernels and select every block.
    const int nbcap = std::max(static_cast<int>(max_visible / kCompress), 1);
    const InvFreq f = inv_freq(geometry);
    auto scope      = workspace.scope();
    auto* keys      = static_cast<__nv_bfloat16*>(
        workspace.alloc_bytes(static_cast<std::size_t>(batch) * nbcap * kIndexDim * 2).data);
    block_keys_kernel<<<dim3((nbcap + 7) / 8, batch), 256, 0, stream>>>(
        r, plane_of(pages), static_cast<const __nv_bfloat16*>(key_norm.data), f, geometry.eps,
        nbcap, keys);
    check_launch("qsa_select block keys");
    auto* scores = static_cast<float*>(
        workspace.alloc_bytes(static_cast<std::size_t>(std::min(columns, kColumnChunk)) * nbcap * 4)
            .data);
    for (int begin = 0; begin < columns; begin += kColumnChunk) {
        const int chunk = std::min(kColumnChunk, columns - begin);
        scores_kernel<<<dim3((nbcap + kScoreBlocks - 1) / kScoreBlocks, chunk), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(raw_query.data), stride,
            static_cast<const __nv_bfloat16*>(query_norm.data), r, f, geometry.eps, nbcap, begin,
            keys, scores);
        check_launch("qsa_select scores");
        topk_kernel<<<chunk, 256, 0, stream>>>(r, nbcap, begin, scores,
                                               static_cast<std::int32_t*>(selected.data));
        check_launch("qsa_select top-k");
    }
}

std::size_t qsa_attention_workspace_bytes(std::int32_t columns) {
    require(columns > 0, "invalid workspace envelope");
    const int splits = attention_splits(columns);
    if (splits == 1) { return 0; }
    const std::size_t slots = static_cast<std::size_t>(splits) * columns * kQHeads;
    return ((slots * kHeadDim * 4 + 255) / 256 + (slots * 8 + 255) / 256) * 256 + 512;
}

void qsa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   const Tensor& valid_columns, const Tensor& kv_table_rows,
                   const Tensor& selected, const QsaGeometry& geometry, float scale,
                   PagedKVBatchLayerView cache, WorkspaceArena& workspace, Tensor& out,
                   cudaStream_t stream) {
    check_geometry(geometry);
    const Rows r      = rows_of(positions, valid_columns, kv_table_rows, cache.block_tables);
    const int width   = positions.ne[0];
    const int batch   = positions.ne[1];
    const int columns = width * batch;
    const auto shaped = [&](const Tensor& t, int heads) {
        return t.dtype == DType::BF16 && t.is_contiguous() && t.ne[0] == kHeadDim &&
               t.ne[1] == heads && t.ne[2] == width && t.ne[3] == batch;
    };
    require(shaped(q, kQHeads) && shaped(out, kQHeads), "q/out must be BF16 [256,24,W,B]");
    require(shaped(k, kKvHeads) && shaped(v, kKvHeads), "k/v must be BF16 [256,2,W,B]");
    require(cache.storage == KvCacheStorage::BFloat16 && cache.head_dim == kHeadDim &&
                cache.num_kv_heads == kKvHeads,
            "cache must use the BF16 profile with [256,2] heads");
    require_plane(cache.k_pages, DType::BF16, kHeadDim, kKvHeads, "K plane must be BF16");
    require_plane(cache.v_pages, DType::FP16, kHeadDim, kKvHeads, "V plane must be FP16");
    require(selected.dtype == DType::I32 && selected.is_contiguous() &&
                selected.ne[0] == kBlockTopk && selected.ne[1] == width && selected.ne[2] == batch,
            "selected must be contiguous I32 [512,W,B]");
    require(std::isfinite(scale) && scale > 0.0F, "scale must be positive and finite");
    if (columns == 0) { return; }
    kv_append_kernel<<<columns, kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data), r,
        plane_of(cache.k_pages), plane_of(cache.v_pages));
    check_launch("qsa_attention append");
    const int splits = attention_splits(columns);
    auto scope       = workspace.scope();
    float* part_acc  = nullptr;
    float2* part_ml  = nullptr;
    if (splits > 1) {
        const std::size_t slots = static_cast<std::size_t>(splits) * columns * kQHeads;
        part_acc = static_cast<float*>(workspace.alloc_bytes(slots * kHeadDim * 4).data);
        part_ml  = static_cast<float2*>(workspace.alloc_bytes(slots * 8).data);
    }
    attention_kernel<<<dim3(columns, kKvHeads, splits), 32 * kGroup, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), r, plane_of(cache.k_pages),
        plane_of(cache.v_pages), static_cast<const std::int32_t*>(selected.data),
        scale * 1.4426950408889634F, columns, static_cast<__nv_bfloat16*>(out.data), part_acc,
        part_ml);
    check_launch("qsa_attention");
    if (splits > 1) {
        combine_kernel<<<dim3(columns, kQHeads), kHeadDim, 0, stream>>>(
            r, columns, splits, part_acc, part_ml, static_cast<__nv_bfloat16*>(out.data));
        check_launch("qsa_attention combine");
    }
}

} // namespace ninfer::ops
