#include "ops/sparse_moe/host/host_rowsplit_dot.h"

#include <immintrin.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops::host {
namespace {

constexpr std::int32_t kKAlign = 128;

struct Codec {
    std::int32_t group;
    std::int32_t code_bytes;
    std::int32_t high_bytes;
};

Codec codec(QType qtype) {
    switch (qtype) {
    case QType::Q4_G64_FP16:
        return {64, 32, 0};
    case QType::Q5_G64_FP16:
        return {64, 32, 8};
    case QType::Q6_G64_FP16:
        return {64, 32, 16};
    case QType::Q8_G32_FP16:
        return {32, 32, 0};
    default:
        throw std::invalid_argument("host row-split dot: unsupported qtype");
    }
}

// Eight FP32 weight vectors covering one group in prepared-activation order.
struct GroupWeights {
    __m256 w[8];
};

__attribute__((always_inline)) inline void widen(__m256i codes, __m256* out) {
    const __m128i lo = _mm256_castsi256_si128(codes);
    const __m128i hi = _mm256_extracti128_si256(codes, 1);
    out[0]           = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
    out[1]           = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
    out[2]           = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
    out[3]           = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
}

// Expands 32 bits (one per lane) into bytes whose bit 4 is set for a one bit.
__attribute__((always_inline)) inline __m256i expand_bit4(std::uint32_t bits) {
    constexpr std::uint64_t kMask = 0x1010101010101010ULL;
    return _mm256_setr_epi64x(static_cast<long long>(_pdep_u64(bits & 0xffU, kMask)),
                              static_cast<long long>(_pdep_u64((bits >> 8) & 0xffU, kMask)),
                              static_cast<long long>(_pdep_u64((bits >> 16) & 0xffU, kMask)),
                              static_cast<long long>(_pdep_u64(bits >> 24, kMask)));
}

// Expands 32 two-bit fields into bytes holding them at bits 4..5.
__attribute__((always_inline)) inline __m256i expand_bits45(std::uint64_t fields) {
    constexpr std::uint64_t kMask = 0x3030303030303030ULL;
    return _mm256_setr_epi64x(static_cast<long long>(_pdep_u64(fields & 0xffffU, kMask)),
                              static_cast<long long>(_pdep_u64((fields >> 16) & 0xffffU, kMask)),
                              static_cast<long long>(_pdep_u64((fields >> 32) & 0xffffU, kMask)),
                              static_cast<long long>(_pdep_u64(fields >> 48, kMask)));
}

template <int Bits>
__attribute__((always_inline)) inline void decode_group(const std::uint8_t* codes,
                                                        const std::uint8_t* high,
                                                        GroupWeights& g) {
    const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes));
    if constexpr (Bits == 8) {
        widen(raw, g.w);
        return;
    } else {
        const __m256i nibble = _mm256_set1_epi8(0x0f);
        __m256i even         = _mm256_and_si256(raw, nibble);
        __m256i odd          = _mm256_and_si256(_mm256_srli_epi16(raw, 4), nibble);
        if constexpr (Bits == 5) {
            std::uint64_t h;
            std::memcpy(&h, high, sizeof(h));
            even = _mm256_or_si256(
                even, expand_bit4(static_cast<std::uint32_t>(_pext_u64(h, 0x5555555555555555ULL))));
            odd = _mm256_or_si256(
                odd, expand_bit4(static_cast<std::uint32_t>(_pext_u64(h, 0xaaaaaaaaaaaaaaaaULL))));
        } else if constexpr (Bits == 6) {
            std::uint64_t h0;
            std::uint64_t h1;
            std::memcpy(&h0, high, sizeof(h0));
            std::memcpy(&h1, high + 8, sizeof(h1));
            // Lane i owns stream bits 2i and 2i+1. Even lanes use nibble pairs 0,2,..., odd 1,3,...
            const std::uint64_t e =
                _pext_u64(h0, 0x3333333333333333ULL) | (_pext_u64(h1, 0x3333333333333333ULL) << 32);
            const std::uint64_t o =
                _pext_u64(h0, 0xccccccccccccccccULL) | (_pext_u64(h1, 0xccccccccccccccccULL) << 32);
            even = _mm256_or_si256(even, expand_bits45(e));
            odd  = _mm256_or_si256(odd, expand_bits45(o));
        }
        // Sign-extend Bits-wide two's-complement words stored in the low bits of each byte.
        const __m256i sign = _mm256_set1_epi8(static_cast<char>(1 << (Bits - 1)));
        even               = _mm256_sub_epi8(_mm256_xor_si256(even, sign), sign);
        odd                = _mm256_sub_epi8(_mm256_xor_si256(odd, sign), sign);
        widen(even, g.w);
        widen(odd, g.w + 4);
    }
}

__attribute__((always_inline)) inline float hsum(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s        = _mm_add_ps(lo, hi);
    s               = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s               = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// Computes one row against Tokens prepared activations.
template <int Bits, int Tokens>
void row_tokens(const RowSplitMatrix& m, const Codec& c, std::int32_t row, const float* prepared,
                std::int32_t prepared_stride, float* out) {
    const std::int64_t first_group = static_cast<std::int64_t>(row) * m.groups_per_row;
    const std::uint8_t* codes      = m.codes + first_group * c.code_bytes;
    const std::uint8_t* high       = c.high_bytes ? m.high + first_group * c.high_bytes : nullptr;
    const std::uint16_t* scales    = m.scales + first_group;
    constexpr int kVectors         = Bits == 8 ? 4 : 8;
    const std::int32_t logical_groups = (m.k + c.group - 1) / c.group;

    __m256 total[Tokens];
    for (int t = 0; t < Tokens; ++t) total[t] = _mm256_setzero_ps();
    for (std::int32_t group = 0; group < logical_groups; ++group) {
        GroupWeights g;
        decode_group<Bits>(codes + static_cast<std::int64_t>(group) * c.code_bytes,
                           high ? high + static_cast<std::int64_t>(group) * c.high_bytes : nullptr,
                           g);
        const __m256 scale = _mm256_set1_ps(_cvtsh_ss(scales[group]));
        for (int t = 0; t < Tokens; ++t) {
            const float* x = prepared + static_cast<std::int64_t>(t) * prepared_stride +
                             static_cast<std::int64_t>(group) * c.group;
            __m256 a = _mm256_mul_ps(g.w[0], _mm256_loadu_ps(x));
            __m256 b = _mm256_mul_ps(g.w[1], _mm256_loadu_ps(x + 8));
            a        = _mm256_fmadd_ps(g.w[2], _mm256_loadu_ps(x + 16), a);
            b        = _mm256_fmadd_ps(g.w[3], _mm256_loadu_ps(x + 24), b);
            if constexpr (kVectors == 8) {
                a = _mm256_fmadd_ps(g.w[4], _mm256_loadu_ps(x + 32), a);
                b = _mm256_fmadd_ps(g.w[5], _mm256_loadu_ps(x + 40), b);
                a = _mm256_fmadd_ps(g.w[6], _mm256_loadu_ps(x + 48), a);
                b = _mm256_fmadd_ps(g.w[7], _mm256_loadu_ps(x + 56), b);
            }
            total[t] = _mm256_fmadd_ps(_mm256_add_ps(a, b), scale, total[t]);
        }
    }
    for (int t = 0; t < Tokens; ++t) out[t] = hsum(total[t]);
}

template <int Bits>
void rows_dot_bits(const RowSplitMatrix& m, std::int32_t row_begin, std::int32_t rows,
                   const float* prepared, std::int32_t prepared_stride, std::int32_t tokens,
                   float* out, std::int32_t out_stride) {
    const Codec c = codec(m.qtype);
    for (std::int32_t r = 0; r < rows; ++r) {
        float* dst = out + static_cast<std::int64_t>(r) * out_stride;
        std::int32_t t = 0;
        // Token chunks of four keep the decoded group plus accumulators in registers.
        for (; t + 4 <= tokens; t += 4) {
            row_tokens<Bits, 4>(m, c, row_begin + r,
                                prepared + static_cast<std::int64_t>(t) * prepared_stride,
                                prepared_stride, dst + t);
        }
        switch (tokens - t) {
        case 3:
            row_tokens<Bits, 3>(m, c, row_begin + r,
                                prepared + static_cast<std::int64_t>(t) * prepared_stride,
                                prepared_stride, dst + t);
            break;
        case 2:
            row_tokens<Bits, 2>(m, c, row_begin + r,
                                prepared + static_cast<std::int64_t>(t) * prepared_stride,
                                prepared_stride, dst + t);
            break;
        case 1:
            row_tokens<Bits, 1>(m, c, row_begin + r,
                                prepared + static_cast<std::int64_t>(t) * prepared_stride,
                                prepared_stride, dst + t);
            break;
        default:
            break;
        }
    }
}

} // namespace

RowSplitMatrix row_split_matrix(const Weight& weight) {
    if (weight.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("host row-split dot: weight is not row_split_k128_v1");
    }
    const Codec c = codec(weight.qtype);
    RowSplitMatrix m;
    m.codes          = static_cast<const std::uint8_t*>(weight.qdata);
    m.high           = static_cast<const std::uint8_t*>(weight.qhigh);
    m.scales         = static_cast<const std::uint16_t*>(weight.scales);
    m.qtype          = weight.qtype;
    m.rows           = weight.n;
    m.k              = weight.k;
    m.groups_per_row = (weight.k + kKAlign - 1) / kKAlign * kKAlign / c.group;
    if (!m.codes || !m.scales || (c.high_bytes && !m.high)) {
        throw std::invalid_argument("host row-split dot: missing plane");
    }
    return m;
}

std::int32_t prepared_activation_floats(const RowSplitMatrix& m) {
    return m.groups_per_row * codec(m.qtype).group;
}

void prepare_activation(const RowSplitMatrix& m, const float* x, float* prepared) {
    const Codec c         = codec(m.qtype);
    const std::int32_t n  = prepared_activation_floats(m);
    if (c.group == 32) {
        std::copy_n(x, m.k, prepared);
        std::fill(prepared + m.k, prepared + n, 0.0F);
        return;
    }
    for (std::int32_t base = 0; base < n; base += 64) {
        for (std::int32_t j = 0; j < 32; ++j) {
            const std::int32_t even = base + 2 * j;
            const std::int32_t odd  = even + 1;
            prepared[base + j]      = even < m.k ? x[even] : 0.0F;
            prepared[base + 32 + j] = odd < m.k ? x[odd] : 0.0F;
        }
    }
}

void rows_dot(const RowSplitMatrix& m, std::int32_t row_begin, std::int32_t rows,
              const float* prepared, std::int32_t prepared_stride, std::int32_t tokens,
              float* out, std::int32_t out_stride) {
    if (row_begin < 0 || rows < 0 || row_begin + rows > m.rows || tokens <= 0) {
        throw std::invalid_argument("host row-split dot: invalid row span or token count");
    }
    switch (m.qtype) {
    case QType::Q4_G64_FP16:
        return rows_dot_bits<4>(m, row_begin, rows, prepared, prepared_stride, tokens, out,
                                out_stride);
    case QType::Q5_G64_FP16:
        return rows_dot_bits<5>(m, row_begin, rows, prepared, prepared_stride, tokens, out,
                                out_stride);
    case QType::Q6_G64_FP16:
        return rows_dot_bits<6>(m, row_begin, rows, prepared, prepared_stride, tokens, out,
                                out_stride);
    case QType::Q8_G32_FP16:
        return rows_dot_bits<8>(m, row_begin, rows, prepared, prepared_stride, tokens, out,
                                out_stride);
    default:
        throw std::invalid_argument("host row-split dot: unsupported qtype");
    }
}

void decode_row_range(const RowSplitMatrix& m, std::int32_t row, std::int32_t column,
                      std::int32_t count, float* out) {
    if (row < 0 || row >= m.rows || column < 0 || count < 0 || column + count > m.k) {
        throw std::invalid_argument("host row-split decode: invalid row or column span");
    }
    const Codec c = codec(m.qtype);
    for (std::int32_t i = 0; i < count; ++i) {
        const std::int32_t k      = column + i;
        const std::int64_t group  = static_cast<std::int64_t>(row) * m.groups_per_row + k / c.group;
        const std::int32_t lane   = k % c.group;
        const std::uint8_t* codes = m.codes + group * c.code_bytes;
        std::int32_t value;
        if (m.qtype == QType::Q8_G32_FP16) {
            value = static_cast<std::int8_t>(codes[lane]);
        } else {
            // Byte j holds lane 2j in its low nibble and lane 2j+1 in its high nibble.
            std::uint32_t word = (codes[lane >> 1] >> ((lane & 1) * 4)) & 0xfU;
            std::int32_t bits  = 4;
            if (c.high_bytes != 0) {
                const std::uint8_t* high = m.high + group * c.high_bytes;
                if (m.qtype == QType::Q5_G64_FP16) {
                    word |= ((high[lane >> 3] >> (lane & 7)) & 1U) << 4;
                    bits = 5;
                } else {
                    const std::int32_t bit = 2 * lane;
                    word |= ((high[bit >> 3] >> (bit & 7)) & 3U) << 4;
                    bits = 6;
                }
            }
            const std::uint32_t sign = 1U << (bits - 1);
            value = static_cast<std::int32_t>(word ^ sign) - static_cast<std::int32_t>(sign);
        }
        out[i] = static_cast<float>(value) * _cvtsh_ss(m.scales[group]);
    }
}

bool host_kernels_supported() noexcept {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("f16c") && __builtin_cpu_supports("bmi2");
}

} // namespace ninfer::ops::host
